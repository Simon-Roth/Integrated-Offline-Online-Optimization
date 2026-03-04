from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

from generic.core.config import Config, load_config as load_generic_config
from generic.core.models import AssignmentState, Decision, Instance, StepSpec
from generic.core.utils import feasible_option_indices, option_is_feasible, set_global_seed
from generic.data.instance_generators import BaseInstanceGenerator
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.offline.solver import OfflineMILPSolver
from generic.online import state_utils
from generic.online.policies import (
    BaseOnlinePolicy,
    PolicyInfeasibleError,
    PrimalDualPolicy,
    SimDualPolicy,
)
from generic.online.policy_utils import current_cost_row, lookup_assignment_cost, remaining_capacities


POLICY_ALIASES: Dict[str, str] = {
    "sim_dual": "generic.online.policies.SimDualPolicy",
    "primal_dual": "generic.online.policies.PrimalDualPolicy",
}


def _import_symbol(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Expected a module path like 'pkg.mod.Class', got '{path}'.")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


def _policy_accepts_pricing_seed(policy_cls: Type[BaseOnlinePolicy]) -> bool:
    try:
        params = inspect.signature(policy_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    if "pricing_sample_seed" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _is_bgap_override_path(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return "/configs/bgap/" in normalized or path.name.lower().startswith("bgap")


def _load_config(path: Path, mode: str) -> Config:
    if mode == "generic":
        return load_generic_config(path)
    if mode == "bgap":
        from bgap.core.config import load_config as load_bgap_config

        return load_bgap_config(path)
    if mode != "auto":
        raise ValueError(f"Unknown --config-loader value: {mode}")

    if _is_bgap_override_path(path):
        from bgap.core.config import load_config as load_bgap_config

        return load_bgap_config(path)
    return load_generic_config(path)


def _resolve_policy_path(token: str) -> str:
    key = token.strip().lower()
    return POLICY_ALIASES.get(key, token)


def _dedupe_preserve(values: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_scales(values: Sequence[float]) -> List[float]:
    if not values:
        raise ValueError("At least one --price-scales value is required.")
    out = _dedupe_preserve(round(float(v), 12) for v in values)
    return [float(v) for v in out]


def _has_scale(scales: Sequence[float], target: float, *, atol: float = 1e-12) -> bool:
    return any(abs(float(s) - float(target)) <= atol for s in scales)


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=float)))


def _sum_or_zero(values: Iterable[int | float | None]) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        total += float(value)
    return total


def _default_offline_solver_path(cfg: Config) -> str:
    mode = str(getattr(getattr(cfg, "generation", None), "generator", "generic")).lower()
    if mode == "bgap":
        return "bgap.offline.solver.OfflineMILPSolver"
    return "generic.offline.solver.OfflineMILPSolver"


def _solve_offline(
    cfg: Config,
    instance: Instance,
    offline_solver_cls: Type[OfflineMILPSolver],
) -> Tuple[AssignmentState, Any]:
    solver = offline_solver_cls(cfg)
    if hasattr(solver, "solve_from_data"):
        data = build_offline_milp_data(instance, cfg)
        warm_start = None
        if getattr(cfg.solver, "use_warm_start", False) and hasattr(solver, "_generate_warm_start"):
            warm_start = solver._generate_warm_start(instance)
        return solver.solve_from_data(data, warm_start=warm_start)
    return solver.solve(instance)


def _instantiate_policy(
    policy_cls: Type[BaseOnlinePolicy],
    cfg: Config,
    *,
    pricing_seed: int,
) -> BaseOnlinePolicy:
    if _policy_accepts_pricing_seed(policy_cls):
        return policy_cls(cfg, pricing_sample_seed=pricing_seed)
    return policy_cls(cfg)


def _extract_pd_delegate(policy: BaseOnlinePolicy) -> Optional[PrimalDualPolicy]:
    if isinstance(policy, PrimalDualPolicy):
        return policy
    delegate = getattr(policy, "_delegate", None)
    if isinstance(delegate, PrimalDualPolicy):
        return delegate
    return None


def _fallback_decision(cfg: Config, instance: Instance, step: StepSpec) -> Decision | None:
    if not bool(cfg.problem.fallback_is_enabled):
        return None
    if not bool(cfg.problem.fallback_allowed_online):
        return None
    fallback_idx = int(instance.fallback_option_index)
    if fallback_idx < 0:
        return None
    if not option_is_feasible(step.feas_matrix, step.feas_rhs, fallback_idx):
        return None
    return Decision(
        placed_step=(step.step_id, fallback_idx),
        evicted_offline_steps=[],
        reassigned_offline_steps=[],
        incremental_cost=lookup_assignment_cost(cfg, instance, step.step_id, fallback_idx),
    )


def _capacity_fit_indices(
    cap_matrix: np.ndarray,
    option_ids: Sequence[int],
    capacities: np.ndarray,
    *,
    tol: float = 1e-9,
) -> List[int]:
    fit: List[int] = []
    for option_id in option_ids:
        if option_id < 0 or option_id >= cap_matrix.shape[1]:
            continue
        if np.all(cap_matrix[:, option_id] <= capacities + tol):
            fit.append(int(option_id))
    return fit


def _step_counterfactual_diag(
    pd_delegate: PrimalDualPolicy,
    cfg: Config,
    step: StepSpec,
    state: AssignmentState,
    instance: Instance,
) -> Dict[str, Any]:
    n = int(instance.n)
    fallback_idx = int(instance.fallback_option_index)
    cols = n + (1 if fallback_idx >= 0 else 0)
    cap_matrix = np.asarray(step.cap_matrix, dtype=float)
    feas_matrix = np.asarray(step.feas_matrix, dtype=float)
    feas_rhs = np.asarray(step.feas_rhs, dtype=float)
    costs = current_cost_row(cfg, instance, step.step_id, cols).reshape(1, -1)

    effective_caps = getattr(pd_delegate, "_effective_caps", None)
    capacities = remaining_capacities(cfg, state, instance, effective_caps)
    target = pd_delegate._current_target(capacities)

    costs_unpriced = costs.copy()
    if bool(cfg.primal_dual.normalize_costs):
        costs_unpriced = costs_unpriced / float(pd_delegate._cost_scale)
    costs_priced = pd_delegate._price_aware_costs(costs.copy(), cap_matrix, n, target)

    price_terms = np.zeros((n,), dtype=float)
    if n > 0:
        price_terms = np.asarray(pd_delegate._price_terms(cap_matrix, target), dtype=float).reshape(-1)

    priced_choice: int | None
    unpriced_choice: int | None
    try:
        priced_choice = int(
            pd_delegate._solve_price_aware_milp(
                cap_matrix,
                costs_priced.copy(),
                feas_matrix,
                feas_rhs,
                capacities,
                fallback_idx,
            )
        )
    except PolicyInfeasibleError:
        priced_choice = None

    try:
        unpriced_choice = int(
            pd_delegate._solve_price_aware_milp(
                cap_matrix,
                costs_unpriced.copy(),
                feas_matrix,
                feas_rhs,
                capacities,
                fallback_idx,
            )
        )
    except PolicyInfeasibleError:
        unpriced_choice = None

    feasible_regular = feasible_option_indices(feas_matrix, feas_rhs, option_ids=range(n))
    fit_regular = _capacity_fit_indices(cap_matrix, feasible_regular, capacities)

    raw_gap: float | None = None
    price_spread: float | None = None
    price_abs_mean: float | None = None
    cost_abs_mean: float | None = None
    price_to_cost_ratio: float | None = None
    spread_gap_ratio: float | None = None
    has_nonzero_price_on_fit_regular = False
    if fit_regular:
        fit_costs = np.asarray(costs_unpriced[0, fit_regular], dtype=float)
        fit_price_terms = np.asarray(price_terms[fit_regular], dtype=float)
        sorted_costs = np.sort(fit_costs)
        if sorted_costs.size >= 2:
            raw_gap = float(sorted_costs[1] - sorted_costs[0])
        price_spread = float(np.max(fit_price_terms) - np.min(fit_price_terms))
        price_abs_mean = float(np.mean(np.abs(fit_price_terms)))
        cost_abs_mean = float(np.mean(np.abs(fit_costs)))
        denom = max(cost_abs_mean, 1e-12)
        price_to_cost_ratio = float(price_abs_mean / denom)
        if raw_gap is not None:
            spread_gap_ratio = float(price_spread / max(raw_gap, 1e-12))
        has_nonzero_price_on_fit_regular = bool(np.any(np.abs(fit_price_terms) > 1e-12))

    lam = np.asarray(getattr(pd_delegate, "_lambda", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
    lambda_zero_share = float(np.mean(np.isclose(lam, 0.0, atol=1e-12))) if lam.size else 1.0
    lambda_l1_mean = float(np.mean(np.abs(lam))) if lam.size else 0.0
    lambda_linf = float(np.max(np.abs(lam))) if lam.size else 0.0
    direct_flip = (
        priced_choice is not None
        and unpriced_choice is not None
        and int(priced_choice) != int(unpriced_choice)
    )

    return {
        "step_id": int(step.step_id),
        "priced_choice": priced_choice,
        "unpriced_choice": unpriced_choice,
        "direct_flip": bool(direct_flip),
        "lambda_zero_share": float(lambda_zero_share),
        "lambda_l1_mean": float(lambda_l1_mean),
        "lambda_linf": float(lambda_linf),
        "fit_regular_count": int(len(fit_regular)),
        "nonzero_price_on_fit_regular": bool(has_nonzero_price_on_fit_regular),
        "raw_gap": raw_gap,
        "price_spread": price_spread,
        "price_abs_mean": price_abs_mean,
        "cost_abs_mean": cost_abs_mean,
        "price_to_cost_ratio": price_to_cost_ratio,
        "spread_gap_ratio": spread_gap_ratio,
    }


def _run_online_with_diagnostics(
    cfg: Config,
    instance: Instance,
    initial_state: AssignmentState,
    *,
    policy_cls: Type[BaseOnlinePolicy],
    pricing_seed: int,
    max_steps: int | None,
) -> Dict[str, Any]:
    state = state_utils.clone_state(initial_state)
    policy = _instantiate_policy(policy_cls, cfg, pricing_seed=pricing_seed)
    policy.begin_instance(instance, state)
    cap_lookup = state_utils.build_cap_lookup(instance)

    stop_on_first_failure = bool(getattr(cfg.costs, "stop_online_on_first_failure", True))
    had_infeasible_step = False
    decisions: List[int] = []
    step_diags: List[Dict[str, Any]] = []
    total_objective = 0.0
    max_steps_effective = max_steps if max_steps is not None and max_steps > 0 else None

    start = time.perf_counter()
    for pos, step in enumerate(instance.online_steps):
        if max_steps_effective is not None and pos >= max_steps_effective:
            break

        pd_delegate = _extract_pd_delegate(policy)
        if pd_delegate is not None:
            step_diags.append(_step_counterfactual_diag(pd_delegate, cfg, step, state, instance))

        decision: Decision | None
        try:
            decision = policy.select_action(step, state, instance)
        except PolicyInfeasibleError:
            decision = None

        needs_fallback = decision is None or (
            not bool(cfg.problem.allow_reassignment)
            and (
                len(decision.evicted_offline_steps) > 0
                or len(decision.reassigned_offline_steps) > 0
            )
        )
        if needs_fallback:
            decision = _fallback_decision(cfg, instance, step)
            if decision is None:
                had_infeasible_step = True
                if stop_on_first_failure:
                    break
                continue

        state_utils.apply_decision(decision, step, state, instance, cap_lookup)
        decisions.append(int(decision.placed_step[1]))
        total_objective += float(decision.incremental_cost)

    runtime = time.perf_counter() - start
    status = "INFEASIBLE" if had_infeasible_step else "COMPLETED"

    eval_steps = [
        row
        for row in step_diags
        if row["priced_choice"] is not None and row["unpriced_choice"] is not None
    ]
    direct_flips = sum(1 for row in eval_steps if row["direct_flip"])
    fit_regular_steps = sum(1 for row in step_diags if int(row["fit_regular_count"]) > 0)
    nonzero_price_steps = sum(1 for row in step_diags if bool(row["nonzero_price_on_fit_regular"]))
    spread_gt_gap_steps = sum(
        1
        for row in step_diags
        if row["price_spread"] is not None
        and row["raw_gap"] is not None
        and float(row["price_spread"]) > float(row["raw_gap"])
    )

    return {
        "status": status,
        "runtime": float(runtime),
        "objective": float(total_objective),
        "decisions": decisions,
        "steps_processed": int(len(decisions)),
        "steps_profiled": int(len(step_diags)),
        "steps_profiled_with_priced_and_unpriced_choice": int(len(eval_steps)),
        "direct_flip_steps": int(direct_flips),
        "direct_flip_rate": (
            float(direct_flips) / float(len(eval_steps)) if eval_steps else None
        ),
        "fit_regular_steps": int(fit_regular_steps),
        "nonzero_price_on_fit_regular_steps": int(nonzero_price_steps),
        "nonzero_price_step_share": (
            float(nonzero_price_steps) / float(fit_regular_steps) if fit_regular_steps else None
        ),
        "spread_gt_gap_steps": int(spread_gt_gap_steps),
        "spread_gt_gap_share": (
            float(spread_gt_gap_steps) / float(fit_regular_steps) if fit_regular_steps else None
        ),
        "lambda_zero_share_mean": _mean_or_none(row["lambda_zero_share"] for row in step_diags),
        "lambda_l1_mean": _mean_or_none(row["lambda_l1_mean"] for row in step_diags),
        "lambda_linf_mean": _mean_or_none(row["lambda_linf"] for row in step_diags),
        "price_to_cost_ratio_mean": _mean_or_none(row["price_to_cost_ratio"] for row in step_diags),
        "spread_gap_ratio_mean": _mean_or_none(row["spread_gap_ratio"] for row in step_diags),
        "price_abs_mean": _mean_or_none(row["price_abs_mean"] for row in step_diags),
        "cost_abs_mean": _mean_or_none(row["cost_abs_mean"] for row in step_diags),
        "fallback_steps_final_state": int(state_utils.count_fallback_steps(state, instance)),
    }


def _apply_price_scale(cfg: Config, policy_cls: Type[BaseOnlinePolicy], scale: float) -> Config:
    out = copy.deepcopy(cfg)
    factor = float(scale)
    if factor < 0.0:
        raise ValueError(f"Price scale must be >= 0, got {scale}.")

    if issubclass(policy_cls, SimDualPolicy):
        out.primal_dual.sim_lp_init_scale = float(out.primal_dual.sim_lp_init_scale) * factor
        return out

    if issubclass(policy_cls, PrimalDualPolicy):
        mode = str(out.primal_dual.lambda0_init).lower()
        if mode == "sim_lp":
            out.primal_dual.sim_lp_init_scale = float(out.primal_dual.sim_lp_init_scale) * factor
        elif mode == "offline_util":
            init_scale = out.primal_dual.offline_util_init_scale
            if init_scale is None:
                raise ValueError(
                    "primal_dual.offline_util_init_scale must be set when lambda0_init='offline_util'."
                )
            out.primal_dual.offline_util_init_scale = float(init_scale) * factor
        out.primal_dual.eta0 = float(out.primal_dual.eta0) * factor
        out.primal_dual.eta_min = float(out.primal_dual.eta_min) * factor
    return out


def _compare_trajectories(base: Sequence[int], other: Sequence[int]) -> Dict[str, Any]:
    compared = min(len(base), len(other))
    if compared > 0:
        diff = sum(1 for idx in range(compared) if int(base[idx]) != int(other[idx]))
        diff_rate = float(diff) / float(compared)
    else:
        diff = 0
        diff_rate = None
    return {
        "compared_steps": int(compared),
        "diff_steps": int(diff),
        "diff_rate": diff_rate,
        "length_base": int(len(base)),
        "length_other": int(len(other)),
        "length_delta": int(len(other) - len(base)),
    }


def _aggregate_runs(
    seed_runs: List[Dict[str, Any]],
    *,
    policy_path: str,
    scale: float,
) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    for row in seed_runs:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1

    total_profiled = int(_sum_or_zero(row["steps_profiled"] for row in seed_runs))
    total_eval = int(
        _sum_or_zero(row["steps_profiled_with_priced_and_unpriced_choice"] for row in seed_runs)
    )
    total_flips = int(_sum_or_zero(row["direct_flip_steps"] for row in seed_runs))
    total_fit_regular = int(_sum_or_zero(row["fit_regular_steps"] for row in seed_runs))
    total_nonzero_price = int(_sum_or_zero(row["nonzero_price_on_fit_regular_steps"] for row in seed_runs))
    total_spread_gt_gap = int(_sum_or_zero(row["spread_gt_gap_steps"] for row in seed_runs))

    return {
        "policy": policy_path,
        "price_scale": float(scale),
        "seed_count": int(len(seed_runs)),
        "status_counts": statuses,
        "objective_mean": _mean_or_none(row["objective"] for row in seed_runs),
        "runtime_mean": _mean_or_none(row["runtime"] for row in seed_runs),
        "steps_processed_mean": _mean_or_none(row["steps_processed"] for row in seed_runs),
        "steps_profiled_total": total_profiled,
        "steps_profiled_with_priced_and_unpriced_choice_total": total_eval,
        "direct_flip_steps_total": total_flips,
        "direct_flip_rate_weighted": (
            float(total_flips) / float(total_eval) if total_eval > 0 else None
        ),
        "fit_regular_steps_total": total_fit_regular,
        "nonzero_price_on_fit_regular_steps_total": total_nonzero_price,
        "nonzero_price_step_share_weighted": (
            float(total_nonzero_price) / float(total_fit_regular) if total_fit_regular > 0 else None
        ),
        "spread_gt_gap_steps_total": total_spread_gt_gap,
        "spread_gt_gap_share_weighted": (
            float(total_spread_gt_gap) / float(total_fit_regular) if total_fit_regular > 0 else None
        ),
        "lambda_zero_share_mean": _mean_or_none(row["lambda_zero_share_mean"] for row in seed_runs),
        "lambda_l1_mean": _mean_or_none(row["lambda_l1_mean"] for row in seed_runs),
        "lambda_linf_mean": _mean_or_none(row["lambda_linf_mean"] for row in seed_runs),
        "price_to_cost_ratio_mean": _mean_or_none(row["price_to_cost_ratio_mean"] for row in seed_runs),
        "spread_gap_ratio_mean": _mean_or_none(row["spread_gap_ratio_mean"] for row in seed_runs),
        "price_abs_mean": _mean_or_none(row["price_abs_mean"] for row in seed_runs),
        "cost_abs_mean": _mean_or_none(row["cost_abs_mean"] for row in seed_runs),
        "by_seed": seed_runs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether SimDual/PrimalDual prices influence online choices via "
            "local counterfactual flips and scale ablations."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generic/generic.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--config-loader",
        type=str,
        default="auto",
        choices=["auto", "generic", "bgap"],
        help="Config loader mode.",
    )
    parser.add_argument(
        "--offline-solver",
        type=str,
        default=None,
        help="Optional import path for offline solver class.",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=["sim_dual", "primal_dual"],
        help=(
            "Policies to test (aliases: sim_dual, primal_dual) or full import paths."
        ),
    )
    parser.add_argument(
        "--price-scales",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 10.0],
        help="Scale factors applied to policy price terms.",
    )
    parser.add_argument(
        "--baseline-scale",
        type=float,
        default=1.0,
        help="Reference scale for trajectory comparisons.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional seed override (defaults to cfg.eval.seeds).",
    )
    parser.add_argument(
        "--m-onl",
        dest="T_onl",
        type=int,
        default=None,
        help="Optional override for generated online horizon.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max number of online steps to profile per seed.",
    )
    parser.add_argument(
        "--pricing-seed-offset",
        type=int,
        default=10000,
        help="Offset added to base seed for deterministic pricing samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generic/results/price_effect"),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output file name.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_config(args.config, args.config_loader)
    seeds = list(args.seeds) if args.seeds is not None else list(cfg.eval.seeds)
    if not seeds:
        raise ValueError("No seeds provided and config eval.seeds is empty.")

    policy_paths = [_resolve_policy_path(item) for item in args.policies]
    policy_paths = _dedupe_preserve(policy_paths)
    if not policy_paths:
        raise ValueError("At least one policy is required.")

    scales = _normalize_scales(args.price_scales)
    if not _has_scale(scales, args.baseline_scale):
        scales.append(float(args.baseline_scale))
    scales = _normalize_scales(scales)

    offline_solver_path = args.offline_solver or _default_offline_solver_path(cfg)
    offline_solver_cls = _import_symbol(offline_solver_path)
    generator = BaseInstanceGenerator.from_config(cfg)

    # Generate instances + offline states once per seed.
    seed_data: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        set_global_seed(seed)
        instance = generator.generate_full_instance(cfg, seed=seed, T_onl=args.T_onl)
        offline_state, offline_info = _solve_offline(cfg, instance, offline_solver_cls)
        seed_data[int(seed)] = {
            "instance": instance,
            "offline_state": offline_state,
            "offline_status": str(offline_info.status),
            "offline_objective": float(offline_info.obj_value),
        }

    policy_cls_map: Dict[str, Type[BaseOnlinePolicy]] = {}
    for policy_path in policy_paths:
        policy_cls = _import_symbol(policy_path)
        required_methods = ("begin_instance", "select_action")
        if not all(callable(getattr(policy_cls, method, None)) for method in required_methods):
            raise TypeError(
                f"{policy_path} does not implement required policy methods: "
                f"{', '.join(required_methods)}."
            )
        policy_cls_map[policy_path] = policy_cls

    run_rows: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    aggregated_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for policy_path in policy_paths:
        policy_cls = policy_cls_map[policy_path]
        for scale in scales:
            scaled_cfg = _apply_price_scale(cfg, policy_cls, scale)
            per_seed_rows: List[Dict[str, Any]] = []
            for seed in seeds:
                data = seed_data[int(seed)]
                instance = data["instance"]
                initial_state = data["offline_state"]
                run = _run_online_with_diagnostics(
                    scaled_cfg,
                    instance,
                    initial_state,
                    policy_cls=policy_cls,
                    pricing_seed=int(seed) + int(args.pricing_seed_offset),
                    max_steps=args.max_steps,
                )
                row = {
                    "seed": int(seed),
                    "offline_status": data["offline_status"],
                    "offline_objective": data["offline_objective"],
                    **run,
                }
                per_seed_rows.append(row)
                run_rows[(policy_path, float(scale), int(seed))] = row
            aggregated_rows.append(
                _aggregate_runs(
                    per_seed_rows,
                    policy_path=policy_path,
                    scale=scale,
                )
            )

    # Trajectory deltas versus baseline scale for each policy.
    trajectory_comparisons: List[Dict[str, Any]] = []
    baseline_scale = float(args.baseline_scale)
    for policy_path in policy_paths:
        if not _has_scale(scales, baseline_scale):
            continue
        for scale in scales:
            if abs(float(scale) - baseline_scale) <= 1e-12:
                continue
            per_seed: List[Dict[str, Any]] = []
            for seed in seeds:
                base_row = run_rows[(policy_path, baseline_scale, int(seed))]
                other_row = run_rows[(policy_path, float(scale), int(seed))]
                cmp_row = _compare_trajectories(
                    base_row["decisions"],
                    other_row["decisions"],
                )
                cmp_row["seed"] = int(seed)
                cmp_row["base_status"] = base_row["status"]
                cmp_row["other_status"] = other_row["status"]
                per_seed.append(cmp_row)

            total_compared = int(_sum_or_zero(row["compared_steps"] for row in per_seed))
            total_diff = int(_sum_or_zero(row["diff_steps"] for row in per_seed))
            seeds_any_diff = int(sum(1 for row in per_seed if int(row["diff_steps"]) > 0))
            trajectory_comparisons.append(
                {
                    "policy": policy_path,
                    "baseline_scale": baseline_scale,
                    "compared_scale": float(scale),
                    "seed_count": int(len(seeds)),
                    "seeds_with_any_diff": seeds_any_diff,
                    "compared_steps_total": total_compared,
                    "diff_steps_total": total_diff,
                    "diff_rate_weighted": (
                        float(total_diff) / float(total_compared) if total_compared > 0 else None
                    ),
                    "by_seed": per_seed,
                }
            )

    duration = time.perf_counter() - start
    payload = {
        "config": str(args.config),
        "config_loader": args.config_loader,
        "offline_solver": offline_solver_path,
        "policies": policy_paths,
        "price_scales": scales,
        "baseline_scale": baseline_scale,
        "seeds": seeds,
        "T_onl_override": args.T_onl,
        "max_steps": args.max_steps,
        "pricing_seed_offset": int(args.pricing_seed_offset),
        "runtime_sec": float(duration),
        "diagnostics": aggregated_rows,
        "trajectory_comparisons": trajectory_comparisons,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = output_dir / args.output_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"price_effect_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"Wrote price-effect diagnostics to {output_path}")
    for row in aggregated_rows:
        print(
            f"{row['policy']} scale={row['price_scale']:.6g} "
            f"flip_rate={row['direct_flip_rate_weighted']} "
            f"price_to_cost={row['price_to_cost_ratio_mean']} "
            f"lambda_zero_share={row['lambda_zero_share_mean']}"
        )
    for row in trajectory_comparisons:
        print(
            f"{row['policy']} scale {row['compared_scale']:.6g} vs {baseline_scale:.6g}: "
            f"choice_diff_rate={row['diff_rate_weighted']}"
        )


if __name__ == "__main__":
    main()
