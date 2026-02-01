from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

import numpy as np

from generic.config import Config, load_config
from generic.data.instance_generators import generate_instance_with_online
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.general_utils import effective_capacity, set_global_seed
from generic.experiments.pipeline_registry import (
    ONLINE_SIM_DUAL,
    online_policy_needs_prices,
    online_policy_price_path,
)
from generic.offline.offline_solver import OfflineMILPSolver
from generic.online.online_solver import OnlineSolver
from generic.online.policies import BaseOnlinePolicy
from generic.online import state_utils


def _import_symbol(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Expected a module path like 'pkg.mod.Class', got '{path}'.")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a generic offline+online evaluation and write an aggregated JSON summary."
    )
    parser.add_argument("--config", default="configs/generic.yaml", help="Path to generic YAML.")
    parser.add_argument(
        "--offline-solver",
        default="generic.offline.offline_solver.OfflineMILPSolver",
        help="Import path to offline solver class.",
    )
    parser.add_argument(
        "--online-policy",
        required=True,
        help="Import path to online policy class.",
    )
    parser.add_argument(
        "--output-dir",
        default="generic/results",
        help="Directory for aggregated JSON output.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename (default: auto-generated).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override seed list (defaults to cfg.eval.seeds).",
    )
    parser.add_argument(
        "--m-onl",
        dest="M_onl",
        type=int,
        default=None,
        help="Optional override for the number of online items.",
    )
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _mean_or_placeholder(
    values: Iterable[float],
    placeholder: float | None = None,
) -> float | None:
    values_list = list(values)
    if not values_list:
        return placeholder
    return _mean(values_list)


def _build_run_summary(
    seed: int,
    cfg: Config,
    offline_info: Any,
    online_info: Any,
    offline_fallback: int,
    final_fallback: int,
    instance: Any,
    offline_util_per_bin: List[float] | None = None,
    *,
    offline_unplaced: int = 0,
    online_unplaced: int = 0,
    offline_fail_penalty: float = 0.0,
    online_fail_penalty: float = 0.0,
    offline_objective_penalized: float | None = None,
    online_objective_penalized: float | None = None,
    total_objective_penalized: float | None = None,
) -> Dict[str, Any]:
    summary = {
        "seed": seed,
        "problem": {
            "n": int(instance.n),
            "M_off": len(instance.offline_items),
            "M_on": len(instance.online_items),
            "m": int(instance.m),
        },
        "offline": {
            "status": offline_info.status,
            "objective": float(offline_info.obj_value),
            "runtime": float(offline_info.runtime),
            "fallback_items": int(offline_fallback),
            "unplaced_items": int(offline_unplaced),
            "failure_penalty": float(offline_fail_penalty),
            "objective_penalized": offline_objective_penalized,
        },
        "online": {
            "status": online_info.status,
            "objective": float(online_info.total_objective),
            "runtime": float(online_info.runtime),
            "fallback_items": int(online_info.fallback_items),
            "evicted_offline": int(online_info.evicted_offline),
            "unplaced_items": int(online_unplaced),
            "failure_penalty": float(online_fail_penalty),
            "objective_penalized": online_objective_penalized,
        },
        "final_items_in_fallback": int(final_fallback),
        "total_objective_penalized": total_objective_penalized,
        "slack": {
            "enforce_slack": bool(cfg.slack.enforce_slack),
            "fraction": float(cfg.slack.fraction),
            "apply_to_online": bool(getattr(cfg.slack, "apply_to_online", True)),
        },
    }
    if offline_util_per_bin is not None:
        summary["offline_util_per_bin"] = offline_util_per_bin
    return summary


def run_eval(
    cfg: Config,
    *,
    offline_solver_cls: Type[OfflineMILPSolver],
    online_policy_cls: Type[BaseOnlinePolicy],
    seeds: List[int],
    M_onl: Optional[int],
    offline_solver_name: Optional[str] = None,
    online_policy_name: Optional[str] = None,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    offline_util_means: List[float] = []
    track_offline_util = bool(getattr(cfg.eval, "track_offline_util_for_binpacking", False))
    offline_fail_statuses = {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}
    online_fail_statuses = {"INFEASIBLE"}




    for seed in seeds:
        set_global_seed(seed)
        instance = generate_instance_with_online(cfg, seed=seed, M_onl=M_onl)
        offline_solver = offline_solver_cls(cfg)
        # the following check is here because we have some binpacking specific offline algos here in the generic/pipeline_registry.py
        # These do not have a solve_from_data method (working just on A,b,c) but work on instance (A_cap, b, ...) for simplicity.
        if hasattr(offline_solver, "solve_from_data"):
            data = build_offline_milp_data(instance, cfg)
            warm_start = None
            if getattr(cfg.solver, "use_warm_start", False) and hasattr(
                offline_solver, "_generate_warm_start"
            ):
                warm_start = offline_solver._generate_warm_start(instance)
            offline_state, offline_info = offline_solver.solve_from_data(
                data, warm_start=warm_start
            )
        else:
            offline_state, offline_info = offline_solver.solve(instance)

        per_bin_util: List[float] | None = None
        if track_offline_util:
            if instance.n <= 0 or instance.m % instance.n != 0:
                raise ValueError(
                    f"track_offline_util_for_binpacking expects m = n * d. "
                    f"Got n={instance.n}, m={instance.m}."
                )
            b_eff = np.asarray(
                effective_capacity(instance.b, cfg.slack.enforce_slack, cfg.slack.fraction),
                dtype=float,
            )
            load = np.asarray(offline_state.load, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                util = np.where(b_eff > 0, load / b_eff, np.nan)
            d = instance.m // instance.n
            util_bins = util.reshape(instance.n, d)
            per_bin_mean = np.nanmean(util_bins, axis=1)
            offline_util_means.append(float(np.nanmean(per_bin_mean)))
            per_bin_util = per_bin_mean.astype(float).tolist()

        policy_path = online_policy_name
        if policy_path is None:
            policy_path = f"{online_policy_cls.__module__}.{online_policy_cls.__name__}"
        policy_kwargs: Dict[str, Any] = {}
        if policy_path and online_policy_needs_prices(policy_path):
            if policy_path == ONLINE_SIM_DUAL:
                from generic.online.generic_pricing import compute_prices
            

            price_path_str = online_policy_price_path(policy_path)
            if price_path_str is None:
                raise ValueError(f"No price output path configured for {policy_path}")
            price_path = Path(price_path_str)
            # Offset the seed to ensure pricing uses a different online sample.
            price_seed = seed + 10000
            price_samples = 1
            sample_online_caps = True
            sample_online_costs = False
            if policy_path == ONLINE_SIM_DUAL:
                price_samples = max(1, int(cfg.sim_dual.saa_samples))
                sample_online_caps = bool(cfg.sim_dual.sample_online_caps)
                sample_online_costs = bool(cfg.sim_dual.sample_online_costs)
            compute_prices(
                cfg,
                instance,
                offline_state,
                price_path,
                sample_online_caps=sample_online_caps,
                sample_online_costs=sample_online_costs,
                sample_seed=price_seed,
                num_samples=price_samples,
            )
            policy_kwargs["price_path"] = price_path

        if policy_kwargs:
            try:
                online_policy = online_policy_cls(cfg, **policy_kwargs)
            except TypeError as exc:
                raise ValueError(
                    f"{online_policy_cls.__name__} does not accept price_path but "
                    f"pricing was requested for policy '{policy_path}'."
                ) from exc
        else:
            online_policy = online_policy_cls(cfg)
        online_solver = OnlineSolver(cfg, online_policy)
        final_state, online_info = online_solver.run(instance, offline_state)

        penalty_per_item = float(getattr(cfg.costs, "fail_penalty_per_item", 0.0))
        penalty_scale = float(getattr(cfg.costs, "fail_penalty_scale", 1.0))
        penalty_per_item *= penalty_scale

        offline_unplaced = 0
        online_unplaced = 0
        offline_fail_penalty = 0.0
        online_fail_penalty = 0.0
        if penalty_per_item > 0:
            if offline_info.status in offline_fail_statuses:
                offline_unplaced = max(0, len(instance.offline_items) - len(offline_state.assigned_action))
                offline_fail_penalty = penalty_per_item * offline_unplaced
            if online_info.status in online_fail_statuses:
                online_unplaced = max(0, len(instance.online_items) - len(online_info.decisions))
                online_fail_penalty = penalty_per_item * online_unplaced

        offline_obj_base = float(offline_info.obj_value)
        if not np.isfinite(offline_obj_base):
            offline_obj_base = 0.0
        offline_obj_pen = offline_obj_base + offline_fail_penalty
        online_obj_pen = float(online_info.total_objective) + online_fail_penalty
        total_obj_pen = offline_obj_pen + online_obj_pen

        offline_fallback = state_utils.count_fallback_items(offline_state, instance)
        final_fallback = state_utils.count_fallback_items(final_state, instance)

        runs.append(
            _build_run_summary(
                seed,
                cfg,
                offline_info,
                online_info,
                offline_fallback,
                final_fallback,
                instance,
                per_bin_util,
                offline_unplaced=offline_unplaced,
                online_unplaced=online_unplaced,
                offline_fail_penalty=offline_fail_penalty,
                online_fail_penalty=online_fail_penalty,
                offline_objective_penalized=offline_obj_pen,
                online_objective_penalized=online_obj_pen,
                total_objective_penalized=total_obj_pen,
            )
        )

    offline_obj = [run["offline"]["objective"] for run in runs]
    offline_runtime = [run["offline"]["runtime"] for run in runs]
    online_obj = [run["online"]["objective"] for run in runs]
    online_runtime = [run["online"]["runtime"] for run in runs]
    total_obj_penalized_all = [run.get("total_objective_penalized") for run in runs]
    total_obj_penalized_all = [v for v in total_obj_penalized_all if v is not None]
    completed_runs = [
        run
        for run in runs
        if run["offline"]["status"] not in offline_fail_statuses
        and run["online"]["status"] not in online_fail_statuses
    ]
    completed_offline_obj = [run["offline"]["objective"] for run in completed_runs]
    completed_online_obj = [run["online"]["objective"] for run in completed_runs]
    completed_total_obj = [
        off + on for off, on in zip(completed_offline_obj, completed_online_obj)
    ]
    offline_failures = sum(
        1 for run in runs if run["offline"]["status"] in offline_fail_statuses
    )
    online_failures = sum(
        1 for run in runs if run["online"]["status"] in online_fail_statuses
    )
    offline_statuses: Dict[str, int] = {}
    online_statuses: Dict[str, int] = {}
    for run in runs:
        offline_status = run["offline"]["status"]
        online_status = run["online"]["status"]
        offline_statuses[offline_status] = offline_statuses.get(offline_status, 0) + 1
        online_statuses[online_status] = online_statuses.get(online_status, 0) + 1

    per_seed = []
    for run in runs:
        entry = {
            "seed": run["seed"],
            "offline_objective": run["offline"]["objective"],
            "online_objective": run["online"]["objective"],
            "total_objective": run["offline"]["objective"] + run["online"]["objective"],
            "offline_objective_penalized": run["offline"].get("objective_penalized"),
            "online_objective_penalized": run["online"].get("objective_penalized"),
            "total_objective_penalized": run.get("total_objective_penalized"),
            "offline_runtime": run["offline"]["runtime"],
            "online_runtime": run["online"]["runtime"],
            "offline_status": run["offline"]["status"],
            "online_status": run["online"]["status"],
        }
        if "offline_util_per_bin" in run:
            entry["offline_util_per_bin"] = run["offline_util_per_bin"]
        per_seed.append(entry)

    summary = {
        "seed_count": len(seeds),
        "offline_solver": offline_solver_name,
        "online_policy": online_policy_name,
        "offline_statuses": offline_statuses,
        "online_statuses": online_statuses,
        "per_seed": per_seed,
        "aggregate": {
            "total_objective_mean": _mean_or_placeholder(completed_total_obj),
            "total_objective_penalized_mean": _mean_or_placeholder(total_obj_penalized_all),
            "offline_objective_mean": _mean_or_placeholder(completed_offline_obj),
            "offline_runtime_mean": _mean(offline_runtime),
            "online_objective_mean": _mean_or_placeholder(completed_online_obj),
            "online_runtime_mean": _mean(online_runtime),
            "offline_failures": int(offline_failures),
            "online_failures": int(online_failures),
        },
    }
    if track_offline_util:
        summary["aggregate"]["offline_utilization_mean"] = _mean_or_placeholder(offline_util_means)
    return summary


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else list(cfg.eval.seeds)

    offline_solver_cls = _import_symbol(args.offline_solver)
    online_policy_cls = _import_symbol(args.online_policy)

    # If the online policy needs prices (e.g. sim_base), they are computed per seed.

    summary = run_eval(
        cfg,
        offline_solver_cls=offline_solver_cls,
        online_policy_cls=online_policy_cls,
        seeds=seeds,
        M_onl=args.M_onl,
        offline_solver_name=args.offline_solver,
        online_policy_name=args.online_policy,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = output_dir / args.output_name
    else:
        policy_label = args.online_policy.split(".")[-1]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{policy_label}_{timestamp}.json"

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote aggregated results to {output_path}")


if __name__ == "__main__":
    main()
