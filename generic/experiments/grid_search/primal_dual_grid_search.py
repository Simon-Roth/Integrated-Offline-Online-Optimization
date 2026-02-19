from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from generic.core.config import load_config
from generic.experiments.run_eval import run_eval


OFFLINE_SOLVER = "generic.offline.solver.OfflineMILPSolver"
ONLINE_POLICY = "generic.online.policies.PrimalDualPolicy"
VALID_LAMBDA0_INITS = ("offline_util", "zero", "sim_lp")

GRID_PROFILES = (
    {
        "label": "raw",
        "normalize_update": False,
        "normalize_costs": False,
        "use_remaining_capacity_target": (False, True),
        "cost_scale_modes": ("assign_mean",),
        "eta_modes": ("constant", "sqrt"),
        "eta0s": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
        "eta_decays": (0.0,),
        "eta_mins": (0.0,),
    },
    {
        "label": "norm_update",
        "normalize_update": True,
        "normalize_costs": False,
        "use_remaining_capacity_target": (False, True),
        "cost_scale_modes": ("assign_mean",),
        "eta_modes": ("constant", "sqrt", "exponential"),
        "eta0s": (0.05, 0.1, 0.2, 0.3),
        "eta_decays": (0.0, 0.001, 0.01),
        "eta_mins": (0.0, 0.01),
    },
    {
        "label": "norm_update_costs",
        "normalize_update": True,
        "normalize_costs": True,
        "use_remaining_capacity_target": (False, True),
        "cost_scale_modes": ("assign_mean", "assign_bounds"),
        "eta_modes": ("constant", "sqrt", "exponential"),
        "eta0s": (0.05, 0.1, 0.2, 0.3),
        "eta_decays": (0.0, 0.001, 0.01),
        "eta_mins": (0.0, 0.01),
    },
)


def _import_symbol(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Expected a module path like 'pkg.mod.Class', got '{path}'.")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


def _iter_param_grid(
    *,
    profiles: Sequence[Dict[str, Any]],
    eta_modes_override: Sequence[str] | None,
    eta0s_override: Sequence[float] | None,
    eta_decays_override: Sequence[float] | None,
    eta_mins_override: Sequence[float] | None,
    lambda0_inits: Sequence[str],
    pricing_num_samples: Sequence[int],
    pricing_sample_online_caps: Sequence[bool],
    default_pricing_num_samples: int,
    default_pricing_sample_online_caps: bool,
) -> Iterable[Dict[str, Any]]:
    for profile in profiles:
        eta_modes = eta_modes_override or profile["eta_modes"]
        eta0s = eta0s_override or profile["eta0s"]
        eta_decays = eta_decays_override or profile["eta_decays"]
        eta_mins = eta_mins_override or profile["eta_mins"]
        rem_targets = profile.get("use_remaining_capacity_target", (False,))
        for cost_mode in profile["cost_scale_modes"]:
            for mode in eta_modes:
                mode_lower = str(mode).lower()
                decays = eta_decays if mode_lower in ("linear", "exponential") else (0.0,)
                mins = eta_mins if mode_lower in ("linear", "exponential") else (0.0,)
                for eta0 in eta0s:
                    for decay in decays:
                        for eta_min in mins:
                            for rem_target in rem_targets:
                                base = {
                                    "profile": profile["label"],
                                    "normalize_update": profile["normalize_update"],
                                    "normalize_costs": profile["normalize_costs"],
                                    "use_remaining_capacity_target": bool(rem_target),
                                    "cost_scale_mode": str(cost_mode),
                                    "eta_mode": mode_lower,
                                    "eta0": float(eta0),
                                    "eta_decay": float(decay),
                                    "eta_min": float(eta_min),
                                }
                                for lambda0_init in lambda0_inits:
                                    init = str(lambda0_init).lower()
                                    if init == "sim_lp":
                                        for num_samples in pricing_num_samples:
                                            for sample_caps in pricing_sample_online_caps:
                                                row = dict(base)
                                                row.update(
                                                    {
                                                        "lambda0_init": init,
                                                        "pricing_num_samples": int(num_samples),
                                                        "pricing_sample_online_caps": bool(sample_caps),
                                                    }
                                                )
                                                yield row
                                    else:
                                        row = dict(base)
                                        row.update(
                                            {
                                                "lambda0_init": init,
                                                "pricing_num_samples": int(default_pricing_num_samples),
                                                "pricing_sample_online_caps": bool(default_pricing_sample_online_caps),
                                            }
                                        )
                                        yield row


def _apply_primal_dual_params(cfg, params: Dict[str, Any], horizon: int) -> None:
    cfg.primal_dual.eta_mode = str(params["eta_mode"])
    cfg.primal_dual.eta0 = float(params["eta0"])
    cfg.primal_dual.eta_decay = float(params["eta_decay"])
    cfg.primal_dual.eta_min = float(params["eta_min"])
    cfg.primal_dual.normalize_update = bool(params["normalize_update"])
    cfg.primal_dual.normalize_costs = bool(params["normalize_costs"])
    cfg.primal_dual.use_remaining_capacity_target = bool(params["use_remaining_capacity_target"])
    cfg.primal_dual.cost_scale_mode = str(params["cost_scale_mode"])
    cfg.primal_dual.lambda0_init = str(params["lambda0_init"])
    cfg.pricing_sim.num_samples = int(params["pricing_num_samples"])
    cfg.pricing_sim.sample_online_caps = bool(params["pricing_sample_online_caps"])
    cfg.problem.T_off = 0
    cfg.stoch.T_onl = int(horizon)
    cfg.solver.use_warm_start = False


def _is_valid_summary(summary: Dict[str, Any]) -> bool:
    agg = summary.get("aggregate", {})
    return (
        summary.get("aggregate", {}).get("online_objective_mean") is not None
        and int(agg.get("offline_failures", 0)) == 0
        and int(agg.get("online_failures", 0)) == 0
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _params_key(params: Dict[str, Any], horizon: int) -> Tuple[Any, ...]:
    return (
        str(params["profile"]),
        bool(params["normalize_update"]),
        bool(params["normalize_costs"]),
        bool(params["use_remaining_capacity_target"]),
        str(params["cost_scale_mode"]),
        str(params["eta_mode"]).lower(),
        float(params["eta0"]),
        float(params["eta_decay"]),
        float(params["eta_min"]),
        str(params["lambda0_init"]).lower(),
        int(params["pricing_num_samples"]),
        bool(params["pricing_sample_online_caps"]),
        int(horizon),
    )


def _row_key(
    row: Dict[str, Any],
    *,
    horizon: int,
    default_lambda0_init: str,
    default_pricing_num_samples: int,
    default_pricing_sample_online_caps: bool,
) -> Tuple[Any, ...]:
    return (
        str(row["profile"]),
        bool(row["normalize_update"]),
        bool(row["normalize_costs"]),
        bool(row["use_remaining_capacity_target"]),
        str(row["cost_scale_mode"]),
        str(row["eta_mode"]).lower(),
        float(row["eta0"]),
        float(row["eta_decay"]),
        float(row["eta_min"]),
        str(row.get("lambda0_init", default_lambda0_init)).lower(),
        int(row.get("pricing_num_samples", default_pricing_num_samples)),
        bool(row.get("pricing_sample_online_caps", default_pricing_sample_online_caps)),
        int(row.get("horizon", horizon)),
    )


def _compute_best_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_rows = [
        row
        for row in results
        if row.get("valid") and row.get("online_objective_mean") is not None
    ]
    best_rows.sort(key=lambda r: float(r["online_objective_mean"]))
    return best_rows


def _write_outputs(
    *,
    out_dir: Path,
    config_path: str,
    profiles: Sequence[Dict[str, Any]],
    lambda0_inits: Sequence[str],
    pricing_num_samples: Sequence[int],
    pricing_sample_online_caps: Sequence[bool],
    horizon: int,
    results: Sequence[Dict[str, Any]],
    total_param_count: int,
) -> None:
    ordered_results = list(results)
    best_rows = _compute_best_rows(ordered_results)

    summary_path = out_dir / "results.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "offline_solver": OFFLINE_SOLVER,
                "online_policy": ONLINE_POLICY,
                "horizon": int(horizon),
                "profiles": [p["label"] for p in profiles],
                "lambda0_inits": list(lambda0_inits),
                "pricing_num_samples": [int(v) for v in pricing_num_samples],
                "pricing_sample_online_caps": [bool(v) for v in pricing_sample_online_caps],
                "total_param_count": int(total_param_count),
                "completed_param_count": int(len(ordered_results)),
                "remaining_param_count": int(max(0, total_param_count - len(ordered_results))),
                "results": ordered_results,
                "best": best_rows[:10],
            },
            indent=2,
        )
    )
    _write_csv(out_dir / "results.csv", ordered_results)

    best_path = out_dir / "best.json"
    if best_rows:
        best_path.write_text(json.dumps(best_rows[:10], indent=2))
    elif best_path.exists():
        best_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search primal-dual MILP parameters (online-only objective)."
    )
    parser.add_argument(
        "--config",
        default="configs/generic/generic.yaml",
        help="Path to the base generic YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generic/results/primal_dual_grid"),
        help="Directory for grid search results.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help=(
            "Resume/append into this existing grid directory. If set, writes checkpoints "
            "to this directory instead of creating grid_<timestamp>."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write results.json/results.csv checkpoints after this many newly completed combos.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override seed list (defaults to cfg.eval.seeds).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=300,
        help="Single online horizon to evaluate (default: 300).",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Optional subset of profiles: raw, norm_update, norm_update_costs.",
    )
    parser.add_argument(
        "--eta-modes",
        nargs="*",
        default=None,
        help="Optional list of eta modes to evaluate.",
    )
    parser.add_argument(
        "--eta0s",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of eta0 values.",
    )
    parser.add_argument(
        "--eta-decays",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of eta_decay values.",
    )
    parser.add_argument(
        "--eta-mins",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of eta_min values.",
    )
    parser.add_argument(
        "--lambda0-inits",
        nargs="*",
        default=None,
        help="Optional list of primal_dual.lambda0_init values: offline_util, zero, sim_lp.",
    )
    parser.add_argument(
        "--pricing-num-samples",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of pricing_sim.num_samples values (used when lambda0_init=sim_lp).",
    )
    parser.add_argument(
        "--pricing-sample-online-caps",
        nargs="*",
        type=_parse_bool,
        default=None,
        help="Optional list of pricing_sim.sample_online_caps values (used when lambda0_init=sim_lp).",
    )
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'.")


def main() -> None:
    args = _parse_args()
    if int(args.checkpoint_every) < 1:
        raise ValueError("--checkpoint-every must be >= 1.")

    base_cfg = load_config(args.config)
    if args.seeds is not None:
        base_cfg.eval.seeds = tuple(args.seeds)
    seeds = list(base_cfg.eval.seeds)

    profiles = list(GRID_PROFILES)
    if args.profiles is not None:
        requested = {p.lower() for p in args.profiles}
        profiles = [p for p in profiles if p["label"] in requested]
        if not profiles:
            known = ", ".join(p["label"] for p in GRID_PROFILES)
            raise ValueError(f"Unknown profiles {requested}. Known profiles: {known}")

    eta_modes = None if args.eta_modes is None else tuple(args.eta_modes)
    eta0s = None if args.eta0s is None else tuple(args.eta0s)
    eta_decays = None if args.eta_decays is None else tuple(args.eta_decays)
    eta_mins = None if args.eta_mins is None else tuple(args.eta_mins)
    if args.lambda0_inits is None:
        lambda0_inits = VALID_LAMBDA0_INITS
    else:
        lambda0_inits = tuple(str(v).lower() for v in args.lambda0_inits)
        unknown_inits = sorted(set(lambda0_inits) - set(VALID_LAMBDA0_INITS))
        if unknown_inits:
            known = ", ".join(VALID_LAMBDA0_INITS)
            raise ValueError(
                f"Unknown lambda0_init values {unknown_inits}. Known values: {known}"
            )
    if not lambda0_inits:
        raise ValueError("At least one --lambda0-inits value is required.")

    default_pricing_num_samples = max(1, int(base_cfg.pricing_sim.num_samples))
    default_pricing_sample_online_caps = bool(base_cfg.pricing_sim.sample_online_caps)
    pricing_num_samples = (
        tuple(args.pricing_num_samples)
        if args.pricing_num_samples is not None
        else (default_pricing_num_samples,)
    )
    if not pricing_num_samples:
        raise ValueError("At least one --pricing-num-samples value is required.")
    if any(int(v) < 1 for v in pricing_num_samples):
        raise ValueError("All --pricing-num-samples values must be >= 1.")
    pricing_sample_online_caps = (
        tuple(args.pricing_sample_online_caps)
        if args.pricing_sample_online_caps is not None
        else (default_pricing_sample_online_caps,)
    )
    if not pricing_sample_online_caps:
        raise ValueError("At least one --pricing-sample-online-caps value is required.")

    if args.resume_dir is not None:
        out_dir = args.resume_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_dir / f"grid_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    param_grid = list(
        _iter_param_grid(
            profiles=profiles,
            eta_modes_override=eta_modes,
            eta0s_override=eta0s,
            eta_decays_override=eta_decays,
            eta_mins_override=eta_mins,
            lambda0_inits=lambda0_inits,
            pricing_num_samples=pricing_num_samples,
            pricing_sample_online_caps=pricing_sample_online_caps,
            default_pricing_num_samples=default_pricing_num_samples,
            default_pricing_sample_online_caps=default_pricing_sample_online_caps,
        )
    )
    param_keys = [_params_key(params, int(args.horizon)) for params in param_grid]
    expected_key_set = set(param_keys)

    results_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    existing_results_path = out_dir / "results.json"
    if existing_results_path.exists():
        try:
            existing_payload = json.loads(existing_results_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Existing resume file is not valid JSON: {existing_results_path}"
            ) from exc
        existing_rows = existing_payload.get("results", [])
        if not isinstance(existing_rows, list):
            raise ValueError(
                f"Expected 'results' list in resume file: {existing_results_path}"
            )
        for row in existing_rows:
            if not isinstance(row, dict):
                continue
            try:
                key = _row_key(
                    row,
                    horizon=int(args.horizon),
                    default_lambda0_init=str(base_cfg.primal_dual.lambda0_init).lower(),
                    default_pricing_num_samples=default_pricing_num_samples,
                    default_pricing_sample_online_caps=default_pricing_sample_online_caps,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if key in expected_key_set:
                results_by_key[key] = row

    offline_solver_cls = _import_symbol(OFFLINE_SOLVER)
    online_policy_cls = _import_symbol(ONLINE_POLICY)

    newly_completed = 0
    for params, key in zip(param_grid, param_keys):
        if key in results_by_key:
            continue
        cfg = copy.deepcopy(base_cfg)
        _apply_primal_dual_params(cfg, params, args.horizon)

        summary = run_eval(
            cfg,
            offline_solver_cls=offline_solver_cls,
            online_policy_cls=online_policy_cls,
            seeds=seeds,
            T_onl=int(args.horizon),
            offline_solver_name=OFFLINE_SOLVER,
            online_policy_name=ONLINE_POLICY,
        )

        agg = summary.get("aggregate", {})
        row = {
            "profile": params["profile"],
            "normalize_update": params["normalize_update"],
            "normalize_costs": params["normalize_costs"],
            "use_remaining_capacity_target": params["use_remaining_capacity_target"],
            "cost_scale_mode": params["cost_scale_mode"],
            "eta_mode": params["eta_mode"],
            "eta0": params["eta0"],
            "eta_decay": params["eta_decay"],
            "eta_min": params["eta_min"],
            "lambda0_init": params["lambda0_init"],
            "pricing_num_samples": params["pricing_num_samples"],
            "pricing_sample_online_caps": params["pricing_sample_online_caps"],
            "horizon": int(args.horizon),
            "m_off": 0,
            "m_onl": int(args.horizon),
            "seed_count": summary.get("seed_count"),
            "online_objective_mean": agg.get("online_objective_mean"),
            "online_runtime_mean": agg.get("online_runtime_mean"),
            "offline_failures": agg.get("offline_failures"),
            "online_failures": agg.get("online_failures"),
            "valid": _is_valid_summary(summary),
        }
        results_by_key[key] = row
        newly_completed += 1

        if newly_completed % int(args.checkpoint_every) == 0:
            ordered_rows = [results_by_key[k] for k in param_keys if k in results_by_key]
            _write_outputs(
                out_dir=out_dir,
                config_path=str(args.config),
                profiles=profiles,
                lambda0_inits=lambda0_inits,
                pricing_num_samples=pricing_num_samples,
                pricing_sample_online_caps=pricing_sample_online_caps,
                horizon=int(args.horizon),
                results=ordered_rows,
                total_param_count=len(param_grid),
            )

    ordered_rows = [results_by_key[k] for k in param_keys if k in results_by_key]
    _write_outputs(
        out_dir=out_dir,
        config_path=str(args.config),
        profiles=profiles,
        lambda0_inits=lambda0_inits,
        pricing_num_samples=pricing_num_samples,
        pricing_sample_online_caps=pricing_sample_online_caps,
        horizon=int(args.horizon),
        results=ordered_rows,
        total_param_count=len(param_grid),
    )

    print(f"Wrote primal-dual grid search results to {out_dir}")


if __name__ == "__main__":
    main()
