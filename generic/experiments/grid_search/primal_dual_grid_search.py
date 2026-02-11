from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from generic.core.config import load_config
from generic.experiments.run_eval import run_eval


OFFLINE_SOLVER = "generic.offline.solver.OfflineMILPSolver"
ONLINE_POLICY = "generic.online.policies.PrimalDualPolicy"

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
                                yield {
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


def _apply_primal_dual_params(cfg, params: Dict[str, Any], horizon: int) -> None:
    cfg.primal_dual.eta_mode = str(params["eta_mode"])
    cfg.primal_dual.eta0 = float(params["eta0"])
    cfg.primal_dual.eta_decay = float(params["eta_decay"])
    cfg.primal_dual.eta_min = float(params["eta_min"])
    cfg.primal_dual.normalize_update = bool(params["normalize_update"])
    cfg.primal_dual.normalize_costs = bool(params["normalize_costs"])
    cfg.primal_dual.use_remaining_capacity_target = bool(params["use_remaining_capacity_target"])
    cfg.primal_dual.cost_scale_mode = str(params["cost_scale_mode"])
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"grid_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_solver_cls = _import_symbol(OFFLINE_SOLVER)
    online_policy_cls = _import_symbol(ONLINE_POLICY)

    results: List[Dict[str, Any]] = []
    for params in _iter_param_grid(
        profiles=profiles,
        eta_modes_override=eta_modes,
        eta0s_override=eta0s,
        eta_decays_override=eta_decays,
        eta_mins_override=eta_mins,
    ):
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
        results.append(row)

    best_rows = [
        row
        for row in results
        if row["valid"] and row["online_objective_mean"] is not None
    ]
    best_rows.sort(key=lambda r: float(r["online_objective_mean"]))

    summary_path = out_dir / "results.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "offline_solver": OFFLINE_SOLVER,
                "online_policy": ONLINE_POLICY,
                "horizon": int(args.horizon),
                "profiles": [p["label"] for p in profiles],
                "results": results,
                "best": best_rows[:10],
            },
            indent=2,
        )
    )
    _write_csv(out_dir / "results.csv", results)

    if best_rows:
        best_path = out_dir / "best.json"
        best_path.write_text(json.dumps(best_rows[:10], indent=2))

    print(f"Wrote primal-dual grid search results to {out_dir}")


if __name__ == "__main__":
    main()
