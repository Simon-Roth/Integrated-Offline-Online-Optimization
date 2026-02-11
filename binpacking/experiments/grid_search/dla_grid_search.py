from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from binpacking.core.config import load_config
from generic.experiments.run_eval import run_eval


OFFLINE_SOLVER = "generic.offline.solver.OfflineMILPSolver" # just for initialisation as Moff = 0 here
ONLINE_POLICY = "binpacking.online.policies.dynamic_learning.DynamicLearningPolicy"

# Senseful parameter ranges for DLA.
HORIZONS = (60, 100, 150)
EPSILONS = (0.01, 0.05, 0.1, 0.2, 0.25,0.3)
MIN_PHASE_LENS = (1, 5, 10, 20,25,30)
USE_OFFLINE_SLACK = (True, False)


def _import_symbol(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Expected a module path like 'pkg.mod.Class', got '{path}'.")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


def _iter_param_grid() -> Iterable[Dict[str, Any]]:
    for eps in EPSILONS:
        for min_len in MIN_PHASE_LENS:
            for use_slack in USE_OFFLINE_SLACK:
                yield {
                    "epsilon": eps,
                    "min_phase_len": min_len,
                    "use_offline_slack": use_slack,
                }


def _apply_dla_params(cfg, params: Dict[str, Any]) -> None:
    cfg.dla.epsilon = float(params["epsilon"])
    cfg.dla.min_phase_len = int(params["min_phase_len"])
    cfg.dla.use_offline_slack = bool(params["use_offline_slack"])
    cfg.dla.log_prices = False
    cfg.solver.use_warm_start = False


def _is_valid_summary(summary: Dict[str, Any]) -> bool:
    agg = summary.get("aggregate", {})
    return (
        summary.get("total_objective_mean") is not None
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search DLA parameters.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/binpacking/binpacking.yaml"),
        help="Base YAML config (merged with configs/generic/generic.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/binpacking/results/dla_grid"),
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
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of online horizons to evaluate.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    if args.seeds is not None:
        base_cfg.eval.seeds = tuple(args.seeds)
    seeds = list(base_cfg.eval.seeds)

    horizons = list(HORIZONS if args.horizons is None else args.horizons)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"grid_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    combo_results: List[Dict[str, Any]] = []

    offline_solver_cls = _import_symbol(OFFLINE_SOLVER)
    online_policy_cls = _import_symbol(ONLINE_POLICY)

    for params in _iter_param_grid():
        objective_values: List[float] = []
        valid_count = 0
        for horizon in horizons:
            cfg = copy.deepcopy(base_cfg)
            _apply_dla_params(cfg, params)
            cfg.problem.T_off = 0
            cfg.stoch.T_onl = int(horizon)

            summary = run_eval(
                cfg,
                offline_solver_cls=offline_solver_cls,
                online_policy_cls=online_policy_cls,
                seeds=seeds,
                T_onl=int(horizon),
                offline_solver_name=OFFLINE_SOLVER,
                online_policy_name=ONLINE_POLICY,
            )

            row = {
                "epsilon": params["epsilon"],
                "min_phase_len": params["min_phase_len"],
                "use_offline_slack": params["use_offline_slack"],
                "horizon": int(horizon),
                "m_off": 0,
                "m_onl": int(horizon),
                "seed_count": summary.get("seed_count"),
                "total_objective_mean": summary.get("total_objective_mean"),
                "offline_failures": summary.get("aggregate", {}).get("offline_failures"),
                "online_failures": summary.get("aggregate", {}).get("online_failures"),
                "valid": _is_valid_summary(summary),
            }
            results.append(row)
            if row["valid"] and row["total_objective_mean"] is not None:
                objective_values.append(float(row["total_objective_mean"]))
                valid_count += 1

        combo_valid = valid_count == len(horizons)
        combo_mean = None
        if combo_valid and objective_values:
            combo_mean = sum(objective_values) / len(objective_values)
        combo_results.append(
            {
                "epsilon": params["epsilon"],
                "min_phase_len": params["min_phase_len"],
                "use_offline_slack": params["use_offline_slack"],
                "horizons": list(horizons),
                "horizon_count": len(horizons),
                "valid_horizon_count": valid_count,
                "total_objective_mean": combo_mean,
                "valid": combo_valid,
            }
        )

    best_rows = [
        row
        for row in combo_results
        if row["valid"] and row["total_objective_mean"] is not None
    ]
    best_rows.sort(key=lambda r: float(r["total_objective_mean"]))

    summary_path = out_dir / "results.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(args.base_config),
                "offline_solver": OFFLINE_SOLVER,
                "online_policy": ONLINE_POLICY,
                "horizons": horizons,
                "results": results,
                "combo_results": combo_results,
                "best": best_rows[:10],
            },
            indent=2,
        )
    )
    _write_csv(out_dir / "results.csv", results)
    _write_csv(out_dir / "combo_results.csv", combo_results)

    if best_rows:
        best_path = out_dir / "best.json"
        best_path.write_text(json.dumps(best_rows[:10], indent=2))

    print(f"Wrote DLA grid search results to {out_dir}")


if __name__ == "__main__":
    main()
