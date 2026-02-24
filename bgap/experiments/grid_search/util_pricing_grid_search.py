from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bgap.core.config import load_config
from generic.experiments.pipeline_registry import default_registry
from generic.experiments.run_eval import run_eval


PIPELINES = ("util+sim_dual", "util+cost_best_fit")
OFFLINE_RATIO = 0.8
TOTAL_HORIZON = 300

# Senseful parameter ranges for utilization pricing.
UPDATE_RULES = ("polynomial", "exponential")
PRICE_EXPONENTS = (0.005,0.01,0.05,0.1,0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EXP_RATES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0,5.0,6.0)
VECTOR_PRICES = (True)


def _import_symbol(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Expected a module path like 'pkg.mod.Class', got '{path}'.")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


def _compute_counts(
    m_off: int,
    ratio_offline: float,
    total_horizon: Optional[int],
) -> tuple[int, int]:
    if ratio_offline <= 0.0 or ratio_offline >= 1.0:
        raise ValueError("offline_ratio must be in (0, 1).")
    if total_horizon is None:
        m_onl = int(round(m_off * (1.0 - ratio_offline) / ratio_offline))
        return m_off, max(m_onl, 0)
    m_off = int(round(total_horizon * ratio_offline))
    m_onl = int(total_horizon - m_off)
    return max(m_off, 0), max(m_onl, 0)


def _iter_param_grid() -> Iterable[Dict[str, Any]]:
    vector_values = VECTOR_PRICES
    if isinstance(vector_values, bool):
        vector_values = (vector_values,)
    for vector_prices in vector_values:
        for update_rule in UPDATE_RULES:
            if update_rule == "polynomial":
                for exponent in PRICE_EXPONENTS:
                    yield {
                        "vector_prices": vector_prices,
                        "update_rule": update_rule,
                        "price_exponent": exponent,
                        "exp_rate": None,
                    }
            else:
                for rate in EXP_RATES:
                    yield {
                        "vector_prices": vector_prices,
                        "update_rule": update_rule,
                        "price_exponent": None,
                        "exp_rate": rate,
                    }


def _resolve_pipelines() -> Dict[str, Any]:
    registry = default_registry()
    lookup = {spec.name: spec for spec in registry.list()}
    missing = [name for name in PIPELINES if name not in lookup]
    if missing:
        known = ", ".join(sorted(lookup))
        raise ValueError(f"Unknown pipelines: {missing}. Known pipelines: {known}")
    return {name: lookup[name] for name in PIPELINES}


def _apply_pricing_params(cfg, params: Dict[str, Any]) -> None:
    cfg.util_pricing.vector_prices = bool(params["vector_prices"])
    cfg.util_pricing.update_rule = str(params["update_rule"])
    if params.get("price_exponent") is not None:
        cfg.util_pricing.price_exponent = float(params["price_exponent"])
    if params.get("exp_rate") is not None:
        cfg.util_pricing.exp_rate = float(params["exp_rate"])


def _is_valid_summary(summary: Dict[str, Any], max_online_failures: int) -> bool:
    agg = summary.get("aggregate", {})
    return (
        agg.get("total_objective_mean") is not None
        and int(agg.get("offline_failures", 0)) == 0
        and int(agg.get("online_failures", 0)) <= max_online_failures
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
    parser = argparse.ArgumentParser(description="Grid-search utilization pricing parameters.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/bgap/bgap.yaml"),
        help="Base YAML config (merged with configs/generic/generic.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bgap/results/util_pricing_grid"),
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
        "--offline-ratio",
        type=float,
        default=OFFLINE_RATIO,
        help="Fraction of offline items (T_off / (T_off + T_onl)).",
    )
    parser.add_argument(
        "--total-horizon",
        type=int,
        default=TOTAL_HORIZON,
        help="Total horizon (T_off + T_onl).",
    )
    parser.add_argument(
        "--max-online-failures",
        type=int,
        default=10,
        help="Allow up to this many online failures per pipeline.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    if args.seeds is not None:
        base_cfg.eval.seeds = tuple(args.seeds)
    seeds = list(base_cfg.eval.seeds)
    m_off, m_onl = _compute_counts(
        base_cfg.problem.T_off,
        args.offline_ratio,
        args.total_horizon,
    )
    base_cfg.problem.T_off = m_off

    pipelines = _resolve_pipelines()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"grid_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    combo_results: List[Dict[str, Any]] = []

    for params in _iter_param_grid():
        objective_values: List[float] = []
        valid_count = 0
        online_failures: List[int] = []
        offline_failures: List[int] = []
        for pipeline_name, spec in pipelines.items():
            cfg = copy.deepcopy(base_cfg)
            _apply_pricing_params(cfg, params)

            offline_solver_cls = _import_symbol(spec.offline_solver)
            online_policy_cls = _import_symbol(spec.online_policy)

            summary = run_eval(
                cfg,
                offline_solver_cls=offline_solver_cls,
                online_policy_cls=online_policy_cls,
                seeds=seeds,
                T_onl=m_onl,
                offline_solver_name=spec.offline_solver,
                online_policy_name=spec.online_policy,
            )
            agg = summary.get("aggregate", {})
            offline_failure_count = agg.get("offline_failures")
            online_failure_count = agg.get("online_failures")

            row = {
                "pipeline": pipeline_name,
                "update_rule": params["update_rule"],
                "price_exponent": params.get("price_exponent"),
                "exp_rate": params.get("exp_rate"),
                "vector_prices": params["vector_prices"],
                "offline_ratio": args.offline_ratio,
                "total_horizon": args.total_horizon,
                "m_onl": m_onl,
                "seed_count": summary.get("seed_count"),
                "total_objective_mean": summary.get("aggregate", {}).get("total_objective_mean"),
                "offline_failures": offline_failure_count,
                "online_failures": online_failure_count,
                "valid": _is_valid_summary(summary, args.max_online_failures),
            }
            results.append(row)
            if row["valid"] and row["total_objective_mean"] is not None:
                objective_values.append(float(row["total_objective_mean"]))
                valid_count += 1
            if offline_failure_count is not None:
                offline_failures.append(int(offline_failure_count))
            if online_failure_count is not None:
                online_failures.append(int(online_failure_count))

        combo_valid = valid_count == len(pipelines)
        combo_mean = None
        if combo_valid and objective_values:
            combo_mean = sum(objective_values) / len(objective_values)
        online_failures_total = sum(online_failures) if online_failures else None
        online_failures_max = max(online_failures) if online_failures else None
        offline_failures_total = sum(offline_failures) if offline_failures else None
        offline_failures_max = max(offline_failures) if offline_failures else None
        combo_results.append(
            {
                "update_rule": params["update_rule"],
                "price_exponent": params.get("price_exponent"),
                "exp_rate": params.get("exp_rate"),
                "vector_prices": params["vector_prices"],
                "offline_ratio": args.offline_ratio,
                "total_horizon": args.total_horizon,
                "m_onl": m_onl,
                "pipeline_count": len(pipelines),
                "valid_pipeline_count": valid_count,
                "online_failures_total": online_failures_total,
                "online_failures_max": online_failures_max,
                "offline_failures_total": offline_failures_total,
                "offline_failures_max": offline_failures_max,
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
                "offline_ratio": args.offline_ratio,
                "total_horizon": args.total_horizon,
                "m_onl": m_onl,
                "max_online_failures": args.max_online_failures,
                "pipelines": list(pipelines),
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

    print(f"Wrote grid search results to {out_dir}")


if __name__ == "__main__":
    main()
