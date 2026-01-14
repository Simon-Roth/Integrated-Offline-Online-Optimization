from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, List

from generic.config import load_config
from generic.experiments.pipeline_registry import (
    PipelineSpec,
    default_registry,
    online_policy_needs_prices,
)
from generic.experiments.optimal_benchmark import run_optimal_benchmark
from generic.experiments.run_eval import run_eval
from generic.data.instance_generators import generate_instance_with_online
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.offline.offline_solver import OfflineMILPSolver as GenericOfflineMILPSolver


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
        description="Run multiple generic pipelines and write aggregated JSON outputs."
    )
    parser.add_argument("--config", default="configs/generic.yaml", help="Path to generic YAML.")
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=None,
        help="Pipeline names to run (defaults to all registered).",
    )
    parser.add_argument(
        "--output-dir",
        default="generic/results",
        help="Directory for aggregated JSON outputs.",
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
        default=None,
        help="Optional override for the number of online items.",
    )
    parser.add_argument(
        "--compute-optimal",
        action="store_true",
        help="Compute a full-horizon optimal benchmark once per seed and attach it to each summary.",
    )
    return parser.parse_args()


def _select_pipelines(all_specs: List[PipelineSpec], names: List[str] | None) -> List[PipelineSpec]:
    if not names:
        return all_specs
    lookup = {spec.name: spec for spec in all_specs}
    missing = [name for name in names if name not in lookup]
    if missing:
        known = ", ".join(sorted(lookup))
        raise ValueError(f"Unknown pipelines: {missing}. Known pipelines: {known}")
    return [lookup[name] for name in names]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else list(cfg.eval.seeds)

    registry = default_registry()
    specs = _select_pipelines(registry.list(), args.pipelines)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    optimal_summary = None
    if args.compute_optimal:
        optimal_summary = run_optimal_benchmark(
            cfg,
            seeds=seeds,
            horizon=args.horizon,
        )
        optimal_path = output_dir / f"optimal_full_horizon_{timestamp}.json"
        optimal_path.write_text(json.dumps(optimal_summary, indent=2))
        print(f"Wrote optimal_full_horizon -> {optimal_path}")

    for spec in specs:
        offline_solver_cls = _import_symbol(spec.offline_solver)
        online_policy_cls = _import_symbol(spec.online_policy)

        if online_policy_needs_prices(spec.online_policy):
            from binpacking.online.prices import compute_prices

            seed_for_prices = seeds[0] if seeds else cfg.eval.seeds[0]
            price_instance = generate_instance_with_online(cfg, seed=seed_for_prices, horizon=args.horizon)
            price_data = build_offline_milp_data(price_instance, cfg)
            price_solver = GenericOfflineMILPSolver(cfg)
            price_state, _ = price_solver.solve_from_data(price_data)
            price_path = Path("binpacking/results/primal_dual.json")
            compute_prices(cfg, price_instance, price_state, price_path)

        summary = run_eval(
            cfg,
            offline_solver_cls=offline_solver_cls,
            online_policy_cls=online_policy_cls,
            seeds=seeds,
            horizon=args.horizon,
            offline_solver_name=spec.offline_solver,
            online_policy_name=spec.online_policy,
        )
        summary["pipeline"] = spec.name

        output_path = output_dir / f"eval_{spec.name}_{timestamp}.json"
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote {spec.name} -> {output_path}")


if __name__ == "__main__":
    main()
