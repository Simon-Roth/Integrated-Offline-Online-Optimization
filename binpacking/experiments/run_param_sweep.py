from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, List, Sequence

from binpacking.config import load_config
from binpacking.experiments.scenarios import apply_config_overrides, select_scenarios
from generic.data.instance_generators import generate_instance_with_online
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.experiments.optimal_benchmark import run_optimal_benchmark
from generic.experiments.pipeline_registry import (
    PipelineSpec,
    default_registry,
    online_policy_needs_prices,
)
from generic.experiments.run_eval import run_eval
from generic.offline.offline_solver import OfflineMILPSolver as GenericOfflineMILPSolver


SCENARIO_CHOICES = [scenario.name for scenario in select_scenarios(None)]


def _import_symbol(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Expected a module path like 'pkg.mod.Class', got '{path}'.")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps over pipelines.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/binpacking.yaml"),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("binpacking/results/param_sweep"),
        help="Root directory for results (scenario subfolders created).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_CHOICES,
        help="Optional subset of scenarios to run (defaults to all registered).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        help="Optional subset of pipeline names (defaults to all registered).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override seed list (defaults to cfg.eval.seeds or scenario overrides).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional override for the number of online items.",
    )
    parser.add_argument(
        "--skip-optimal",
        action="store_true",
        help="Skip computing the full-horizon optimal benchmark.",
    )
    return parser.parse_args()


def _select_pipelines(all_specs: List[PipelineSpec], names: Sequence[str] | None) -> List[PipelineSpec]:
    if not names:
        return all_specs
    lookup = {spec.name: spec for spec in all_specs}
    missing = [name for name in names if name not in lookup]
    if missing:
        known = ", ".join(sorted(lookup))
        raise ValueError(f"Unknown pipelines: {missing}. Known pipelines: {known}")
    return [lookup[name] for name in names]


def _scenario_problem_meta(cfg, horizon_override: int | None) -> dict[str, Any]:
    horizon = horizon_override if horizon_override is not None else cfg.stoch.horizon
    return {
        "N": int(cfg.problem.N),
        "M_off": int(cfg.problem.M_off),
        "M_on": int(horizon),
        "dimensions": int(getattr(cfg.problem, "dimensions", 1)),
    }


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)

    registry = default_registry()
    pipeline_specs = _select_pipelines(registry.list(), args.pipelines)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    scenarios = select_scenarios(args.scenarios)

    for scenario in scenarios:
        scenario_cfg = apply_config_overrides(base_cfg, scenario.overrides)
        if scenario.seeds:
            scenario_cfg.eval.seeds = tuple(scenario.seeds)
        if args.seeds is not None:
            scenario_cfg.eval.seeds = tuple(args.seeds)
        seeds = list(scenario_cfg.eval.seeds)

        scenario_dir = output_root / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_optimal:
            optimal_summary = run_optimal_benchmark(
                scenario_cfg,
                seeds=seeds,
                horizon=args.horizon,
            )
            optimal_summary["scenario"] = scenario.name
            optimal_summary["scenario_description"] = scenario.description
            optimal_summary["problem"] = _scenario_problem_meta(scenario_cfg, args.horizon)
            optimal_path = scenario_dir / f"optimal_full_horizon_{timestamp}.json"
            optimal_path.write_text(json.dumps(optimal_summary, indent=2))

        if any(online_policy_needs_prices(spec.online_policy) for spec in pipeline_specs):
            from binpacking.online.prices import compute_prices

            seed_for_prices = seeds[0] if seeds else scenario_cfg.eval.seeds[0]
            price_instance = generate_instance_with_online(
                scenario_cfg, seed=seed_for_prices, horizon=args.horizon
            )
            price_data = build_offline_milp_data(price_instance, scenario_cfg)
            price_solver = GenericOfflineMILPSolver(scenario_cfg)
            price_state, _ = price_solver.solve_from_data(price_data)
            price_path = Path("binpacking/results/primal_dual.json")
            compute_prices(scenario_cfg, price_instance, price_state, price_path)

        for spec in pipeline_specs:
            offline_solver_cls = _import_symbol(spec.offline_solver)
            online_policy_cls = _import_symbol(spec.online_policy)

            summary = run_eval(
                scenario_cfg,
                offline_solver_cls=offline_solver_cls,
                online_policy_cls=online_policy_cls,
                seeds=seeds,
                horizon=args.horizon,
                offline_solver_name=spec.offline_solver,
                online_policy_name=spec.online_policy,
            )
            summary["pipeline"] = spec.name
            summary["scenario"] = scenario.name
            summary["scenario_description"] = scenario.description
            summary["problem"] = _scenario_problem_meta(scenario_cfg, args.horizon)

            output_path = scenario_dir / f"eval_{spec.name}_{timestamp}.json"
            output_path.write_text(json.dumps(summary, indent=2))
            print(f"Wrote {scenario.name}:{spec.name} -> {output_path}")


if __name__ == "__main__":
    main()
