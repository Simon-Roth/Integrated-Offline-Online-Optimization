from __future__ import annotations

"""
Evaluate every registered pipeline across multiple slack fractions per scenario.

Usage:
    python -m binpacking.experiments.run_slack_sweep --slack-fractions 0.0 0.03 0.05 0.08 0.1
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Dict

from generic.config import Config
from binpacking.config import load_config
from generic.general_utils import set_global_seed
from generic.models import AssignmentState, Instance
from binpacking.data.generators import generate_instance_with_online
from binpacking.experiments.optimal_benchmark import solve_full_horizon_optimum
from binpacking.experiments.pipeline_registry import PIPELINE_REGISTRY, PIPELINES
from binpacking.experiments.pipeline_runner import PipelineSpec, run_pipeline
from binpacking.experiments.scenarios import ScenarioConfig, apply_config_overrides, select_scenarios
from binpacking.experiments.utils import save_combined_result
from generic.offline.models import OfflineSolutionInfo
from binpacking.offline.offline_heuristics.core import HeuristicSolutionInfo
from binpacking.offline.offline_solver import OfflineMILPSolver
from generic.online.state_utils import count_fallback_items


OfflineResult = Tuple[AssignmentState, OfflineSolutionInfo | HeuristicSolutionInfo]
SCENARIO_CHOICES = [scenario.name for scenario in select_scenarios(None)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipelines across multiple slack fractions.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/binpacking.yaml"),
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("binpacking/results/slack_sweep"),
        help="Root directory where scenario/slack subfolders will be created.",
    )
    parser.add_argument(
        "--slack-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        help="Slack fractions (0-1) to evaluate. 0 disables slack.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_CHOICES,
        help="Optional subset of scenario names to run.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        help="Optional subset of pipeline names. Defaults to all registered pipelines.",
    )
    return parser.parse_args()


def format_fraction(value: float) -> str:
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def validate_fractions(values: Sequence[float]) -> Tuple[float, ...]:
    unique = sorted({round(val, 6) for val in values})
    for val in unique:
        if val < 0.0 or val >= 1.0:
            raise ValueError(f"Slack fraction {val} must be in [0, 1).")
    return tuple(unique)


def select_pipeline_specs(names: Sequence[str] | None) -> Sequence[PipelineSpec]:
    if not names:
        return PIPELINES
    missing = [name for name in names if name not in PIPELINE_REGISTRY]
    if missing:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline(s) {missing}. Known pipelines: {known}")
    return [PIPELINE_REGISTRY[name] for name in names]


def run_slack_sweep(
    base_cfg: Config,
    scenarios: Iterable[ScenarioConfig],
    pipeline_specs: Sequence[PipelineSpec],
    slack_fractions: Sequence[float],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    slack_values = validate_fractions(slack_fractions)

    for scenario in scenarios:
        scenario_cfg = apply_config_overrides(base_cfg, scenario.overrides)
        if scenario.seeds:
            scenario_cfg.eval = copy.deepcopy(scenario_cfg.eval)
            scenario_cfg.eval.seeds = tuple(scenario.seeds)

        seeds = list(scenario_cfg.eval.seeds)
        scenario_dir = output_root / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Scenario: {scenario.name} ({scenario.description or 'no description'}) ===")
        print(
            f"Offline items: {scenario_cfg.problem.M_off}, "
            f"online horizon: {scenario_cfg.stoch.horizon}, "
            f"seeds: {seeds}"
        )

        shared_instances: Dict[int, Instance] = {}
        offline_cache: Dict[Tuple[str, str, int, str], OfflineResult] = {}

        for seed in seeds:
            set_global_seed(seed)
            inst = generate_instance_with_online(scenario_cfg, seed=seed)
            shared_instances[seed] = inst
            offline_count = len(inst.offline_items)
            online_count = len(inst.online_items)
            print(
                f"\nSeed {seed}: offline items={offline_count}, online items={online_count}"
            )
            if offline_count + online_count == 0:
                continue

            optimal_prefix = (
                f"pipeline_optimal_full_horizon_{scenario.name}_"
                f"N{scenario_cfg.problem.N}_Moff{offline_count}_Mon{online_count}_seed{seed}"
            )
            optimal_path = scenario_dir / f"{optimal_prefix}.json"
            if optimal_path.exists():
                optimal_summary = json.loads(optimal_path.read_text())
                fallback_opt = int(optimal_summary["final_items_in_fallback"])
                print(
                    "  OPTIMAL_FULL_HORIZON: reuse (objective="
                    f"{optimal_summary['online']['total_objective']:.3f}) -> {optimal_path}"
                )
            else:
                solver_factory = lambda cfg_: OfflineMILPSolver(
                    cfg_,
                    time_limit=300,
                    mip_gap=0.01,
                    log_to_console=False,
                )
                optimal_state, optimal_info = solve_full_horizon_optimum(
                    scenario_cfg,
                    copy.deepcopy(inst),
                    solver_factory,
                )
                fallback_opt = count_fallback_items(optimal_state, inst)
                optimal_summary = {
                    "pipeline": "OPTIMAL_FULL_HORIZON",
                    "seed": seed,
                    "problem": {
                        "N": scenario_cfg.problem.N,
                        "M_off": offline_count,
                        "M_on": online_count,
                    },
                    "offline": {
                        "method": "FullHorizonMILP",
                        "status": optimal_info.status,
                        "runtime": float(optimal_info.runtime),
                        "obj_value": float(optimal_info.obj_value),
                        "mip_gap": float(optimal_info.mip_gap),
                        "items_in_fallback": fallback_opt,
                    },
                    "online": {
                        "policy": "FullHorizonMILP",
                        "status": optimal_info.status,
                        "runtime": 0.0,
                        "total_cost": 0.0,
                        "fallback_items": fallback_opt,
                        "evicted_offline": 0,
                        "total_objective": float(optimal_info.obj_value),
                    },
                    "final_items_in_fallback": fallback_opt,
                    "offline_assignments": {
                        str(k): int(v) for k, v in optimal_state.assigned_bin.items()
                    },
                }
                optimal_path = save_combined_result(
                    optimal_prefix,
                    optimal_summary,
                    output_dir=str(scenario_dir),
                )
                print(
                    f"  OPTIMAL_FULL_HORIZON: offline {optimal_summary['offline']['runtime']:.3f}s, "
                    f"objective={optimal_summary['online']['total_objective']:.3f} -> {optimal_path}"
                )

        for fraction in slack_values:
            slack_label = format_fraction(fraction)
            cfg_slack = copy.deepcopy(scenario_cfg)
            cfg_slack.slack.enforce_slack = fraction > 0.0
            cfg_slack.slack.fraction = fraction
            slack_dir = scenario_dir / f"slack_{slack_label}"
            slack_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n-- Slack fraction {fraction:.3f} ({slack_dir}) --")

            for seed in seeds:
                instance = shared_instances[seed]
                offline_items = len(instance.offline_items)
                online_items = len(instance.online_items)
                if offline_items + online_items == 0:
                    continue

                set_global_seed(seed)
                cache_base = (scenario.name, slack_label, seed)

                for spec in pipeline_specs:
                    cache_id = spec.offline_cache_key or spec.name
                    cache_key = (*cache_base, cache_id)
                    cached_solution = offline_cache.get(cache_key)
                    if cached_solution is None and instance.offline_items:
                        solver = spec.offline_factory(cfg_slack)
                        offline_state, offline_info = solver.solve(copy.deepcopy(instance))
                        offline_cache[cache_key] = (offline_state, offline_info)
                        cached_solution = offline_cache[cache_key]

                    summary, path = run_pipeline(
                        cfg_slack,
                        spec,
                        seed=seed,
                        instance=instance,
                        offline_solution=cached_solution,
                        output_dir=str(slack_dir),
                    )
                    offline_runtime = summary["offline"]["runtime"]
                    online_runtime = summary["online"]["runtime"]
                    total_obj = summary["online"]["total_objective"]
                    fallback_final = summary["final_items_in_fallback"]
                    print(
                        f"  {spec.name}: offline {offline_runtime:.3f}s, online {online_runtime:.3f}s, "
                        f"fallback={fallback_final}, objective={total_obj:.3f} -> {path}"
                    )


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)
    scenarios = list(select_scenarios(args.scenarios))
    pipelines = select_pipeline_specs(args.pipelines)
    run_slack_sweep(base_cfg, scenarios, pipelines, args.slack_fractions, args.output_root)


if __name__ == "__main__":
    main()
