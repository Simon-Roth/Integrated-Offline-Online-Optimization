from __future__ import annotations

"""
Simple runner: iterate scenarios × (optional) pipelines × seeds, without extra sweep axes.
Scenarios encode the variations (volumes, graphs, load) themselves.
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

GOOD_STATUSES = {"COMPLETED", "OPTIMAL", "NO_ITEMS"}


# ---- Core sweep logic -------------------------------------------------------

def _scenario_family_variant(name: str) -> tuple[str, str]:
    """
    Derive (family, variant) from a scenario name.

    Convention in `experiments/scenarios.py`:
    - Most scenarios end with a ratio suffix like "off20_on80".
      We treat everything before that suffix as the family.
    """
    marker = "_off"
    idx = name.rfind(marker)
    if idx > 0:
        return name[:idx], name[idx + 1 :]
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, name


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
        help="Root directory for results (scenario/variant subfolders created).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[scenario.name for scenario in select_scenarios(None)],
        help="Optional subset of scenarios to run (defaults to all registered).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        help="Optional subset of pipeline names (defaults to all registered).",
    )
    parser.add_argument(
        "--compute-optimal",
        action="store_true",
        help="If set, solve and store the full-horizon optimum for relative plots.",
    )
    parser.add_argument(
        "--skip-ratio-plot",
        action="store_true",
        help="Skip generating the ratio line plot after the sweep finishes.",
    )
    parser.add_argument(
        "--ratio-plot-metric",
        choices=["objective_ratio", "total_objective", "runtime_total"],
        default="objective_ratio",
        help="Metric to plot against offline ratio in the generated line chart.",
    )
    return parser.parse_args()


def _select_pipelines(names: Sequence[str] | None) -> Sequence[PipelineSpec]:
    if not names:
        return PIPELINES
    missing = [name for name in names if name not in PIPELINE_REGISTRY]
    if missing:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline(s) {missing}. Known pipelines: {known}")
    return [PIPELINE_REGISTRY[name] for name in names]


def _compute_optimal(
    cfg: Config,
    instance: Instance,
    seed: int,
    output_dir: Path,
    label: str,
    *,
    scenario_name: str | None = None,
    scenario_description: str | None = None,
) -> Dict[str, Any]:
    offline_count = len(instance.offline_items)
    online_count = len(instance.online_items)
    if offline_count + online_count == 0:
        raise ValueError("No offline or online items to solve optimally.")
    prefix = (
        f"pipeline_optimal_full_horizon_{label}_"
        f"N{cfg.problem.N}_Moff{offline_count}_Mon{online_count}_seed{seed}"
    )
    opt_path = output_dir / f"{prefix}.json"
    if opt_path.exists():
        return json.loads(opt_path.read_text())

    solver_factory = lambda cfg_: OfflineMILPSolver(
        cfg_,
        time_limit=300,
        mip_gap=0.01,
        log_to_console=False,
    )
    optimal_state, optimal_info = solve_full_horizon_optimum(
        cfg,
        copy.deepcopy(instance),
        solver_factory,
    )
    fallback_opt = count_fallback_items(optimal_state, instance)
    optimal_summary = {
        "pipeline": "OPTIMAL_FULL_HORIZON",
        "seed": seed,
        "scenario": scenario_name,
        "scenario_description": scenario_description,
        "problem": {
            "N": cfg.problem.N,
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
        "offline_assignments": {str(k): int(v) for k, v in optimal_state.assigned_bin.items()},
    }
    optimal_summary["variant"] = label
    save_combined_result(prefix, optimal_summary, output_dir=str(output_dir))
    return optimal_summary


def _parse_offline_ratio_pct(scenario_name: str) -> int | None:
    """
    Extract the offline percentage from names like 'baseline_midvar_off30_on70'.
    Returns None if the pattern is absent.
    """
    marker = "_off"
    idx = scenario_name.rfind(marker)
    if idx < 0:
        return None
    suffix = scenario_name[idx + 1 :]  # offXX_onYY
    if not suffix.startswith("off"):
        return None
    try:
        off_token = suffix.split("_", 1)[0]  # offXX
        return int(off_token.replace("off", ""))
    except Exception:
        return None


def _collect_ratio_metrics(output_root: Path, metric: str) -> Dict[str, Dict[int, List[float]]]:
    """
    Load sweep JSONs and collect metric values per (pipeline, offline_pct).
    Uses only feasible/optimal runs to keep curves clean.
    """
    values: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for path in output_root.rglob("pipeline_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        pipeline = data.get("pipeline")
        if not pipeline:
            continue

        status = data.get("online", {}).get("status", "")
        if status not in GOOD_STATUSES:
            continue

        offline_pct = _parse_offline_ratio_pct(data.get("scenario", "") or path.parent.parent.name)
        if offline_pct is None:
            continue

        try:
            if metric == "runtime_total":
                offline_rt = float(data["offline"]["runtime"])
                online_rt = float(data["online"]["runtime"])
                val = offline_rt + online_rt
            elif metric in data:
                val = float(data[metric])
            elif metric in data.get("online", {}):
                val = float(data["online"][metric])
            else:
                continue
        except Exception:
            continue

        values[pipeline][offline_pct].append(val)

    return values


def _make_ratio_line_plot(output_root: Path, metric: str = "objective_ratio") -> None:
    """
    Create a minimal line chart showing metric vs offline ratio for each pipeline.
    Saves PNG and PDF to output_root.
    """
    metric_values = _collect_ratio_metrics(output_root, metric)
    if not metric_values:
        print("[warn] No data found for ratio line plot (did you run with --compute-optimal for objective ratios?).")
        return

    # Aggregate means per ratio
    curves: Dict[str, List[Tuple[int, float]]] = {}
    for pipe, by_ratio in metric_values.items():
        pts = []
        for ratio, vals in by_ratio.items():
            if not vals:
                continue
            pts.append((ratio, sum(vals) / float(len(vals))))
        if pts:
            curves[pipe] = sorted(pts, key=lambda t: t[0])

    if not curves:
        print("[warn] No aggregated points for ratio line plot.")
        return

    # Lazy import to avoid forcing matplotlib on code paths that don't plot.
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    all_ratios = set()
    for points in curves.values():
        all_ratios.update(r for r, _ in points)
    xticks = sorted(all_ratios)

    for pipe, points in sorted(curves.items()):
        ratios = [r for r, _ in points]
        vals = [v for _, v in points]
        ax.plot(ratios, vals, marker="o", linewidth=2.0, markersize=4, label=pipe)

    y_label = {
        "objective_ratio": "Objective ratio (cost / OPT)",
        "total_objective": "Total objective",
        "runtime_total": "Runtime total (s)",
    }.get(metric, metric)

    ax.set_xlabel("Offline items (%)")
    ax.set_ylabel(y_label)
    ax.set_xticks(xticks)
    ax.set_xlim(min(xticks), max(xticks))
    ax.legend(frameon=False, title="Pipeline")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out_png = output_root / "ratio_line_plot.png"
    out_pdf = output_root / "ratio_line_plot.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[info] Saved ratio line plot: {out_png} and {out_pdf}")
    plt.close(fig)


def run_param_sweep(
    base_cfg: Config,
    scenarios: Sequence[ScenarioConfig],
    pipeline_specs: Sequence[PipelineSpec],
    output_root: Path,
    compute_optimal: bool,
    *,
    make_ratio_plot: bool = True,
    ratio_plot_metric: str = "objective_ratio",
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        scenario_cfg = apply_config_overrides(base_cfg, scenario.overrides)
        seeds = list(scenario.seeds or scenario_cfg.eval.seeds)
        scenario_dir = output_root / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Scenario: {scenario.name} ({scenario.description or 'no description'}) ===")
        print(
            f"Offline items: {scenario_cfg.problem.M_off}, "
            f"online horizon: {scenario_cfg.stoch.horizon}, "
            f"seeds: {seeds}"
        )

        variant_label = "base"
        cfg_variant = scenario_cfg
        meta: Dict[str, Any] = {}
        variant_dir = scenario_dir / variant_label
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n-- Variant: {variant_label} ({meta}) --")

        shared_instances: Dict[int, Instance] = {}
        optimal_by_seed: Dict[int, Dict[str, Any]] = {}

        for seed in seeds:
            set_global_seed(seed)
            # NOTE: `generate_offline_instance` mutates `cfg.problem.capacities` (it fills
            # sampled capacities when the list is empty). Deep-copy here so each seed
            # gets its own independently sampled capacities.
            inst = generate_instance_with_online(copy.deepcopy(cfg_variant), seed=seed)
            shared_instances[seed] = inst
            offline_count = len(inst.offline_items)
            online_count = len(inst.online_items)
            print(
                f"Seed {seed}: offline items={offline_count}, online items={online_count}"
            )
            if compute_optimal and offline_count + online_count > 0:
                optimal_by_seed[seed] = _compute_optimal(
                    cfg_variant,
                    copy.deepcopy(inst),
                    seed,
                    variant_dir,
                    variant_label,
                    scenario_name=scenario.name,
                    scenario_description=scenario.description,
                )

        offline_cache: Dict[Tuple[int, str], OfflineResult] = {}

        for seed in seeds:
            instance = shared_instances[seed]
            offline_items = len(instance.offline_items)
            online_items = len(instance.online_items)
            if offline_items + online_items == 0:
                continue

            set_global_seed(seed)
            for spec in pipeline_specs:
                cache_id = spec.offline_cache_key or spec.name
                cache_key = (seed, cache_id)
                cached_solution = offline_cache.get(cache_key)
                if cached_solution is None and instance.offline_items:
                    solver = spec.offline_factory(cfg_variant)
                    offline_state, offline_info = solver.solve(copy.deepcopy(instance))
                    offline_cache[cache_key] = (offline_state, offline_info)
                    cached_solution = offline_cache[cache_key]

                summary, path = run_pipeline(
                    cfg_variant,
                    spec,
                    seed=seed,
                    instance=instance,
                    offline_solution=cached_solution,
                    output_dir=str(variant_dir),
                )
                optimal = optimal_by_seed.get(seed)
                if optimal and optimal["online"]["total_objective"] > 0:
                    summary["optimal_objective"] = float(optimal["online"]["total_objective"])
                    summary["objective_ratio"] = (
                        summary["online"]["total_objective"] / summary["optimal_objective"]
                    )
                    summary["optimal_fallback"] = int(optimal["final_items_in_fallback"])

                summary["scenario"] = scenario.name
                summary["scenario_description"] = scenario.description
                family, scen_variant = _scenario_family_variant(scenario.name)
                summary["scenario_family"] = family
                summary["scenario_variant"] = scen_variant
                summary["variant"] = variant_label
                summary["variant_params"] = meta
                Path(path).write_text(json.dumps(summary, indent=2))

                offline_runtime = summary["offline"]["runtime"]
                online_runtime = summary["online"]["runtime"]
                total_obj = summary["online"]["total_objective"]
                fallback_final = summary["final_items_in_fallback"]
                print(
                    f"  {spec.name}: offline {offline_runtime:.3f}s, online {online_runtime:.3f}s, "
                    f"fallback={fallback_final}, objective={total_obj:.3f} -> {path}"
                )

    if make_ratio_plot:
        _make_ratio_line_plot(output_root, metric=ratio_plot_metric)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)
    scenarios = list(select_scenarios(args.scenarios))
    pipelines = _select_pipelines(args.pipelines)
    run_param_sweep(
        base_cfg,
        scenarios,
        pipelines,
        args.output_root,
        args.compute_optimal,
        make_ratio_plot=not getattr(args, "skip_ratio_plot", False),
        ratio_plot_metric=getattr(args, "ratio_plot_metric", "objective_ratio"),
    )


if __name__ == "__main__":
    main()
