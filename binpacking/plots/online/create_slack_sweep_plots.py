from __future__ import annotations

"""
Aggregate slack sweep results and visualize absolute, marginal, and relative metrics.

Run after executing `binpacking/experiments/run_slack_sweep.py`.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import numpy as np

from binpacking.config import load_config
from binpacking.experiments.scenarios import apply_config_overrides, select_scenarios


RESULTS_ROOT = Path("binpacking/results/slack_sweep")
OUTPUT_DIR = Path("binpacking/plots/online/slack_sweep")
BASE_CONFIG_PATH = Path("configs/binpacking.yaml")
EXPORT_DPI = 180  # lower dpi to speed up rendering/saving
SCENARIO_LABELS = {
    # Smooth i.i.d.
    "smoothdist_ratio_off0_on100":  "Smooth 0/100",
    "smoothdist_ratio_off20_on80":  "Smooth 20/80",
    "smoothdist_ratio_off50_on50":  "Smooth 50/50",
    "smoothdist_ratio_off80_on20":  "Smooth 80/20",

    # Heavy-tailed volumes
    "heavyvol_ratio_off0_on100":    "HeavyVol 0/100",
    "heavyvol_ratio_off20_on80":    "HeavyVol 20/80",
    "heavyvol_ratio_off50_on50":    "HeavyVol 50/50",
    "heavyvol_ratio_off80_on20":    "HeavyVol 80/20",

    # Sparse online graphs
    "sparse_ratio_off0_on100":      "Sparse 0/100",
    "sparse_ratio_off20_on80":      "Sparse 20/80",
    "sparse_ratio_off50_on50":      "Sparse 50/50",
    "sparse_ratio_off80_on20":      "Sparse 80/20",

    # Overload 
    "overloaddist_ratio_off0_on100": "Overload 0/100",
    "overloaddist_ratio_off20_on80": "Overload 20/80",
    "overloaddist_ratio_off50_on50": "Overload 50/50",
    "overloaddist_ratio_off80_on20": "Overload 80/20",
}
SCENARIO_LABEL_ORDER = [SCENARIO_LABELS[key] for key in SCENARIO_LABELS]
GOOD_STATUSES = {"COMPLETED", "OPTIMAL"}


def _scenario_label(name: str) -> str:
    return SCENARIO_LABELS.get(name, name)


def _scenario_family(label: str) -> str:
    # Use the leading token ("Smooth", "HeavyVol", "Sparse", "Overload", etc.)
    return label.split()[0] if label else "Unknown"


def _slugify(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_").lower()


def _parse_slack_label(name: str) -> float | None:
    prefix = "slack_"
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix) :])
    except ValueError:
        return None


def format_fraction(value: float) -> str:
    formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _load_scenario_cost_params() -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    try:
        base_cfg = load_config(BASE_CONFIG_PATH)
    except Exception:
        return params

    default_costs = {
        "fallback_cost": float(getattr(base_cfg.costs, "huge_fallback", 0.0)),
        "reassignment_penalty": float(getattr(base_cfg.costs, "reassignment_penalty", 0.0)),
    }
    params["__default__"] = default_costs

    try:
        scenarios = list(select_scenarios(None))
    except Exception:
        return params

    for scenario in scenarios:
        cfg = apply_config_overrides(base_cfg, scenario.overrides)
        params[scenario.name] = {
            "fallback_cost": float(cfg.costs.huge_fallback),
            "reassignment_penalty": float(cfg.costs.reassignment_penalty),
        }
    return params


SCENARIO_COST_PARAMS = _load_scenario_cost_params()


def _cost_param(name: str, key: str) -> float:
    params = SCENARIO_COST_PARAMS.get(name) or SCENARIO_COST_PARAMS.get("__default__", {})
    return float(params.get(key, 0.0))


def load_optimal_map() -> Dict[Tuple[str, int], Dict[str, float]]:
    ref: Dict[Tuple[str, int], Dict[str, float]] = {}
    if not RESULTS_ROOT.exists():
        return ref

    for scenario_dir in RESULTS_ROOT.iterdir():
        if not scenario_dir.is_dir():
            continue
        for opt_file in scenario_dir.glob("pipeline_optimal_full_horizon_*.json"):
            data = json.loads(opt_file.read_text())
            ref[(scenario_dir.name, int(data["seed"]))] = {
                "objective": float(data["online"]["total_objective"]),
                "fallback": float(data["final_items_in_fallback"]),
                "offline_obj": float(data["offline"].get("obj_value", 0.0)),
                "online_cost": float(data["online"].get("total_cost", 0.0)),
                "offline_fallback": int(data["offline"].get("items_in_fallback", 0)),
            }
    return ref


def load_pipeline_records() -> pd.DataFrame:
    records = []
    optimal_map = load_optimal_map()
    if not RESULTS_ROOT.exists():
        return pd.DataFrame()

    scenario_slacks: Dict[str, set] = defaultdict(set)

    for scenario_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        for slack_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            slack_fraction = _parse_slack_label(slack_dir.name)
            if slack_fraction is None:
                continue
            scenario_slacks[scenario_dir.name].add(float(slack_fraction))
            for result_file in slack_dir.glob("pipeline_*.json"):
                data = json.loads(result_file.read_text())
                key = (scenario_dir.name, int(data["seed"]))
                opt = optimal_map.get(key, {})
                records.append(
                    {
                        "scenario": scenario_dir.name,
                        "slack_fraction": float(slack_fraction),
                        "pipeline": data["pipeline"],
                        "seed": int(data["seed"]),
                        "total_objective": float(data["online"]["total_objective"]),
                        "offline_obj": float(data["offline"]["obj_value"]),
                        "offline_fallback": int(data["offline"].get("items_in_fallback", 0)),
                        "online_total_cost": float(data["online"]["total_cost"]),
                        "online_evicted_offline": int(data["online"].get("evicted_offline", 0)),
                        "fallback_final": int(data["final_items_in_fallback"]),
                        "runtime_offline": float(data["offline"]["runtime"]),
                        "runtime_online": float(data["online"]["runtime"]),
                        "slack_enforced": data.get("slack", {}).get("enforce_slack"),
                        "slack_recorded": data.get("slack", {}).get("fraction"),
                        "slack_apply_online": data.get("slack", {}).get("apply_to_online"),
                        "status_online": data.get("online", {}).get("status", "UNKNOWN"),
                        "optimal_objective": opt.get("objective"),
                        "optimal_fallback": opt.get("fallback"),
                    }
                )

    optimal_by_scenario: Dict[str, list] = defaultdict(list)
    for (scenario, seed), opt in optimal_map.items():
        optimal_by_scenario[scenario].append((seed, opt))

    for scenario, slacks in scenario_slacks.items():
        for slack_fraction in sorted(slacks):
            for seed, opt in optimal_by_scenario.get(scenario, []):
                records.append(
                    {
                        "scenario": scenario,
                        "slack_fraction": float(slack_fraction),
                        "pipeline": "OPTIMAL_FULL_HORIZON",
                        "seed": seed,
                        "total_objective": float(opt["objective"]),
                        "offline_obj": float(opt.get("offline_obj", 0.0)),
                        "offline_fallback": int(opt.get("offline_fallback", 0)),
                        "online_total_cost": float(opt.get("online_cost", 0.0)),
                        "online_evicted_offline": 0,
                        "fallback_final": int(opt["fallback"]),
                        "runtime_offline": 0.0,
                        "runtime_online": 0.0,
                        "slack_enforced": None,
                        "slack_recorded": slack_fraction,
                        "slack_apply_online": None,
                        "status_online": "OPTIMAL",
                        "optimal_objective": float(opt["objective"]),
                        "optimal_fallback": float(opt["fallback"]),
                    }
                )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["scenario_label"] = df["scenario"].map(_scenario_label)
    df["fallback_unit_cost"] = df["scenario"].map(lambda name: _cost_param(name, "fallback_cost"))
    df["reassignment_penalty_cost"] = df["scenario"].map(lambda name: _cost_param(name, "reassignment_penalty"))
    return df


def compute_aggregates(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {}

    df = df.copy()
    df["is_feasible"] = df["status_online"].isin(GOOD_STATUSES)
    df = df[df["is_feasible"]]
    if df.empty:
        return {}

    df["runtime_total"] = df["runtime_offline"] + df["runtime_online"]
    df.sort_values(["scenario", "pipeline", "slack_fraction", "seed"], inplace=True)
    df["offline_fallback"] = df["offline_fallback"].fillna(0)
    df["fallback_unit_cost"] = df["fallback_unit_cost"].fillna(0.0)
    df["reassignment_penalty_cost"] = df["reassignment_penalty_cost"].fillna(0.0)
    fallback_increase = df["fallback_final"] - df["offline_fallback"]
    df["fallback_cost"] = fallback_increase.clip(lower=0) * df["fallback_unit_cost"]
    df["penalty_cost"] = df["online_evicted_offline"] * df["reassignment_penalty_cost"]
    df["online_assign_cost"] = (
        df["online_total_cost"] - df["fallback_cost"] - df["penalty_cost"]
    ).clip(lower=0.0)

    # Relative metrics per seed (requires optimal reference)
    rel_mask = df["optimal_objective"].notnull() & (df["optimal_objective"] > 0)
    df.loc[rel_mask, "objective_ratio"] = df.loc[rel_mask, "total_objective"] / df.loc[rel_mask, "optimal_objective"]
    df.loc[df["optimal_fallback"].notnull(), "fallback_gap"] = (
        df["fallback_final"] - df["optimal_fallback"]
    )

    abs_agg = (
        df.groupby(["scenario", "pipeline", "slack_fraction"])
        .agg(
            total_objective_mean=("total_objective", "mean"),
            runtime_total_mean=("runtime_total", "mean"),
            runs=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["scenario", "pipeline", "slack_fraction"])
    )
    abs_agg["scenario_label"] = abs_agg["scenario"].map(_scenario_label)

    rel_agg = pd.DataFrame()
    if "objective_ratio" in df:
        rel_agg = (
            df.dropna(subset=["objective_ratio"])
            .groupby(["scenario", "pipeline", "slack_fraction"])
            .agg(
                objective_ratio_mean=("objective_ratio", "mean"),
                fallback_gap_mean=("fallback_gap", "mean"),
            )
            .reset_index()
            .sort_values(["scenario", "pipeline", "slack_fraction"])
        )
        rel_agg["scenario_label"] = rel_agg["scenario"].map(_scenario_label)

    finite_opt_mask = (
        df["optimal_objective"].notnull()
        & np.isfinite(df["optimal_objective"])
        & df["optimal_fallback"].notnull()
    )
    regret_agg = pd.DataFrame()
    if finite_opt_mask.any():
        regret_df = df[finite_opt_mask].copy()
        regret_df["total_regret"] = regret_df["total_objective"] - regret_df["optimal_objective"]
        regret_df["fallback_regret"] = (
            (regret_df["fallback_final"] - regret_df["optimal_fallback"])
            * regret_df["fallback_unit_cost"]
        )
        regret_df["penalty_regret"] = (
            regret_df["online_evicted_offline"] * regret_df["reassignment_penalty_cost"]
        )
        regret_df["residual_regret"] = (
            regret_df["total_regret"]
            - regret_df["fallback_regret"]
            - regret_df["penalty_regret"]
        )
        regret_agg = (
            regret_df.groupby(["scenario", "pipeline", "slack_fraction"])
            .agg(
                total_regret_mean=("total_regret", "mean"),
                fallback_regret_mean=("fallback_regret", "mean"),
                penalty_regret_mean=("penalty_regret", "mean"),
                residual_regret_mean=("residual_regret", "mean"),
            )
            .reset_index()
            .sort_values(["scenario", "pipeline", "slack_fraction"])
        )
        regret_agg["scenario_label"] = regret_agg["scenario"].map(_scenario_label)

    composition_df = (
        df.groupby(["scenario", "pipeline", "slack_fraction"])
        .agg(
            offline_cost_mean=("offline_obj", "mean"),
            online_assign_cost_mean=("online_assign_cost", "mean"),
            fallback_cost_mean=("fallback_cost", "mean"),
            penalty_cost_mean=("penalty_cost", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "pipeline", "slack_fraction"])
    )
    composition_df["scenario_label"] = composition_df["scenario"].map(_scenario_label)
    composition_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return {
        "absolute": abs_agg,
        "relative": rel_agg,
        "regret": regret_agg,
        "composition": composition_df,
    }


def _plot_line(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    filename: str,
    *,
    sharey: bool = False,
    baseline: float | None = None,
) -> None:
    if df.empty:
        return
    col_order = [label for label in SCENARIO_LABEL_ORDER if label in df["scenario_label"].unique()]
    g = sns.relplot(
        data=df,
        x="slack_fraction",
        y=value_col,
        hue="pipeline",
        col="scenario_label",
        col_order=col_order or None,
        col_wrap=4,
        kind="line",
        marker="o",
        facet_kws={"sharey": sharey},
    )
    g.set_axis_labels("Slack fraction", ylabel)
    g.set_titles("{col_name}")
    if baseline is not None:
        for ax in g.axes.flat:
            if ax is not None:
                ax.axhline(baseline, color="gray", linestyle="--", linewidth=1)

    g.fig.suptitle(title, y=1.02)
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.borderaxespad = 0.0
        g._legend.set_title("Pipeline")
        g._legend._loc = 6  # left center
    plt.tight_layout()
    g.savefig(OUTPUT_DIR / f"{filename}.png", dpi=EXPORT_DPI)
    g.savefig(OUTPUT_DIR / f"{filename}.pdf", dpi=EXPORT_DPI)
    plt.close(g.fig)


COMPONENT_STYLES = [
    ("offline_cost_mean", "#4C72B0", "Offline assignment"),
    ("online_assign_cost_mean", "#55A868", "Online assignment"),
    ("fallback_cost_mean", "#C44E52", "Fallback penalty"),
    ("penalty_cost_mean", "#8172B3", "Eviction penalty"),
]


def _stacked_barplot(data, component_order, component_colors, **kws):
    data = data.dropna(subset=["slack_fraction"])
    if data.empty:
        return
    data = data.sort_values("slack_fraction")
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    ax = plt.gca()
    for comp in component_order:
        if comp not in data:
            values = np.zeros(len(data))
        else:
            values = data[comp].to_numpy()
        values = np.nan_to_num(values, nan=0.0)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=component_colors.get(comp, None),
            width=0.9,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_fraction(val) for val in data["slack_fraction"]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylim(bottom=0.0)


def _plot_cost_composition(df: pd.DataFrame) -> None:
    if df.empty:
        return
    col_order = [label for label in SCENARIO_LABEL_ORDER if label in df["scenario_label"].unique()]
    component_order = [comp for comp, _, _ in COMPONENT_STYLES]
    component_labels = {comp: label for comp, _, label in COMPONENT_STYLES}
    component_colors = {comp: color for comp, color, _ in COMPONENT_STYLES}

    for pipeline, subset in df.groupby("pipeline"):
        subset_clean = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["slack_fraction"])
        if subset_clean.empty:
            continue
        grid = sns.FacetGrid(
            subset_clean,
            col="scenario_label",
            col_order=col_order or None,
            col_wrap=4,
            sharey=False,
            margin_titles=True,
        )
        grid.map_dataframe(
            _stacked_barplot,
            component_order=component_order,
            component_colors=component_colors,
        )
        grid.set_axis_labels("Slack fraction", "Cost contribution")
        grid.set_titles("{col_name}")

        handles = [Patch(color=component_colors[comp], label=component_labels[comp]) for comp in component_order]
        labels = [component_labels[comp] for comp in component_order]
        grid.fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 1.08),
        )
        grid.fig.suptitle(f"Cost Composition – {pipeline}", y=1.12)
        grid.fig.subplots_adjust(top=0.82)
        fname = f"slack_cost_composition_{pipeline.replace('+', '_').replace(' ', '_')}"
        grid.savefig(OUTPUT_DIR / f"{fname}.png", dpi=EXPORT_DPI)
        grid.savefig(OUTPUT_DIR / f"{fname}.pdf", dpi=EXPORT_DPI)
        plt.close(grid.fig)


def create_plots(aggregates: Dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    abs_df = aggregates.get("absolute", pd.DataFrame())
    rel_df = aggregates.get("relative", pd.DataFrame())
    regret_df = aggregates.get("regret", pd.DataFrame())
    comp_df = aggregates.get("composition", pd.DataFrame())

    _plot_line(
        abs_df,
        value_col="total_objective_mean",
        ylabel="Total objective (mean)",
        title="Objective vs Slack",
        filename="slack_absolute_objective",
    )
    _plot_line(
        rel_df,
        value_col="objective_ratio_mean",
        ylabel="Objective / Optimal",
        title="Objective vs Optimal",
        filename="slack_relative_objective",
        sharey=True,
        baseline=1.0,
    )
    _plot_line(
        regret_df,
        value_col="total_regret_mean",
        ylabel="Regret (vs optimal)",
        title="Total Regret vs Slack",
        filename="slack_total_regret",
    )
    _plot_line(
        regret_df,
        value_col="fallback_regret_mean",
        ylabel="Fallback Regret",
        title="Fallback Regret vs Slack",
        filename="slack_fallback_regret",
    )
    _plot_line(
        regret_df,
        value_col="penalty_regret_mean",
        ylabel="Reassignment Regret",
        title="Penalty Regret vs Slack",
        filename="slack_penalty_regret",
    )
    _plot_line(
        regret_df,
        value_col="residual_regret_mean",
        ylabel="Residual Regret",
        title="Residual Regret vs Slack",
        filename="slack_residual_regret",
    )
    _plot_cost_composition(comp_df)


def main() -> None:
    df = load_pipeline_records()
    if df.empty:
        print(f"No results found under {RESULTS_ROOT}. Run binpacking/experiments/run_slack_sweep.py first.")
        return
    aggregates = compute_aggregates(df)
    if not aggregates:
        print("No aggregates could be computed.")
        return
    print(f"Loaded {len(df)} runs across {len(df['scenario'].unique())} scenarios.")
    create_plots(aggregates)
    print(f"Slack sweep plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
