from __future__ import annotations

"""
Readable plots for scenario-driven experiments produced by experiments/run_param_sweep.py.

Key fixes vs your current plotting:
- Correct grouping: scenario names are parsed into (family_base, knob_variant, ratio_variant).
  This ensures e.g. vol_lowvar_off20_on80 and vol_highvar_off20_on80 appear in the SAME figure.
- One figure per family_base.
- X axis chosen automatically per family: knob_variant if it varies, else ratio_variant if it varies, else scenario.
- Grouped BAR charts with SEM error bars (across seeds), which is usually far more readable than many line plots.
- Optional "top-k pipelines" to prevent unreadable legends.

Example:
  python -m binpacking.plots.online.create_param_sweep_plots --metric objective_ratio
  python -m binpacking.plots.online.create_param_sweep_plots --metric objective_ratio --topk 6
  python -m binpacking.plots.online.create_param_sweep_plots --families vol load graph baseline
"""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binpacking.config import load_config
from binpacking.data.instance_generators import generate_offline_instance
from binpacking.experiments.scenarios import apply_config_overrides, select_scenarios
from generic.online.state_utils import effective_capacities
from generic.general_utils import scalarize_vector

RESULTS_ROOT = Path("binpacking/results/param_sweep")
OUTPUT_DIR = Path("binpacking/plots/online/param_sweep")

GOOD_STATUSES = {"COMPLETED", "OPTIMAL", "NO_ITEMS"}


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot scenario-driven sweep results (readable).")
    p.add_argument(
        "--base_config",
        type=Path,
        default=Path("configs/binpacking.yaml"),
        help="Base YAML config used during experiments (needed to recover bin capacities).",
    )
    p.add_argument(
        "--metric",
        choices=["objective_ratio", "total_objective", "runtime_total"],
        default="objective_ratio",
        help="Metric to visualize.",
    )
    p.add_argument("--scenarios", nargs="+", help="Optional subset of scenario names to include.")
    p.add_argument("--pipelines", nargs="+", help="Optional subset of pipelines to include.")
    p.add_argument("--families", nargs="+", help="Optional subset of family_base to include (e.g. vol load graph baseline).")
    p.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Plot only top-k pipelines per family (ranked by metric_mean, lower is better for objective_ratio/total_objective). "
             "Set <=0 to plot all pipelines (not recommended).",
    )
    p.add_argument(
        "--min_feasible_runs",
        type=int,
        default=1,
        help="Drop pipelines with fewer feasible runs than this threshold (per family).",
    )
    p.add_argument(
        "--annotate_infeasible",
        action="store_true",
        help="Annotate infeasible counts above bars (can clutter if many).",
    )
    return p.parse_args()


# -------------------------
# Scenario parsing (FIX)
# -------------------------

def parse_scenario_name(name: str) -> Tuple[str, str, str]:
    """
    Parse scenario naming convention into:
      family_base: first token (e.g. 'vol', 'load', 'graph', 'baseline')
      knob_variant: remainder before ratio suffix (e.g. 'lowvar', 'overload', 'sparse', 'midvar', or 'base')
      ratio_variant: ratio suffix if present (e.g. 'off20_on80'), else 'none'

    Examples:
      baseline_midvar_off20_on80 -> ('baseline', 'midvar', 'off20_on80')
      vol_lowvar_off20_on80      -> ('vol', 'lowvar', 'off20_on80')
      load_overload_off20_on80   -> ('load', 'overload', 'off20_on80')
      something                  -> ('something', 'base', 'none')
    """
    ratio_variant = "none"
    prefix = name

    marker = "_off"
    idx = name.rfind(marker)
    if idx > 0:
        prefix = name[:idx]          # everything before "_off.."
        ratio_variant = name[idx + 1 :]  # starts at "off.."

    # prefix may be "vol_lowvar" or "baseline_midvar" etc
    parts = prefix.split("_", 1)
    family_base = parts[0] if parts else prefix
    knob_variant = parts[1] if len(parts) == 2 and parts[1] else "base"

    return family_base, knob_variant, ratio_variant


def _scenario_order() -> Dict[str, int]:
    """
    Stable ordering by ScenarioConfig list order when available.
    """
    try:
        scenarios = select_scenarios(None)
    except Exception:
        return {}
    order: Dict[str, int] = {}
    for idx, scen in enumerate(scenarios):
        order[scen.name] = idx
    return order


SCENARIO_ORDER = _scenario_order()


class EffectiveCapacityResolver:
    """
    Recover effective bin capacities for a (scenario, seed) pair so we can express residuals
    as percentage of capacity consumed. Results are cached per key to avoid redundant
    regeneration of offline instances.
    """

    def __init__(self, base_config_path: Path) -> None:
        self.base_config_path = Path(base_config_path)
        self._scenario_cfgs: Dict[str, Any] = {}
        self._cache: Dict[Tuple[str, int], Optional[List[float]]] = {}
        self._missing: set[str] = set()
        self._init_configs()

    def _init_configs(self) -> None:
        base_cfg = load_config(self.base_config_path)
        try:
            scenarios = select_scenarios(None)
        except Exception:
            scenarios = []
        for scen in scenarios:
            cfg = apply_config_overrides(copy.deepcopy(base_cfg), scen.overrides)
            self._scenario_cfgs[scen.name] = cfg

    def get(self, scenario_name: str, seed: int) -> Optional[List[float]]:
        key = (scenario_name, int(seed))
        if key in self._cache:
            return self._cache[key]
        cfg_template = self._scenario_cfgs.get(scenario_name)
        if cfg_template is None:
            if scenario_name not in self._missing:
                print(f"[warn] No scenario config registered for '{scenario_name}'. Skipping capacity lookup.")
                self._missing.add(scenario_name)
            self._cache[key] = None
            return None

        cfg = copy.deepcopy(cfg_template)
        try:
            inst = generate_offline_instance(cfg, int(seed))
            caps_vec = effective_capacities(inst, cfg, use_slack=cfg.slack.enforce_slack)
            caps = [scalarize_vector(np.asarray(c), cfg.heuristics.residual_scalarization) for c in caps_vec]
        except Exception as exc:
            if scenario_name not in self._missing:
                print(f"[warn] Failed to recover capacities for scenario '{scenario_name}' seed {seed}: {exc}")
                self._missing.add(scenario_name)
            caps = None
        self._cache[key] = caps
        return caps


# -------------------------
# Loading records
# -------------------------

def iter_records() -> Iterable[Dict[str, Any]]:
    if not RESULTS_ROOT.exists():
        return []

    for path in RESULTS_ROOT.rglob("pipeline_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        pipeline = data.get("pipeline", "")
        if pipeline == "OPTIMAL_FULL_HORIZON":
            continue

        scenario = data.get("scenario") or path.parent.parent.name
        seed = int(data.get("seed", 0))
        status = data.get("online", {}).get("status", "UNKNOWN")

        family_base, knob_variant, ratio_variant = parse_scenario_name(scenario)
        final_fallback = data.get("final_items_in_fallback")
        penalty_events = data.get("online", {}).get("evicted_offline")

        # Compute objective_ratio if missing but optimal_objective present
        obj_ratio = data.get("objective_ratio")
        if obj_ratio is None:
            opt_obj = data.get("optimal_objective")
            if opt_obj is not None and float(opt_obj) > 0:
                obj_ratio = float(data["online"]["total_objective"]) / float(opt_obj)

        yield {
            "scenario": scenario,
            "scenario_order": SCENARIO_ORDER.get(scenario, 10**9),
            "family_base": family_base,
            "knob_variant": knob_variant,
            "ratio_variant": ratio_variant,
            "pipeline": pipeline,
            "seed": seed,
            "status_online": status,
            "total_objective": float(data["online"]["total_objective"]),
            "runtime_total": float(data["offline"]["runtime"] + data["online"]["runtime"]),
            "objective_ratio": obj_ratio,
            "infeasible_flag": 0 if status in GOOD_STATUSES else 1,
            "final_fallback": float(final_fallback) if final_fallback is not None else np.nan,
            "evicted_offline": float(penalty_events) if penalty_events is not None else np.nan,
            "offline_residuals": data.get("offline_residual_capacities"),
            "final_residuals": data.get("final_residual_capacities"),
            "offline_loads": data.get("offline_loads"),
            "final_loads": data.get("final_loads"),
        }


# -------------------------
# Aggregation
# -------------------------

def sem(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    n = len(x)
    if n <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(n))


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Aggregate per (family_base, knob_variant, ratio_variant, pipeline).
    Mean/SEM computed over feasible runs only.
    """
    if df.empty:
        return df

    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found.")

    keys = ["family_base", "knob_variant", "ratio_variant", "pipeline"]

    counts = (
        df.groupby(keys)
        .agg(
            runs_total=("seed", "nunique"),
            infeasible_runs=("infeasible_flag", "sum"),
        )
        .reset_index()
    )

    feasible = df[df["status_online"].isin(GOOD_STATUSES)].copy()
    stats = (
        feasible.groupby(keys)
        .agg(
            feasible_runs=("seed", "nunique"),
            metric_mean=(metric, "mean"),
            metric_sem=(metric, sem),
        )
        .reset_index()
    )

    out = counts.merge(stats, on=keys, how="left")
    out["feasible_runs"] = out["feasible_runs"].fillna(0).astype(int)
    out["metric_mean"] = out["metric_mean"].astype(float)
    out["metric_sem"] = out["metric_sem"].astype(float)
    out["infeasible_rate"] = out.apply(
        lambda r: float(r["infeasible_runs"]) / float(r["runs_total"]) if r["runs_total"] else 0.0,
        axis=1,
    )
    return out


# -------------------------
# Plotting (readable bars)
# -------------------------

def choose_x_dimension(df_family: pd.DataFrame) -> str:
    """
    Choose x dimension for a family:
      - if knob_variant varies -> use knob_variant
      - else if ratio_variant varies -> use ratio_variant
      - else fallback -> scenario-like label (knob_variant anyway)
    """
    if df_family["knob_variant"].nunique() > 1:
        return "knob_variant"
    if df_family["ratio_variant"].nunique() > 1:
        return "ratio_variant"
    return "knob_variant"


def order_x(df_family: pd.DataFrame, x_dim: str) -> List[str]:
    """
    Stable ordering:
    - If x is ratio_variant, order by off% increasing (if parsable), else lexicographic.
    - Else lexicographic (or scenario order if you embed it in knob names).
    """
    vals = df_family[x_dim].astype(str).unique().tolist()

    if x_dim == "ratio_variant":
        def key(v: str) -> Tuple[int, str]:
            # parse "off20_on80" -> 20
            try:
                if v.startswith("off"):
                    off = int(v.split("_")[0].replace("off", ""))
                    return (off, v)
            except Exception:
                pass
            return (10**9, v)
        return sorted(vals, key=key)

    return sorted(vals)


def rank_pipelines(df_family: pd.DataFrame, metric: str) -> List[str]:
    """
    Rank pipelines by average metric_mean across x for this family.
    For objective_ratio and total_objective: lower is better (cost minimization).
    For runtime_total: lower is better.
    """
    tmp = (
        df_family.groupby("pipeline")
        .agg(score=("metric_mean", "mean"), feasible=("feasible_runs", "sum"))
        .reset_index()
    )
    tmp = tmp.sort_values(["score", "pipeline"], ascending=[True, True])
    return tmp["pipeline"].tolist()


def plot_family(
    df_family: pd.DataFrame,
    *,
    metric: str,
    topk: int,
    min_feasible_runs: int,
    annotate_infeasible: bool,  # can ignore now; we plot failure rate below
    ) -> List[str]:
    family = str(df_family["family_base"].iloc[0])
    x_dim = choose_x_dimension(df_family)
    x_order = order_x(df_family, x_dim)

    # Drop pipelines with too few feasible runs overall
    df_family = df_family.copy()
    feasible_by_pipe = df_family.groupby("pipeline")["feasible_runs"].sum()
    keep = feasible_by_pipe[feasible_by_pipe >= min_feasible_runs].index.tolist()
    df_family = df_family[df_family["pipeline"].isin(keep)]
    if df_family.empty:
        return

    # Rank pipelines by performance and keep top-k
    pipes = rank_pipelines(df_family, metric)
    if topk and topk > 0:
        pipes = pipes[:topk]
        df_family = df_family[df_family["pipeline"].isin(pipes)]
    else:
        pipes = sorted(df_family["pipeline"].unique().tolist())

    if not pipes:
        return

    # ---- Metric transform for better readability ----
    # For objective_ratio (cost/OPT): plot gap to OPT in percent (0 is best).
    # For other metrics: keep as is.
    perf_label = metric
    perf_is_gap = False
    if metric == "objective_ratio":
        perf_is_gap = True
        perf_label = "Gap to OPT (%)"
        df_family = df_family.copy()
        df_family["perf_mean"] = 100.0 * (df_family["metric_mean"].astype(float) - 1.0)
        df_family["perf_sem"]  = 100.0 * (df_family["metric_sem"].astype(float))
    else:
        df_family["perf_mean"] = df_family["metric_mean"].astype(float)
        df_family["perf_sem"]  = df_family["metric_sem"].astype(float)

    # ---- Layout: modern 2-row figure (performance + failure rate) ----
    n_x = len(x_order)
    n_p = len(pipes)
    width = 0.82 / max(n_p, 1)
    xs = np.arange(n_x)

    fig_w = max(10.0, 1.15 * n_x + 4.0)
    fig_h = 6.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_fail = fig.add_subplot(gs[1, 0], sharex=ax)

    # Style (modern-ish without overdoing it)
    for a in (ax, ax_fail):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, axis="y", alpha=0.22)

    title = f"Family: {family}  |  x = {x_dim}"
    subtitle = f"Metric: {perf_label} (mean ± SEM over feasible runs). Failure rate = infeasible / total runs."
    fig.suptitle(title, y=0.98, fontsize=13)
    ax.set_title(subtitle, fontsize=9, color="dimgray", pad=8)

    # Build lookup per (pipeline, x)
    df_family["_x"] = df_family[x_dim].astype(str)
    lookup = {(str(r["pipeline"]), str(r["_x"])): r for _, r in df_family.iterrows()}

    # ---- Plot performance bars ----
    for j, pipe in enumerate(pipes):
        means, sems = [], []
        for xv in x_order:
            r = lookup.get((pipe, str(xv)))
            if r is None or int(r.get("feasible_runs", 0)) <= 0:
                means.append(np.nan)
                sems.append(0.0)
            else:
                means.append(float(r["perf_mean"]))
                sems.append(float(r["perf_sem"]) if math.isfinite(float(r["perf_sem"])) else 0.0)

        means = np.array(means, dtype=float)
        sems = np.array(sems, dtype=float)

        x_pos = xs - 0.41 + (j + 0.5) * width
        ax.bar(x_pos, means, width=width, label=pipe)
        ax.errorbar(x_pos, means, yerr=sems, fmt="none", capsize=3, linewidth=1)

    if perf_is_gap:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        # Auto-zoom y-limits around observed values (robust)
        vals = df_family["perf_mean"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(vals) > 0:
            lo = np.nanpercentile(vals, 5)
            hi = np.nanpercentile(vals, 95)
            pad = max(0.15, 0.15 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylabel(perf_label)

    # ---- Plot failure rate below (stacked by pipeline) OR aggregated ----
    # Here: pipeline-specific failure rate bars (same grouping), scaled 0..100%.
    for j, pipe in enumerate(pipes):
        fr = []
        for xv in x_order:
            r = lookup.get((pipe, str(xv)))
            if r is None:
                fr.append(np.nan)
            else:
                fr.append(100.0 * float(r.get("infeasible_rate", 0.0)))
        fr = np.array(fr, dtype=float)

        x_pos = xs - 0.41 + (j + 0.5) * width
        ax_fail.bar(x_pos, fr, width=width)

    ax_fail.set_ylabel("Fail (%)")
    ax_fail.set_ylim(0, 100)

    ax_fail.set_xticks(xs)
    ax_fail.set_xticklabels(x_order, rotation=25, ha="right")
    ax_fail.set_xlabel(x_dim)

    # Legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.86, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{metric}_family_{family}_x-{x_dim}".replace("/", "_")
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Debug table
    table = df_family.sort_values([x_dim, "pipeline"])
    (OUTPUT_DIR / f"{fname}.csv").write_text(table.to_csv(index=False))
    return pipes


def plot_residuals_family(
    df_family_raw: pd.DataFrame,
    family: str,
    pipelines: List[str],
    cap_lookup: EffectiveCapacityResolver,
    *,
    stage: str = "offline",
) -> None:
    """
    Visualize per-bin capacity usage grouped by scenario variants for the requested stage.
    `stage` can be "offline" (post-offline heuristic/MILP) or "final" (after running the
    online policy). Both plots also include the fallback bin, scaled by the average
    effective capacity to highlight how many "bin equivalents" land in fallback.
    """
    if df_family_raw.empty or not pipelines or cap_lookup is None:
        return

    stage_norm = stage.lower()
    if stage_norm not in {"offline", "final"}:
        raise ValueError(f"Unsupported residual plot stage '{stage}'.")

    df_family_raw = df_family_raw[df_family_raw["pipeline"].isin(pipelines)].copy()
    if df_family_raw.empty:
        return

    variant_dim = choose_x_dimension(df_family_raw)
    variant_order = order_x(df_family_raw, variant_dim)
    residual_field = "offline_residuals" if stage_norm == "offline" else "final_residuals"
    load_field = "offline_loads" if stage_norm == "offline" else "final_loads"
    title_prefix = (
        "Offline capacity used"
        if stage_norm == "offline"
        else "Final capacity used (after online)"
    )

    usage_rows: List[Dict[str, Any]] = []
    for row in df_family_raw.itertuples():
        residuals = getattr(row, residual_field, None)
        if not residuals:
            continue
        caps = cap_lookup.get(getattr(row, "scenario"), int(getattr(row, "seed")))
        if not caps:
            continue
        limit = min(len(residuals), len(caps))
        if limit <= 0:
            continue
        variant_value = getattr(row, variant_dim, "none")
        variant_value = str(variant_value if variant_value not in (None, "") else "none")
        loads = getattr(row, load_field, None)
        fallback_ref = float(np.mean(caps)) if caps else None

        for bin_idx in range(limit):
            cap = float(caps[bin_idx])
            if cap <= 0:
                continue
            residual = max(0.0, float(residuals[bin_idx]))
            used_ratio = 1.0 - min(1.0, residual / cap if cap else 1.0)
            used_ratio = max(0.0, min(1.0, used_ratio))
            usage_rows.append(
                {
                    "pipeline": getattr(row, "pipeline"),
                    "variant": variant_value,
                    "bin_idx": int(bin_idx),
                    "used_pct": 100.0 * used_ratio,
                    "is_fallback": False,
                }
            )

        if loads and len(loads) > len(caps) and fallback_ref and fallback_ref > 0:
            fallback_load = max(0.0, float(loads[len(caps)]))
            usage_rows.append(
                {
                    "pipeline": getattr(row, "pipeline"),
                    "variant": variant_value,
                    "bin_idx": int(len(caps)),
                    "used_pct": 100.0 * (fallback_load / fallback_ref),
                    "is_fallback": True,
                }
            )

    if not usage_rows:
        return

    usage_df = pd.DataFrame.from_records(usage_rows)
    stats = (
        usage_df.groupby(["pipeline", "variant", "bin_idx"])
        .agg(
            used_mean=("used_pct", "mean"),
            used_sem=("used_pct", sem),
            samples=("used_pct", "count"),
        )
        .reset_index()
    )
    if stats.empty:
        return

    fallback_flags = (
        usage_df.groupby("bin_idx")["is_fallback"].max().to_dict()
        if "is_fallback" in usage_df
        else {}
    )

    variants_present = [v for v in variant_order if v in stats["variant"].unique()]
    if not variants_present:
        variants_present = sorted(stats["variant"].unique())
    if not variants_present:
        variants_present = ["all"]

    bin_indices = sorted(stats["bin_idx"].unique())
    bin_labels = [
        "F" if fallback_flags.get(idx, False) else str(idx + 1) for idx in bin_indices
    ]

    pipes_present = [p for p in pipelines if p in stats["pipeline"].unique()]
    if not pipes_present:
        return

    overall_max = float(stats["used_mean"].max()) if not stats.empty else 0.0

    n = len(pipes_present)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / ncols)) if ncols else 1
    fig_w = max(9.0, 2.6 * ncols)
    fig_h = max(3.8, 2.8 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    suptitle = f"{title_prefix} – family '{family}' (variant={variant_dim})"
    fig.suptitle(suptitle, y=0.98)

    xs = np.arange(len(bin_indices))
    width = 0.82 / max(len(variants_present), 1)
    handles = None
    legend_labels: Optional[List[str]] = None

    for ax, pipe in zip(axes.flat, pipes_present):
        subset = stats[stats["pipeline"] == pipe]
        if subset.empty:
            ax.axis("off")
            continue
        for j, variant in enumerate(variants_present):
            data = (
                subset[subset["variant"] == variant]
                .set_index("bin_idx")[["used_mean", "used_sem"]]
                .reindex(bin_indices)
            )
            if data["used_mean"].isnull().all():
                continue
            means = data["used_mean"].to_numpy(dtype=float)
            sems = data["used_sem"].fillna(0.0).to_numpy(dtype=float)
            x_pos = xs - 0.41 + (j + 0.5) * width
            ax.bar(x_pos, means, width=width, label=variant)
            ax.errorbar(x_pos, means, yerr=sems, fmt="none", capsize=2, linewidth=0.8)

        ax.set_title(pipe, fontsize=9)
        upper = max(100.0, overall_max * 1.15 if overall_max > 0 else 100.0)
        ax.set_ylim(0, upper)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylabel("Used (%)")
        ax.set_xticks(xs)
        ax.set_xticklabels(bin_labels, rotation=0)
        ax.set_xlabel("Bin index")

        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()

    for ax in list(axes.flat)[len(pipes_present) :]:
        ax.axis("off")

    if handles and legend_labels:
        fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.01, 0.95), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.86, 0.96])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"residuals_family_{family}_usage_{stage_norm}".replace("/", "_")
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    stats_sorted = stats.sort_values(["pipeline", "variant", "bin_idx"])
    (OUTPUT_DIR / f"{fname}.csv").write_text(stats_sorted.to_csv(index=False))
    plt.close(fig)


def plot_fallback_penalty_family(
    df_fb: pd.DataFrame,
    df_pen: pd.DataFrame,
    family: str,
    pipelines: List[str],
) -> None:
    """
    Dual-row bar plot: top = mean fallback items, bottom = mean offline evictions (penalty events).
    """
    if not pipelines:
        return
    if df_fb.empty or df_pen.empty:
        return

    df_fb = df_fb[df_fb["pipeline"].isin(pipelines)].copy()
    df_pen = df_pen[df_pen["pipeline"].isin(pipelines)].copy()
    if df_fb.empty or df_pen.empty:
        return

    df_fb = df_fb.rename(columns={"metric_mean": "fallback_mean", "metric_sem": "fallback_sem"})
    df_pen = df_pen.rename(columns={"metric_mean": "penalty_mean", "metric_sem": "penalty_sem"})

    merge_keys = ["family_base", "knob_variant", "ratio_variant", "pipeline"]
    merged = df_fb.merge(df_pen, on=merge_keys, how="inner")
    if merged.empty:
        return

    x_dim = choose_x_dimension(merged)
    x_order = order_x(merged, x_dim)

    merged["_x"] = merged[x_dim].astype(str)
    lookup = {(str(r["pipeline"]), str(r["_x"])): r for _, r in merged.iterrows()}

    xs = np.arange(len(x_order))
    n_p = len(pipelines)
    width = 0.82 / max(n_p, 1)

    fig_w = max(10.0, 1.15 * len(x_order) + 4.0)
    fig_h = 6.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 2.4], hspace=0.1)
    ax_fb = fig.add_subplot(gs[0, 0])
    ax_pen = fig.add_subplot(gs[1, 0], sharex=ax_fb)

    for a in (ax_fb, ax_pen):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, axis="y", alpha=0.22)

    title = f"Family: {family}  |  x = {x_dim}"
    fig.suptitle(title, y=0.98, fontsize=13)
    ax_fb.set_title("Fallback items (mean ± SEM)", fontsize=9, color="dimgray", pad=6)
    ax_pen.set_title("Offline evictions (penalty events, mean ± SEM)", fontsize=9, color="dimgray", pad=4)

    for j, pipe in enumerate(pipelines):
        fb_means, fb_sems, pen_means, pen_sems = [], [], [], []
        for xv in x_order:
            r = lookup.get((pipe, str(xv)))
            if r is None or int(r.get("feasible_runs_x", 0)) <= 0:
                fb_means.append(np.nan)
                fb_sems.append(0.0)
                pen_means.append(np.nan)
                pen_sems.append(0.0)
            else:
                fb_means.append(float(r.get("fallback_mean", np.nan)))
                fb_sems.append(float(r.get("fallback_sem", 0.0)) if math.isfinite(float(r.get("fallback_sem", 0.0))) else 0.0)
                pen_means.append(float(r.get("penalty_mean", np.nan)))
                pen_sems.append(float(r.get("penalty_sem", 0.0)) if math.isfinite(float(r.get("penalty_sem", 0.0))) else 0.0)

        x_pos = xs - 0.41 + (j + 0.5) * width
        ax_fb.bar(x_pos, fb_means, width=width, label=pipe)
        ax_fb.errorbar(x_pos, fb_means, yerr=fb_sems, fmt="none", capsize=3, linewidth=1)

        ax_pen.bar(x_pos, pen_means, width=width, label=pipe)
        ax_pen.errorbar(x_pos, pen_means, yerr=pen_sems, fmt="none", capsize=3, linewidth=1)

    ax_fb.set_ylabel("Fallback items")
    ax_pen.set_ylabel("Penalties (evictions)")

    ax_pen.set_xticks(xs)
    ax_pen.set_xticklabels(x_order, rotation=25, ha="right")
    ax_pen.set_xlabel(x_dim)

    ax_fb.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.86, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"fallback_penalty_family_{family}_x-{x_dim}".replace("/", "_")
    fig.savefig(OUTPUT_DIR / f"{fname}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    merged_sorted = merged.sort_values([x_dim, "pipeline"])
    (OUTPUT_DIR / f"{fname}.csv").write_text(merged_sorted.to_csv(index=False))
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cap_lookup = EffectiveCapacityResolver(args.base_config)

    records = list(iter_records())
    if not records:
        print(f"No results found under {RESULTS_ROOT}.")
        return

    df = pd.DataFrame.from_records(records)

    if args.scenarios:
        df = df[df["scenario"].isin(args.scenarios)]
    if args.pipelines:
        df = df[df["pipeline"].isin(args.pipelines)]
    if args.families:
        df = df[df["family_base"].isin(args.families)]

    if df.empty:
        print("No records after filtering.")
        return

    df_raw = df.copy()

    # objective_ratio needs optimal baselines; warn early
    if args.metric == "objective_ratio" and df["objective_ratio"].isnull().all():
        print(
            "No objective_ratio available. Re-run with --compute-optimal "
            "or plot --metric total_objective."
        )
        return

    agg = aggregate(df, args.metric)
    agg_fallback = aggregate(df, "final_fallback")
    agg_penalty = aggregate(df, "evicted_offline")
    if agg.empty:
        print("No aggregates available.")
        return

    for fam, df_fam in agg.groupby("family_base"):
        pipes = plot_family(
            df_fam,
            metric=args.metric,
            topk=args.topk,
            min_feasible_runs=args.min_feasible_runs,
            annotate_infeasible=args.annotate_infeasible,
        )
        if pipes:
            df_fam_raw = df_raw[df_raw["family_base"] == fam]
            df_fam_raw = df_fam_raw[df_fam_raw["pipeline"].isin(pipes)]
            plot_residuals_family(df_fam_raw, fam, pipes, cap_lookup, stage="offline")
            plot_residuals_family(df_fam_raw, fam, pipes, cap_lookup, stage="final")
            df_fallback = agg_fallback[agg_fallback["family_base"] == fam]
            df_penalty = agg_penalty[agg_penalty["family_base"] == fam]
            plot_fallback_penalty_family(df_fallback, df_penalty, fam, pipes)

    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
