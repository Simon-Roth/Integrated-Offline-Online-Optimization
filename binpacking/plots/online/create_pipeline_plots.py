from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_PATTERN = "pipeline_*.json"
RESULTS_DIR = Path("binpacking/results")
OUTPUT_DIR = Path("binpacking/plots/online/results")


def load_pipeline_records() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for file_path in RESULTS_DIR.glob(RESULTS_PATTERN):
        data = json.loads(file_path.read_text())
        records.append(
            {
                "pipeline": data["pipeline"],
                "offline_method": data["offline"]["method"],
                "online_method": data["online"]["policy"],
                "runtime_offline": float(data["offline"]["runtime"]),
                "runtime_online": float(data["online"]["runtime"]),
                "obj_offline": float(data["offline"]["obj_value"]),
                "total_cost_online": float(data["online"]["total_cost"]),
                "total_objective": float(data["online"]["total_objective"]),
                "fallback_offline": int(data["offline"]["items_in_fallback"]),
                "fallback_final": int(data["final_items_in_fallback"]),
                "seed": int(data["seed"]),
                "status_offline": data.get("offline", {}).get("status", "UNKNOWN"),
                "status_online": data.get("online", {}).get("status", "UNKNOWN"),
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["pipeline", "offline_method", "online_method"])
        .agg(
            runtime_offline=("runtime_offline", "mean"),
            runtime_online=("runtime_online", "mean"),
            obj_offline=("obj_offline", "mean"),
            total_cost_online=("total_cost_online", "mean"),
            total_objective=("total_objective", "mean"),
            fallback_offline=("fallback_offline", "mean"),
            fallback_final=("fallback_final", "mean"),
            runs=("seed", "nunique"),
            status_offline=("status_offline", "first"),
            status_online=("status_online", "first"),
        )
        .reset_index()
    )
    return grouped


def create_plots(agg: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    good_statuses = {"COMPLETED", "OPTIMAL"}
    infeasible_pipelines = agg.loc[~agg["status_online"].isin(good_statuses), "pipeline"].tolist()

    # Figure 1: Offline vs Online runtime
    runtime_df = agg[["pipeline", "runtime_offline", "runtime_online"]].melt(
        id_vars="pipeline",
        var_name="phase",
        value_name="runtime",
    )
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=runtime_df,
        x="pipeline",
        y="runtime",
        hue="phase",
        palette=["#1f77b4", "#ff7f0e"],
    )
    ax.set_title("Runtime Breakdown by Pipeline")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Runtime (s)")
    ax.legend(title="")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pipeline_runtime.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "pipeline_runtime.pdf", dpi=300)
    plt.close()

    # Figure 2: Objective comparison
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=agg.sort_values("total_objective"),
        x="pipeline",
        y="total_objective",
        palette="crest",
    )
    title = "Total Objective (Offline + Online Cost)"
    if infeasible_pipelines:
        title += f"  (Infeasible: {', '.join(infeasible_pipelines)})"
    ax.set_title(title)
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Objective Value")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pipeline_objective.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "pipeline_objective.pdf", dpi=300)
    plt.close()

    # Figure 3: Fallback usage before vs after online phase
    fallback_df = agg[["pipeline", "fallback_offline", "fallback_final"]].melt(
        id_vars="pipeline",
        var_name="stage",
        value_name="fallback_items",
    )
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=fallback_df,
        x="pipeline",
        y="fallback_items",
        hue="stage",
        palette=["#2ca02c", "#d62728"],
    )
    ax.set_title("Fallback Usage Before vs After Online Phase")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Items in Fallback")
    ax.legend(title="")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pipeline_fallback.png", dpi=300)
    plt.savefig(OUTPUT_DIR / "pipeline_fallback.pdf", dpi=300)
    plt.close()


def main() -> None:
    df = load_pipeline_records()
    if df.empty:
        print("No pipeline results found. Run binpacking/experiments/run_all_pipelines.py first.")
        return

    agg = aggregate_results(df)
    print(f"Loaded {len(df)} runs across {len(agg)} pipelines.")
    create_plots(agg)
    print(f"Pipeline plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
