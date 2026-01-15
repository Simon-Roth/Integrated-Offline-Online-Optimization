from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from generic.config import Config, load_config
from generic.data.instance_generators import generate_instance_with_online
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.general_utils import set_global_seed
from generic.models import Costs, FeasibleGraph, Instance, ItemSpec
from generic.offline.offline_solver import OfflineMILPSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the full-horizon (offline+online) MILP and write an aggregated JSON summary."
    )
    parser.add_argument("--config", default="configs/generic.yaml", help="Path to generic YAML.")
    parser.add_argument(
        "--output-dir",
        default="generic/results",
        help="Directory for aggregated JSON output.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename (default: auto-generated).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Override seed list (defaults to cfg.eval.seeds).",
    )
    parser.add_argument(
        "--m-onl",
        dest="M_onl",
        type=int,
        default=None,
        help="Optional override for the number of online items.",
    )
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _build_full_horizon_instance(instance: Instance) -> Instance:
    """
    Convert an offline+online instance into a single offline instance so the MILP
    solves the full-horizon optimum.
    """
    offline_specs = list(instance.offline_items)
    online_specs = [ItemSpec(id=item.id, volume=item.volume) for item in instance.online_items]
    all_items = offline_specs + online_specs

    offline_feas = instance.offline_feasible.feasible
    if instance.online_feasible is not None:
        online_feas = instance.online_feasible.feasible
        feas_full = np.vstack([offline_feas, online_feas])
    else:
        feas_full = offline_feas.copy()

    costs = Costs(
        assignment_costs=instance.costs.assignment_costs.copy(),
        reassignment_penalty=instance.costs.reassignment_penalty,
        penalty_mode=instance.costs.penalty_mode,
        per_volume_scale=instance.costs.per_volume_scale,
        huge_fallback=instance.costs.huge_fallback,
    )

    return Instance(
        bins=instance.bins,
        offline_items=all_items,
        costs=costs,
        offline_feasible=FeasibleGraph(feasible=feas_full),
        fallback_bin_index=instance.fallback_bin_index,
        online_items=[],
        online_feasible=None,
    )


def run_optimal_benchmark(
    cfg: Config,
    *,
    seeds: List[int],
    M_onl: Optional[int],
) -> Dict[str, Any]:
    objectives: List[float] = []
    runtimes: List[float] = []
    offline_statuses: Dict[str, int] = {}

    for seed in seeds:
        set_global_seed(seed)
        instance = generate_instance_with_online(cfg, seed=seed, M_onl=M_onl)
        full_instance = _build_full_horizon_instance(instance)

        offline_solver = OfflineMILPSolver(cfg)
        data = build_offline_milp_data(full_instance, cfg)
        warm_start = None
        if getattr(cfg.solver, "use_warm_start", False):
            warm_start = offline_solver._generate_warm_start(full_instance)
        _, info = offline_solver.solve_from_data(data, warm_start=warm_start)

        objectives.append(float(info.obj_value))
        runtimes.append(float(info.runtime))
        offline_statuses[info.status] = offline_statuses.get(info.status, 0) + 1

    offline_fail_statuses = {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}
    failures = sum(count for status, count in offline_statuses.items() if status in offline_fail_statuses)

    summary = {
        "seed_count": len(seeds),
        "offline_solver": "generic.offline.offline_solver.OfflineMILPSolver",
        "offline_statuses": offline_statuses,
        "aggregate": {
            "total_objective_mean": _mean(objectives),
            "runtime_mean": _mean(runtimes),
            "failures": int(failures),
        },
    }
    return summary


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else list(cfg.eval.seeds)

    summary = run_optimal_benchmark(
        cfg,
        seeds=seeds,
        M_onl=args.M_onl,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = output_dir / args.output_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"optimal_full_horizon_{timestamp}.json"

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote aggregated results to {output_path}")


if __name__ == "__main__":
    main()
