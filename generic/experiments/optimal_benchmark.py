from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from generic.core.config import Config, load_config
from generic.data.instance_generators import BaseInstanceGenerator
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.core.utils import set_global_seed
from generic.core.models import Costs, Instance, StepSpec
from generic.offline.solver import OfflineMILPSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the full-horizon (offline+online) MILP and write an aggregated JSON summary."
    )
    parser.add_argument("--config", default="configs/generic/generic.yaml", help="Path to generic YAML.")
    parser.add_argument(
        "--output-dir",
        default="outputs/generic/results",
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
        dest="T_onl",
        type=int,
        default=None,
        help="Optional override for the number of online steps.",
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
    offline_specs = list(instance.offline_steps)
    online_specs = [
        StepSpec(
            step_id=step.step_id,
            cap_matrix=step.cap_matrix,
            feas_matrix=step.feas_matrix,
            feas_rhs=step.feas_rhs,
        )
        for step in instance.online_steps
    ]
    all_steps = offline_specs + online_specs

    costs = Costs(
        assignment_costs=instance.costs.assignment_costs.copy(),
        reassignment_penalty=instance.costs.reassignment_penalty,
        penalty_mode=instance.costs.penalty_mode,
        per_usage_scale=instance.costs.per_usage_scale,
        huge_fallback=instance.costs.huge_fallback,
    )

    return Instance(
        n=instance.n,
        m=instance.m,
        b=instance.b.copy(),
        offline_steps=all_steps,
        costs=costs,
        fallback_option_index=instance.fallback_option_index,
        online_steps=[],
    )


def run_optimal_benchmark(
    cfg: Config,
    *,
    seeds: List[int],
    T_onl: Optional[int],
) -> Dict[str, Any]:
    objectives: List[float] = []
    runtimes: List[float] = []
    offline_statuses: Dict[str, int] = {}
    per_seed: List[Dict[str, Any]] = []
    generator = BaseInstanceGenerator.from_config(cfg)

    for seed in seeds:
        set_global_seed(seed)
        instance = generator.generate_full_instance(cfg, seed=seed, T_onl=T_onl)
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
        per_seed.append(
            {
                "seed": seed,
                "total_objective": float(info.obj_value),
                "runtime": float(info.runtime),
                "status": info.status,
            }
        )

    offline_fail_statuses = {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}
    failures = sum(count for status, count in offline_statuses.items() if status in offline_fail_statuses)

    summary = {
        "seed_count": len(seeds),
        "offline_solver": "generic.offline.solver.OfflineMILPSolver",
        "offline_statuses": offline_statuses,
        "per_seed": per_seed,
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
        T_onl=args.T_onl,
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
