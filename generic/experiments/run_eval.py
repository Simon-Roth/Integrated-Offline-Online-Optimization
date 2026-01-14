from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

import numpy as np

from generic.config import Config, load_config
from generic.data.instance_generators import generate_instance_with_online
from generic.data.offline_milp_assembly import build_offline_milp_data
from generic.general_utils import set_global_seed
from generic.offline.offline_solver import OfflineMILPSolver
from generic.online.online_solver import OnlineSolver
from generic.online.policies import BaseOnlinePolicy
from generic.online import state_utils


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
        description="Run a generic offline+online evaluation and write an aggregated JSON summary."
    )
    parser.add_argument("--config", default="configs/generic.yaml", help="Path to generic YAML.")
    parser.add_argument(
        "--offline-solver",
        default="generic.offline.offline_solver.OfflineMILPSolver",
        help="Import path to offline solver class.",
    )
    parser.add_argument(
        "--online-policy",
        required=True,
        help="Import path to online policy class.",
    )
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
        "--horizon",
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


def _build_run_summary(
    seed: int,
    cfg: Config,
    offline_info: Any,
    online_info: Any,
    offline_fallback: int,
    final_fallback: int,
    instance: Any,
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "problem": {
            "N": len(instance.bins),
            "M_off": len(instance.offline_items),
            "M_on": len(instance.online_items),
            "dimensions": int(instance.bins[0].capacity.shape[0]) if instance.bins else 1,
        },
        "offline": {
            "status": offline_info.status,
            "objective": float(offline_info.obj_value),
            "runtime": float(offline_info.runtime),
            "fallback_items": int(offline_fallback),
        },
        "online": {
            "status": online_info.status,
            "objective": float(online_info.total_cost),
            "runtime": float(online_info.runtime),
            "fallback_items": int(online_info.fallback_items),
            "evicted_offline": int(online_info.evicted_offline),
        },
        "final_items_in_fallback": int(final_fallback),
        "slack": {
            "enforce_slack": bool(cfg.slack.enforce_slack),
            "fraction": float(cfg.slack.fraction),
            "apply_to_online": bool(getattr(cfg.slack, "apply_to_online", True)),
        },
    }


def run_eval(
    cfg: Config,
    *,
    offline_solver_cls: Type[OfflineMILPSolver],
    online_policy_cls: Type[BaseOnlinePolicy],
    seeds: List[int],
    horizon: Optional[int],
    offline_solver_name: Optional[str] = None,
    online_policy_name: Optional[str] = None,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []

    for seed in seeds:
        set_global_seed(seed)
        instance = generate_instance_with_online(cfg, seed=seed, horizon=horizon)
        offline_solver = offline_solver_cls(cfg)
        # the following check is here because we have some binpacking specific offline algos here in the generic/pipeline_registry.py
        # these do not have a solve_from_data method (working just on A,b,c) but work on instance (volumes, capacities, ...) for simplicity and understanding
        if hasattr(offline_solver, "solve_from_data"):
            data = build_offline_milp_data(instance, cfg)
            warm_start = None
            if getattr(cfg.solver, "use_warm_start", False) and hasattr(
                offline_solver, "_generate_warm_start"
            ):
                warm_start = offline_solver._generate_warm_start(instance)
            offline_state, offline_info = offline_solver.solve_from_data(
                data, warm_start=warm_start
            )
        else:
            offline_state, offline_info = offline_solver.solve(instance)

        online_policy = online_policy_cls(cfg)
        online_solver = OnlineSolver(cfg, online_policy)
        final_state, online_info = online_solver.run(instance, offline_state)

        offline_fallback = state_utils.count_fallback_items(offline_state, instance)
        final_fallback = state_utils.count_fallback_items(final_state, instance)

        runs.append(
            _build_run_summary(
                seed,
                cfg,
                offline_info,
                online_info,
                offline_fallback,
                final_fallback,
                instance,
            )
        )

    offline_obj = [run["offline"]["objective"] for run in runs]
    offline_runtime = [run["offline"]["runtime"] for run in runs]
    online_obj = [run["online"]["objective"] for run in runs]
    online_runtime = [run["online"]["runtime"] for run in runs]
    total_obj = [off + on for off, on in zip(offline_obj, online_obj)]
    offline_fail_statuses = {"INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED"}
    online_fail_statuses = {"INFEASIBLE"}
    offline_failures = sum(
        1 for run in runs if run["offline"]["status"] in offline_fail_statuses
    )
    online_failures = sum(
        1 for run in runs if run["online"]["status"] in online_fail_statuses
    )
    offline_statuses: Dict[str, int] = {}
    online_statuses: Dict[str, int] = {}
    for run in runs:
        offline_status = run["offline"]["status"]
        online_status = run["online"]["status"]
        offline_statuses[offline_status] = offline_statuses.get(offline_status, 0) + 1
        online_statuses[online_status] = online_statuses.get(online_status, 0) + 1

    summary = {
        "seed_count": len(seeds),
        "offline_solver": offline_solver_name,
        "online_policy": online_policy_name,
        "offline_statuses": offline_statuses,
        "online_statuses": online_statuses,
        "total_objective_mean": _mean(total_obj),
        "aggregate": {
            "offline_objective_mean": _mean(offline_obj),
            "offline_runtime_mean": _mean(offline_runtime),
            "online_objective_mean": _mean(online_obj),
            "online_runtime_mean": _mean(online_runtime),
            "offline_failures": int(offline_failures),
            "online_failures": int(online_failures),
        },
    }
    return summary


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else list(cfg.eval.seeds)

    offline_solver_cls = _import_symbol(args.offline_solver)
    online_policy_cls = _import_symbol(args.online_policy)

    # if you run an online policy like primal dual here, be sure to compute prices beforehand
    # prices are only computed manually in run_multiple_evals.py each time a pipeline needs them

    summary = run_eval(
        cfg,
        offline_solver_cls=offline_solver_cls,
        online_policy_cls=online_policy_cls,
        seeds=seeds,
        horizon=args.horizon,
        offline_solver_name=args.offline_solver,
        online_policy_name=args.online_policy,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = output_dir / args.output_name
    else:
        policy_label = args.online_policy.split(".")[-1]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{policy_label}_{timestamp}.json"

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote aggregated results to {output_path}")


if __name__ == "__main__":
    main()
