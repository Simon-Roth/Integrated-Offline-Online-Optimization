from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

from binpacking.config import load_config
from generic.general_utils import set_global_seed
from generic.models import AssignmentState
from binpacking.experiments.pipeline_runner import run_pipeline
from binpacking.experiments.pipeline_registry import PIPELINES

from binpacking.data.generators import generate_instance_with_online
from binpacking.experiments.optimal_benchmark import solve_full_horizon_optimum
from generic.online.state_utils import count_fallback_items
from binpacking.experiments.utils import save_combined_result
from binpacking.offline.offline_solver import OfflineMILPSolver
from generic.offline.models import OfflineSolutionInfo
from binpacking.offline.offline_heuristics.core import HeuristicSolutionInfo
from binpacking.experiments.offline_cache import (
    compute_config_signature,
    load_cached_full_horizon,
    save_cached_full_horizon,
)
import copy

OfflineResult = Tuple[AssignmentState, OfflineSolutionInfo | HeuristicSolutionInfo]


def main() -> None:
    config_path = Path("configs/binpacking.yaml")
    cfg = load_config(config_path)
    base_seed = cfg.eval.seeds[0]
    config_sig = compute_config_signature(config_path)
    
    # Compute full-horizon optimum once for reference
    compute_full_horizon_baseline(base_seed, config_sig)
    set_global_seed(base_seed)
    shared_instance = generate_instance_with_online(cfg, seed=base_seed)
    offline_cache: Dict[str, OfflineResult] = {}
    
    print("Running offline+online pipelines ...")
    for spec in PIPELINES:
        # fresh copy per run to avoid cross-talk
        cfg_run = load_config(config_path)
        set_global_seed(base_seed)

        cached_solution: OfflineResult | None = offline_cache.get(spec.offline_label)
        if cached_solution is None and shared_instance.offline_items:
            solver = spec.offline_factory(cfg_run)
            offline_state, offline_info = solver.solve(copy.deepcopy(shared_instance))
            cached_solution = (offline_state, offline_info)
            offline_cache[spec.offline_label] = cached_solution

        print(f"\n{'=' * 60}")
        print(f"Running {spec.name}")
        print("=" * 60)

        summary, path = run_pipeline(
            cfg_run,
            spec,
            seed=base_seed,
            instance=shared_instance,
            offline_solution=cached_solution,
        )
        print(
            f"{summary['pipeline']}: offline {summary['offline']['status']} "
            f"({summary['offline']['runtime']:.3f}s), "
            f"online {summary['online']['status']} "
            f"({summary['online']['runtime']:.3f}s)"
        )
        print(f"Saved result to {path}")

    print("\nAll pipelines completed!")
    print("Results saved to binpacking/results/ directory")

def compute_full_horizon_baseline(base_seed, config_sig): 
    # Compute full-horizon optimum once for reference
    cfg_opt = load_config("configs/binpacking.yaml")
    # cfg_opt.solver.use_warm_start = True
    # cfg_opt.solver.warm_start_heuristic = "CBFD"
    set_global_seed(base_seed)
    base_instance = generate_instance_with_online(cfg_opt, seed=base_seed)
    offline_count = len(base_instance.offline_items)
    online_count = len(base_instance.online_items)
    if offline_count + online_count == 0:
        print("Skipping optimal full-horizon benchmark (no offline or online items configured).")
        return
    cached = load_cached_full_horizon(config_sig, base_seed)
    if cached is not None:
        optimal_state, optimal_info = cached
    else:
        optimal_state, optimal_info = solve_full_horizon_optimum(
            cfg_opt,
            copy.deepcopy(base_instance),
            lambda cfg_: OfflineMILPSolver(cfg_, time_limit=300, mip_gap=0.01, log_to_console=False),
        )
        save_cached_full_horizon(config_sig, base_seed, optimal_state, optimal_info)
    optimal_fallback = count_fallback_items(optimal_state, base_instance)
    optimal_summary = {
        "pipeline": "OPTIMAL_FULL_HORIZON",
        "seed": base_seed,
        "problem": {
            "N": cfg_opt.problem.N,
            "M_off": offline_count,
            "M_on": online_count,
        },
        "offline": {
            "method": "FullHorizonMILP",
            "status": optimal_info.status,
            "runtime": float(optimal_info.runtime),
            "obj_value": float(optimal_info.obj_value),
            "mip_gap": float(optimal_info.mip_gap),
            "items_in_fallback": optimal_fallback,
        },
        "online": {
            "policy": "FullHorizonMILP",
            "status": optimal_info.status,
            "runtime": 0.0,
            "total_cost": 0.0,
            "fallback_items": optimal_fallback,
            "evicted_offline": 0,
            "total_objective": float(optimal_info.obj_value),
        },
        "final_items_in_fallback": optimal_fallback,
        "offline_assignments": {str(k): int(v) for k, v in optimal_state.assigned_bin.items()},
    }
    optimal_prefix = (
        f"pipeline_optimal_full_horizon_N{cfg_opt.problem.N}_"
        f"Moff{offline_count}_Mon{online_count}_seed{base_seed}"
    )
    optimal_path = save_combined_result(optimal_prefix, optimal_summary)
    print(f"Optimal full-horizon benchmark saved to {optimal_path}")
    
if __name__ == "__main__":
    main()
