from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
from typing import Dict, Any, List, Callable, Tuple, Optional
import numpy as np
from generic.models import AssignmentState, Instance
from generic.offline.models import OfflineSolutionInfo
from binpacking.offline.offline_heuristics.core import HeuristicSolutionInfo
from generic.config import Config, SlackConfig, DLAConfig
from binpacking.data.instance_generators import generate_instance_with_online
from generic.online.online_solver import OnlineSolver
from generic.online.state_utils import count_fallback_items, effective_capacities
from generic.general_utils import scalarize_vector

OfflineSolverProtocol = Callable[[Config], object]
OnlinePolicyProtocol = Callable[[Config], object]

def save_results(method_name: str, state: AssignmentState, 
                 info: OfflineSolutionInfo | HeuristicSolutionInfo,
                 problem_config: Dict[str, int],
                 seed: int,
                 output_dir: str = "binpacking/results") -> None:
    """Save method results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = {
        'method': method_name,
        'problem': problem_config,
        'seed': seed,
        'runtime': info.runtime,
        'obj_value': info.obj_value,
        'status': info.status if hasattr(info, 'status') else 'HEURISTIC',
        'items_in_fallback': sum(1 for bin_id in state.assigned_bin.values() 
                                if bin_id >= problem_config['N']),
        'assignments': {str(k): int(v) for k, v in state.assigned_bin.items()}
    }
    
    filename = f"{output_dir}/{method_name}_N{problem_config['N']}_M{problem_config['M_off']}_seed{seed}.json"
    Path(filename).write_text(json.dumps(result, indent=2))
    print(f"Results saved to {filename}")

def load_results(results_dir: str = "binpacking/results") -> list[Dict[str, Any]]:
    """Load all result files from directory."""
    results = []
    for file in Path(results_dir).glob("*.json"):
        results.append(json.loads(file.read_text()))
    return results


def build_empty_offline_solution(instance: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
    """
    Return a zero-load assignment state and stub OfflineSolutionInfo when no offline items exist.
    """
    fallback_dim = instance.fallback_bin_index + 1
    dims = instance.bins[0].capacity.shape[0] if instance.bins else 1
    state = AssignmentState(
        load=np.zeros((fallback_dim, dims), dtype=float),
        assigned_bin={},
    )
    info = OfflineSolutionInfo(
        status="NO_OFFLINE_ITEMS",
        obj_value=0.0,
        mip_gap=0.1,
        runtime=0.0,
        assignments=np.zeros((0, fallback_dim), dtype=int),
    )
    return state, info


def save_combined_result(
    filename_prefix: str,
    result: Dict[str, Any],
    output_dir: str = "binpacking/results",
) -> str:
    """
    Persist combined offline+online experiment result (single JSON structure).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{output_dir}/{filename_prefix}.json"
    Path(filename).write_text(json.dumps(result, indent=2))
    print(f"Results saved to {filename}")
    return filename


def run_offline_online_pipeline(
    cfg: Config,
    seed: int,
    offline_solver,
    online_policy,
    *,
    instance: Optional[Instance] = None,
):
    """
    Generate an instance with online arrivals, solve the offline phase, then run the
    provided online policy. Returns the generated instance, the offline state, the final
    online state, and their respective info objects.
    """
    if instance is None:
        instance = generate_instance_with_online(cfg, seed=seed)

    if len(instance.offline_items) == 0:
        offline_state, offline_info = build_empty_offline_solution(instance)
    else:
        offline_state, offline_info = offline_solver.solve(instance)
    solver = OnlineSolver(cfg, online_policy)
    final_state, online_info = solver.run(instance, offline_state)
    return instance, offline_state, final_state, offline_info, online_info


def build_pipeline_summary(
    pipeline_name: str,
    seed: int,
    cfg: Config,
    instance,
    offline_state: AssignmentState,
    final_state: AssignmentState,
    offline_info,
    online_info,
    offline_method: str,
    online_method: str,
    *,
    slack_config: Optional[SlackConfig] = None,
    dla_config: Optional[DLAConfig] = None,
) -> Dict[str, Any]:
    """
    Assemble a JSON-safe summary for an offline+online pipeline run.
    """
    problem_meta = {
        "N": len(instance.bins),
        "M_off": len(instance.offline_items),
        "M_on": len(instance.online_items),
        "dimensions": instance.bins[0].capacity.shape[0] if instance.bins else 1,
    }
    # Residual capacity after offline phase (regular bins only), w.r.t. the effective
    # capacities used by the offline solver (slack applied if configured).
    eff_caps = effective_capacities(instance, cfg, use_slack=cfg.slack.enforce_slack)
    residual_caps_vec = [
        np.maximum(0.0, cap - offline_state.load[i]) for i, cap in enumerate(eff_caps)
    ]
    final_residual_caps_vec = [
        np.maximum(
            0.0,
            cap
            - (final_state.load[i] if i < len(final_state.load) else 0.0),
        )
        for i, cap in enumerate(eff_caps)
    ]
    residual_caps = [
        scalarize_vector(vec, cfg.heuristics.residual_scalarization) for vec in residual_caps_vec
    ]
    final_residual_caps = [
        scalarize_vector(vec, cfg.heuristics.residual_scalarization) for vec in final_residual_caps_vec
    ]
    offline_loads_vec = offline_state.load.tolist()
    final_loads_vec = final_state.load.tolist()
    offline_loads = [
        scalarize_vector(np.asarray(v), cfg.heuristics.size_key) for v in offline_loads_vec
    ]
    final_loads = [
        scalarize_vector(np.asarray(v), cfg.heuristics.size_key) for v in final_loads_vec
    ]

    offline_obj = float(offline_info.obj_value)
    offline_status = getattr(offline_info, "status", None)
    if offline_status is None:
        offline_status = "FEASIBLE" if getattr(offline_info, "feasible", False) else "INFEASIBLE"

    offline_result = {
        "method": offline_method,
        "status": offline_status,
        "runtime": float(offline_info.runtime),
        "obj_value": offline_obj,
        "mip_gap": float(getattr(offline_info, "mip_gap", float("nan"))) if hasattr(offline_info, "mip_gap") else None,
        "items_in_fallback": count_fallback_items(offline_state, instance),
    }

    online_result = {
        "policy": online_method,
        "status": online_info.status,
        "runtime": float(online_info.runtime),
        "total_cost": float(online_info.total_cost),
        "fallback_items": online_info.fallback_items,
        "evicted_offline": online_info.evicted_offline,
        "total_objective": float(offline_obj + online_info.total_cost),
    }

    summary = {
        "pipeline": pipeline_name,
        "seed": seed,
        "problem": problem_meta,
        "offline": offline_result,
        "online": online_result,
        "final_items_in_fallback": count_fallback_items(final_state, instance),
        "offline_assignments": {str(k): int(v) for k, v in offline_state.assigned_bin.items()},
        "offline_residual_capacities": residual_caps,
        "final_residual_capacities": final_residual_caps,
        "offline_loads": offline_loads,
        "final_loads": final_loads,
        "offline_residual_capacities_vec": [vec.tolist() for vec in residual_caps_vec],
        "final_residual_capacities_vec": [vec.tolist() for vec in final_residual_caps_vec],
        "offline_loads_vec": offline_loads_vec,
        "final_loads_vec": final_loads_vec,
    }
    if slack_config is not None:
        summary["slack"] = {
            "enforce_slack": bool(slack_config.enforce_slack),
            "fraction": float(slack_config.fraction),
            "apply_to_online": bool(slack_config.apply_to_online),
        }
    if dla_config is not None:
        summary["dla"] = {
            "epsilon": float(dla_config.epsilon),
            "log_prices": bool(dla_config.log_prices),
            "output_dir": str(dla_config.output_dir),
            "min_phase_len": int(dla_config.min_phase_len),
            "use_offline_slack": bool(dla_config.use_offline_slack),
        }
    return summary
