from __future__ import annotations
from typing import Dict, List, Tuple

from generic.core.models import Instance, AssignmentState
from generic.offline.policies import BaseOfflinePolicy
from generic.core.utils import (
    option_is_feasible,
    scalarize_vector,
    vector_fits,
    residual_vector,
)
from generic.core.models import OfflineSolutionInfo
from bgap.offline.policies._policy_utils import (
    init_loads_and_caps,
    objective_with_fallback,
    sorted_steps_by_volume,
)

class BestFitDecreasing(BaseOfflinePolicy):
    """Best-Fit Decreasing heuristic"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """Solve using Best-Fit Decreasing"""
        import time
        start_time = time.perf_counter()
        
        # Sort steps by volume (decreasing)
        size_key = self.cfg.heuristics.size_key
        steps_sorted = sorted_steps_by_volume(inst, size_key)
        
        # Initialize
        loads, eff_caps = init_loads_and_caps(inst, self.cfg)
        assigned_option = {}
        
        n = inst.n

        # Assign items
        for step_idx, volume in steps_sorted:
            step = inst.offline_steps[step_idx]
            best_bin = -1
            best_remaining = float('inf')
            
            # Find best-fitting bin (minimum remaining space)
            for bin_idx in range(n):
                if (
                    option_is_feasible(step.feas_matrix, step.feas_rhs, bin_idx)
                    and vector_fits(loads[bin_idx], volume, eff_caps[bin_idx])
                ):
                    remaining = residual_vector(loads[bin_idx], volume, eff_caps[bin_idx])
                    remaining_score = scalarize_vector(remaining, self.cfg.heuristics.residual_scalarization)
                    if remaining_score < best_remaining:
                        best_remaining = remaining_score
                        best_bin = bin_idx
            
            # Assign to best bin or fallback
            if best_bin != -1:
                loads[best_bin] += volume
                assigned_option[step_idx] = best_bin
            elif option_is_feasible(
                step.feas_matrix, step.feas_rhs, inst.fallback_option_index
            ):
                assigned_option[step_idx] = inst.fallback_option_index
            else:
                raise ValueError(f"Step {step_idx} cannot be assigned")
        
        runtime = time.perf_counter() - start_time
        obj_value = objective_with_fallback(assigned_option, inst)
        
        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_option=assigned_option,
            offline_evicted_steps=set()
        )
        
        info = OfflineSolutionInfo(
            algorithm="Best-Fit Decreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        
        return state, info
