from __future__ import annotations
from typing import Dict, List, Tuple

from generic.core.models import Instance, AssignmentState
from generic.offline.policies import BaseOfflinePolicy
from generic.core.utils import option_is_feasible, vector_fits
from generic.core.models import OfflineSolutionInfo
from binpacking.offline.policies._policy_utils import (
    init_loads_and_caps,
    objective_with_fallback,
    sorted_steps_by_volume,
)

class FirstFitDecreasing(BaseOfflinePolicy):
    """First-Fit Decreasing heuristic for bin packing"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """Solve using First-Fit Decreasing"""
        import time
        start_time = time.perf_counter()
        
        # Sort steps by volume (decreasing)
        size_key = self.cfg.heuristics.size_key
        steps_sorted = sorted_steps_by_volume(inst, size_key)
        
        # Initialize state
        loads, eff_caps = init_loads_and_caps(inst, self.cfg)
        assigned_option = {}
        
        n = inst.n

        # Assign steps
        for step_idx, volume in steps_sorted:
            step = inst.offline_steps[step_idx]
            assigned = False
            
            # Try regular bins first
            for bin_idx in range(n):
                # Check feasibility and capacity
                if (
                    option_is_feasible(step.feas_matrix, step.feas_rhs, bin_idx)
                    and vector_fits(loads[bin_idx], volume, eff_caps[bin_idx])
                ):
                    
                    loads[bin_idx] += volume
                    assigned_option[step_idx] = bin_idx
                    assigned = True
                    break
            
            # Use fallback if enabled and no regular bin found
            if not assigned:
                if (
                    inst.fallback_option_index >= 0
                    and option_is_feasible(step.feas_matrix, step.feas_rhs, inst.fallback_option_index)
                ):
                    assigned_option[step_idx] = inst.fallback_option_index
                else:
                    raise ValueError(f"Step {step_idx} cannot be assigned (no fallback)")
        
        runtime = time.perf_counter() - start_time
        
        # Calculate objective
        obj_value = objective_with_fallback(assigned_option, inst)
        
        # Create assignment state
        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_option=assigned_option,
            offline_evicted_steps=set()
        )
        
        info = OfflineSolutionInfo(
            algorithm="First-Fit Decreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        
        return state, info
