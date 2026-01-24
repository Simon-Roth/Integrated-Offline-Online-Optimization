from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

from generic.models import Instance, AssignmentState
from generic.offline.offline_policies import BaseOfflinePolicy
from generic.general_utils import (
    action_is_feasible,
    effective_capacity,
    scalarize_vector,
    vector_fits,
)
from generic.offline.models import OfflineSolutionInfo
from binpacking.block_utils import block_dim, extract_volume, split_capacities

class FirstFitDecreasing(BaseOfflinePolicy):
    """First-Fit Decreasing heuristic for bin packing"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """Solve using First-Fit Decreasing"""
        import time
        start_time = time.perf_counter()
        
        # Sort items by volume (decreasing)
        size_key = self.cfg.heuristics.size_key
        n = inst.n
        m = inst.m
        d = block_dim(n, m)
        items_sorted = sorted(
            enumerate(inst.offline_items),
            key=lambda x: scalarize_vector(extract_volume(x[1].cap_matrix, n, m), size_key),
            reverse=True,
        )
        
        # Initialize state
        loads = np.zeros((n, d))
        assigned_action = {}
        
        # Effective capacities with slack
        eff_caps = effective_capacity(
            split_capacities(inst.b, n),
            self.cfg.slack.enforce_slack,
            self.cfg.slack.fraction,
        )
        
        # Assign items
        for item_idx, item in items_sorted:
            assigned = False
            volume = extract_volume(item.cap_matrix, n, m)
            
            # Try regular bins first
            for bin_idx in range(n):
                # Check feasibility and capacity
                if (
                    action_is_feasible(item.feas_matrix, item.feas_rhs, bin_idx)
                    and vector_fits(loads[bin_idx], volume, eff_caps[bin_idx])
                ):
                    
                    loads[bin_idx] += volume
                    assigned_action[item_idx] = bin_idx
                    assigned = True
                    break
            
            # Use fallback if enabled and no regular bin found
            if not assigned:
                if (
                    inst.fallback_action_index >= 0
                    and action_is_feasible(item.feas_matrix, item.feas_rhs, inst.fallback_action_index)
                ):
                    assigned_action[item_idx] = inst.fallback_action_index
                else:
                    raise ValueError(f"Item {item_idx} cannot be assigned (no fallback)")
        
        runtime = time.perf_counter() - start_time
        
        # Calculate objective
        obj_value = self._calculate_objective(assigned_action, inst)
        
        # Create assignment state
        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_action=assigned_action,
            offline_evicted=set()
        )
        
        info = OfflineSolutionInfo(
            algorithm="First-Fit Decreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        
        return state, info
    
    def _calculate_objective(self, assigned_action: Dict[int, int], inst: Instance) -> float:
        """Calculate objective value"""
        total_cost = 0.0
        fallback_idx = inst.fallback_action_index
        for item_idx, action_idx in assigned_action.items():
            if action_idx < fallback_idx or fallback_idx < 0:  # Regular action
                total_cost += inst.costs.assignment_costs[item_idx, action_idx]
            else:  # Fallback action
                total_cost += inst.costs.huge_fallback
        return total_cost
