from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

from generic.models import Instance, AssignmentState
from generic.offline.offline_policies import BaseOfflinePolicy
from generic.general_utils import effective_capacity, scalarize_vector, vector_fits, residual_vector
from generic.offline.models import OfflineSolutionInfo
from binpacking.block_utils import block_dim, extract_volume, split_capacities

class BestFitDecreasing(BaseOfflinePolicy):
    """Best-Fit Decreasing heuristic"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """Solve using Best-Fit Decreasing"""
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
        
        # Initialize
        loads = np.zeros((n, d))
        assigned_action = {}
        
        eff_caps = effective_capacity(
            split_capacities(inst.b, n),
            self.cfg.slack.enforce_slack,
            self.cfg.slack.fraction,
        )
        
        # Assign items
        for item_idx, item in items_sorted:
            best_bin = -1
            best_remaining = float('inf')
            volume = extract_volume(item.cap_matrix, n, m)
            
            # Find best-fitting bin (minimum remaining space)
            for bin_idx in range(n):
                if (
                    inst.offline_feasible.feasible[item_idx, bin_idx] == 1
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
                assigned_action[item_idx] = best_bin
            elif (
                self.cfg.problem.fallback_is_enabled and self.cfg.problem.fallback_allowed_offline
                and inst.offline_feasible.feasible[item_idx, inst.fallback_action_index] == 1
            ):
                assigned_action[item_idx] = inst.fallback_action_index
            else:
                raise ValueError(f"Item {item_idx} cannot be assigned")
        
        runtime = time.perf_counter() - start_time
        obj_value = self._calculate_objective(assigned_action, inst)
        
        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_action=assigned_action,
            offline_evicted=set()
        )
        
        info = OfflineSolutionInfo(
            algorithm="Best-Fit Decreasing",
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
