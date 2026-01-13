from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from generic.models import Instance, AssignmentState
from generic.offline.offline_policies import BaseOfflinePolicy
from generic.general_utils import effective_capacity, scalarize_vector, vector_fits
from binpacking.offline.offline_heuristics.core import HeuristicSolutionInfo

class FirstFitDecreasing(BaseOfflinePolicy):
    """First-Fit Decreasing heuristic for bin packing"""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def solve(self, inst: Instance) -> Tuple[AssignmentState, HeuristicSolutionInfo]:
        """Solve using First-Fit Decreasing"""
        import time
        start_time = time.perf_counter()
        
        # Sort items by volume (decreasing)
        size_key = self.cfg.heuristics.size_key
        items_sorted = sorted(
            enumerate(inst.offline_items),
            key=lambda x: scalarize_vector(x[1].volume, size_key),
            reverse=True,
        )
        
        # Initialize state
        dims = inst.bins[0].capacity.shape[0] if inst.bins else 1
        loads = np.zeros((len(inst.bins) + 1, dims))  # +1 for fallback
        assigned_bin = {}
        
        # Effective capacities with slack
        eff_caps = [
            effective_capacity(
                bin.capacity,
                self.cfg.slack.enforce_slack,
                self.cfg.slack.fraction,
            )
            for bin in inst.bins
        ]
        
        # Assign items
        for item_idx, item in items_sorted:
            assigned = False
            
            # Try regular bins first
            for bin_idx in range(len(inst.bins)):
                # Check feasibility and capacity
                if (
                    inst.feasible.feasible[item_idx, bin_idx] == 1
                    and vector_fits(loads[bin_idx], item.volume, eff_caps[bin_idx])
                ):
                    
                    loads[bin_idx] += item.volume
                    assigned_bin[item_idx] = bin_idx
                    assigned = True
                    break
            
            # Use fallback if enabled and no regular bin found
            if not assigned:
                fallback_idx = len(inst.bins)
                if (
                    self.cfg.problem.fallback_is_enabled
                    and inst.feasible.feasible[item_idx, fallback_idx] == 1
                ):
                    fallback_idx = len(inst.bins)
                    loads[fallback_idx] += item.volume
                    assigned_bin[item_idx] = fallback_idx
                else:
                    raise ValueError(f"Item {item_idx} cannot be assigned (no fallback)")
        
        runtime = time.perf_counter() - start_time
        
        # Calculate objective
        obj_value = self._calculate_objective(assigned_bin, inst)
        
        # Create assignment state
        state = AssignmentState(
            load=loads,
            assigned_bin=assigned_bin,
            offline_evicted=set()
        )
        
        info = HeuristicSolutionInfo(
            algorithm="First-Fit Decreasing",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
            items_in_fallback=sum(1 for bin_id in assigned_bin.values() 
                                if bin_id == len(inst.bins)),
            utilization=float(
                np.mean(
                    [
                        np.max(loads[i] / inst.bins[i].capacity)
                        for i in range(len(inst.bins))
                    ]
                )
            ),
        )
        
        return state, info
    
    def _calculate_objective(self, assigned_bin: Dict[int, int], inst: Instance) -> float:
        """Calculate objective value"""
        total_cost = 0.0
        for item_idx, bin_idx in assigned_bin.items():
            if bin_idx < len(inst.bins):  # Regular bin
                total_cost += inst.costs.assign[item_idx, bin_idx]
            else:  # Fallback bin
                total_cost += inst.costs.huge_fallback
        return total_cost
