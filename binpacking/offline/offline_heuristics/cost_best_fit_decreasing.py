from __future__ import annotations

import time
from typing import Dict, Tuple, List

import numpy as np

from generic.general_utils import effective_capacity, scalarize_vector, vector_fits, residual_vector
from generic.models import AssignmentState, Instance
from generic.offline.offline_policies import BaseOfflinePolicy
from generic.offline.models import OfflineSolutionInfo
from binpacking.block_utils import block_dim, extract_volume, split_capacities


class CostAwareBestFitDecreasing(BaseOfflinePolicy):
    """
    Greedy offline heuristic that sorts items by volume (descending) and assigns
    each to the feasible bin with the lowest assignment cost. Ties on cost are
    broken by the smallest residual capacity to avoid fragmentation.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        start_time = time.perf_counter()

        size_key = self.cfg.heuristics.size_key
        n = inst.n
        m = inst.m
        d = block_dim(n, m)
        items_sorted: List[Tuple[int, np.ndarray]] = sorted(
            ((idx, extract_volume(item.cap_matrix, n, m)) for idx, item in enumerate(inst.offline_items)),
            key=lambda pair: scalarize_vector(pair[1], size_key),
            reverse=True,
        )

        regular_actions = n
        fallback_idx = inst.fallback_action_index
        loads = np.zeros((regular_actions, d))
        assigned_action: Dict[int, int] = {}

        eff_caps = effective_capacity(
            split_capacities(inst.b, n),
            self.cfg.slack.enforce_slack,
            self.cfg.slack.fraction,
        )

        for item_idx, volume in items_sorted:
            best_bin = None
            best_cost = float("inf")
            best_residual = float("inf")

            for bin_idx in range(regular_actions):
                if inst.offline_feasible.feasible[item_idx, bin_idx] != 1:
                    continue
                if not vector_fits(loads[bin_idx], volume, eff_caps[bin_idx], 1e-9):
                    continue
                cost = float(inst.costs.assignment_costs[item_idx, bin_idx])
                residual = residual_vector(loads[bin_idx], volume, eff_caps[bin_idx])
                residual_score = scalarize_vector(residual, self.cfg.heuristics.residual_scalarization)
                if (
                    cost < best_cost - 1e-9
                    or (abs(cost - best_cost) <= 1e-9 and residual_score < best_residual)
                ):
                    best_cost = cost
                    best_residual = residual_score
                    best_bin = bin_idx

            if best_bin is not None:
                loads[best_bin] += volume
                assigned_action[item_idx] = best_bin
                continue

            if (
                self.cfg.problem.fallback_is_enabled and self.cfg.problem.fallback_allowed_offline
                and fallback_idx >= 0
                and inst.offline_feasible.feasible[item_idx, fallback_idx] == 1
            ):
                assigned_action[item_idx] = fallback_idx
                continue

            raise ValueError(f"Item {item_idx} cannot be assigned to any feasible bin.")

        runtime = time.perf_counter() - start_time
        obj_value = self._calculate_objective(assigned_action, inst)

        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_action=assigned_action,
            offline_evicted=set(),
        )
        info = OfflineSolutionInfo(
            algorithm="CostAwareBestFitDecreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        return state, info

    def _calculate_objective(
        self, assigned_action: Dict[int, int], inst: Instance
    ) -> float:
        total_cost = 0.0
        fallback_idx = inst.fallback_action_index
        for item_idx, action_idx in assigned_action.items():
            if action_idx < fallback_idx or fallback_idx < 0:
                total_cost += float(inst.costs.assignment_costs[item_idx, action_idx])
            else:
                total_cost += inst.costs.huge_fallback
        return total_cost
