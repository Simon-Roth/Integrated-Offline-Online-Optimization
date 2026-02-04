from __future__ import annotations

import time
import math
from typing import Dict, Tuple, List

import numpy as np

from generic.general_utils import effective_capacity, scalarize_vector
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.models import AssignmentState, Instance
from generic.offline.offline_solver import OfflineMILPSolver
from generic.offline.offline_policies import BaseOfflinePolicy
from generic.offline.models import OfflineSolutionInfo
from binpacking.block_utils import block_dim, extract_volume, split_capacities


class UtilizationPricedDecreasing(BaseOfflinePolicy):
    """
    Volume-ordered heuristic with congestion-based pricing:
    - process offline items in descending volume order
    - each bin keeps a utilization-based price λ_i
    - assignment score is cost_{ji} + λ_i * volume_j
    Encourages distributing load away from congested bins.
    """

    def __init__(self, cfg, *, price_exponent: float | None = None, update_rule: str | None = None, exp_rate: float | None = None) -> None:
        self.cfg = cfg
        self.update_rule = (update_rule or cfg.util_pricing.update_rule).lower()
        self.price_exponent = price_exponent if price_exponent is not None else cfg.util_pricing.price_exponent
        self.exp_rate = exp_rate if exp_rate is not None else cfg.util_pricing.exp_rate

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
        if len(inst.offline_items) == 0:
            runtime = time.perf_counter() - start_time
            state = AssignmentState(
                load=loads.reshape(-1),
                assigned_action=assigned_action,
                offline_evicted=set(),
            )
            info = OfflineSolutionInfo(
                algorithm="UtilizationPricedDecreasing",
                status="FEASIBLE",
                runtime=runtime,
                obj_value=0.0,
                feasible=True,
            )
            return state, info

        eff_caps = effective_capacity(
            split_capacities(inst.b, n),
            self.cfg.slack.enforce_slack,
            self.cfg.slack.fraction,
        )
        avg_cost = float(np.mean(inst.costs.assignment_costs[: len(inst.offline_items), :regular_actions]))
        if not np.isfinite(avg_cost) or avg_cost <= 0.0:
            raise ValueError('Negative or infinite mean cost not fit for UtilDecreasing')
            avg_cost = 1.0
        cost_scale = avg_cost
        lambda_scale = 1.0
        if self.cfg.util_pricing.vector_prices:
            price_cache = np.zeros((regular_actions, d))
        else:
            price_cache = np.zeros(regular_actions)

        # We use this solver (from generic) bc. it allows us to reuse functionality for convenience and consistency
        # However it is equivalent to just a sequential assingment minimizing cost + price
        
        solver = OfflineMILPSolver(self.cfg, log_to_console=False)
        cols = regular_actions + (1 if fallback_idx >= 0 else 0)

        for item_idx, volume in items_sorted:
            remaining = np.maximum(0.0, np.asarray(eff_caps, dtype=float) - loads)
            costs = np.asarray(inst.costs.assignment_costs[item_idx, :cols], dtype=float).reshape(1, -1)
            costs = costs / cost_scale

            for action_idx in range(regular_actions):
                cap = np.asarray(eff_caps[action_idx], dtype=float)
                denom = np.where(cap > 0.0, cap, 1.0)
                norm_volume = volume / denom
                if self.cfg.util_pricing.vector_prices:
                    lam_vec = price_cache[action_idx]
                    costs[0, action_idx] += float(np.dot(lam_vec, norm_volume))
                else:
                    lam = float(price_cache[action_idx])
                    costs[0, action_idx] += lam * scalarize_vector(norm_volume, size_key)

            feas_matrix = np.asarray(inst.offline_items[item_idx].feas_matrix, dtype=float)
            feas_rhs = np.asarray(inst.offline_items[item_idx].feas_rhs, dtype=float)
            cap_matrix = inst.offline_items[item_idx].cap_matrix
            data = build_offline_milp_data_from_arrays(
                cap_matrices=cap_matrix.reshape(1, *cap_matrix.shape),
                costs=costs,
                feas_matrices=[feas_matrix],
                feas_rhs=[feas_rhs],
                b=remaining.reshape(-1),
                fallback_idx=fallback_idx,
                slack_enforce=False,
                slack_fraction=0.0,
            )
            step_state, step_info = solver.solve_from_data(data)
            if not step_info.feasible:
                raise ValueError(f"Item {item_idx} cannot be assigned to any feasible bin.")
            chosen_action = step_state.assigned_action.get(0)
            if chosen_action is None:
                raise ValueError(f"Item {item_idx} could not be assigned by MILP.")

            assigned_action[item_idx] = chosen_action
            if chosen_action < regular_actions:
                loads[chosen_action] += volume
                price_cache[chosen_action] = self._updated_lambda(
                    loads[chosen_action], eff_caps[chosen_action], lambda_scale
                )

        runtime = time.perf_counter() - start_time
        obj_value = self._calculate_objective(assigned_action, inst)

        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_action=assigned_action,
            offline_evicted=set(),
        )
        info = OfflineSolutionInfo(
            algorithm="UtilizationPricedDecreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        return state, info

    def _updated_lambda(self, load: np.ndarray, capacity: np.ndarray, scale: float):
        if np.any(capacity <= 0.0):
            raise KeyError("Capacity <= 0")
        util = np.clip(load / capacity, 0.0, 1.0)
        if self.cfg.util_pricing.vector_prices:
            if self.update_rule == "exponential":
                return scale * (np.exp(self.exp_rate * util) - 1.0)
            return scale * (util ** self.price_exponent)

        util_scalar = scalarize_vector(util, "max")
        if self.update_rule == "exponential":
            return scale * (math.exp(self.exp_rate * util_scalar) - 1.0)
        return scale * (util_scalar ** self.price_exponent)

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
