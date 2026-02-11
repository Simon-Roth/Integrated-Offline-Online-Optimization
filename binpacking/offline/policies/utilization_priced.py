from __future__ import annotations

import time
import math
from typing import Dict, Tuple, List

import numpy as np

from generic.core.utils import scalarize_vector
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.core.models import AssignmentState, Instance
from generic.offline.solver import OfflineMILPSolver
from generic.offline.policies import BaseOfflinePolicy
from generic.core.models import OfflineSolutionInfo
from binpacking.offline.policies._policy_utils import (
    init_loads_and_caps,
    objective_with_fallback,
    sorted_steps_by_volume,
)


class UtilizationPricedDecreasing(BaseOfflinePolicy):
    """
    Volume-ordered heuristic with congestion-based pricing:
    - process offline steps in descending volume order
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
        steps_sorted = sorted_steps_by_volume(inst, size_key)

        regular_options = inst.n
        fallback_idx = inst.fallback_option_index
        loads, eff_caps = init_loads_and_caps(inst, self.cfg)
        d = eff_caps.shape[1]
        assigned_option: Dict[int, int] = {}
        if len(inst.offline_steps) == 0:
            runtime = time.perf_counter() - start_time
            state = AssignmentState(
                load=loads.reshape(-1),
                assigned_option=assigned_option,
                offline_evicted_steps=set(),
            )
            info = OfflineSolutionInfo(
                algorithm="UtilizationPricedDecreasing",
                status="FEASIBLE",
                runtime=runtime,
                obj_value=0.0,
                feasible=True,
            )
            return state, info

        avg_cost = float(np.mean(inst.costs.assignment_costs[: len(inst.offline_steps), :regular_options]))
        if not np.isfinite(avg_cost) or avg_cost <= 0.0:
            raise ValueError('Negative or infinite mean cost not fit for UtilDecreasing')
            avg_cost = 1.0
        cost_scale = avg_cost
        lambda_scale = 1.0
        if self.cfg.util_pricing.vector_prices:
            price_cache = np.zeros((regular_options, d))
        else:
            price_cache = np.zeros(regular_options)

        # We use this solver (from generic) bc. it allows us to reuse functionality for convenience and consistency
        # However it is equivalent to just a sequential assingment minimizing cost + price
        
        solver = OfflineMILPSolver(self.cfg, log_to_console=False)
        cols = regular_options + (1 if fallback_idx >= 0 else 0)

        for step_idx, volume in steps_sorted:
            remaining = np.maximum(0.0, np.asarray(eff_caps, dtype=float) - loads)
            costs = np.asarray(inst.costs.assignment_costs[step_idx, :cols], dtype=float).reshape(1, -1)
            costs = costs / cost_scale

            for option_idx in range(regular_options):
                cap = np.asarray(eff_caps[option_idx], dtype=float)
                denom = np.where(cap > 0.0, cap, 1.0)
                norm_volume = volume / denom
                if self.cfg.util_pricing.vector_prices:
                    lam_vec = price_cache[option_idx]
                    costs[0, option_idx] += float(np.dot(lam_vec, norm_volume))
                else:
                    lam = float(price_cache[option_idx])
                    costs[0, option_idx] += lam * scalarize_vector(norm_volume, size_key)

            feas_matrix = np.asarray(inst.offline_steps[step_idx].feas_matrix, dtype=float)
            feas_rhs = np.asarray(inst.offline_steps[step_idx].feas_rhs, dtype=float)
            cap_matrix = inst.offline_steps[step_idx].cap_matrix
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
                raise ValueError(f"Step {step_idx} cannot be assigned to any feasible bin.")
            chosen_option = step_state.assigned_option.get(0)
            if chosen_option is None:
                raise ValueError(f"Step {step_idx} could not be assigned by MILP.")

            assigned_option[step_idx] = chosen_option
            if chosen_option < regular_options:
                loads[chosen_option] += volume
                price_cache[chosen_option] = self._updated_lambda(
                    loads[chosen_option], eff_caps[chosen_option], lambda_scale
                )

        runtime = time.perf_counter() - start_time
        obj_value = objective_with_fallback(assigned_option, inst)

        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_option=assigned_option,
            offline_evicted_steps=set(),
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

    
