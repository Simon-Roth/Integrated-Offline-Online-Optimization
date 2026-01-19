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
        items_sorted: List[Tuple[int, np.ndarray]] = sorted(
            ((idx, item.volume) for idx, item in enumerate(inst.offline_items)),
            key=lambda pair: scalarize_vector(pair[1], size_key),
            reverse=True,
        )

        regular_bins = len(inst.bins)
        fallback_idx = inst.fallback_bin_index
        bins_total = regular_bins + (1 if fallback_idx >= 0 else 0)
        dims = inst.bins[0].capacity.shape[0] if inst.bins else 1
        loads = np.zeros((bins_total, dims))
        assigned_bin: Dict[int, int] = {}
        if len(inst.offline_items) == 0:
            runtime = time.perf_counter() - start_time
            state = AssignmentState(
                load=loads,
                assigned_bin=assigned_bin,
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

        eff_caps = [
            effective_capacity(
                bin_spec.capacity,
                self.cfg.slack.enforce_slack,
                self.cfg.slack.fraction,
            )
            for bin_spec in inst.bins
        ]
        avg_cost = float(np.mean(inst.costs.assignment_costs[: len(inst.offline_items), :regular_bins]))
        if not np.isfinite(avg_cost) or avg_cost <= 0.0:
            raise ValueError('Negative or infinite mean cost not fit for UtilDecreasing')
            avg_cost = 1.0
        cost_scale = avg_cost
        lambda_scale = 1.0
        if self.cfg.util_pricing.vector_prices:
            price_cache = np.zeros((regular_bins, dims))
        else:
            price_cache = np.zeros(regular_bins)

        solver = OfflineMILPSolver(self.cfg, log_to_console=False)
        cols = regular_bins + (1 if fallback_idx >= 0 else 0)

        for item_idx, volume in items_sorted:
            remaining = np.maximum(0.0, np.asarray(eff_caps, dtype=float) - loads[:regular_bins])
            costs = np.asarray(inst.costs.assignment_costs[item_idx, :cols], dtype=float).reshape(1, -1)
            costs = costs / cost_scale

            for bin_idx in range(regular_bins):
                cap = np.asarray(eff_caps[bin_idx], dtype=float)
                denom = np.where(cap > 0.0, cap, 1.0)
                norm_volume = volume / denom
                if self.cfg.util_pricing.vector_prices:
                    lam_vec = price_cache[bin_idx]
                    costs[0, bin_idx] += float(np.dot(lam_vec, norm_volume))
                else:
                    lam = float(price_cache[bin_idx])
                    costs[0, bin_idx] += lam * scalarize_vector(norm_volume, size_key)

            feasible_row = np.asarray(
                inst.offline_feasible.feasible[item_idx, :cols],
                dtype=int,
            ).reshape(1, -1)
            data = build_offline_milp_data_from_arrays(
                volumes=np.asarray(volume, dtype=float).reshape(1, -1),
                costs=costs,
                feasible=feasible_row,
                capacities=remaining,
                fallback_idx=fallback_idx,
                fallback_capacity=self.cfg.problem.fallback_capacity_offline,
                slack_enforce=False,
                slack_fraction=0.0,
            )
            step_state, step_info = solver.solve_from_data(data)
            if not step_info.feasible:
                raise ValueError(f"Item {item_idx} cannot be assigned to any feasible bin.")
            chosen_bin = step_state.assigned_bin.get(0)
            if chosen_bin is None:
                raise ValueError(f"Item {item_idx} could not be assigned by MILP.")

            assigned_bin[item_idx] = chosen_bin
            if chosen_bin < regular_bins:
                loads[chosen_bin] += volume
                price_cache[chosen_bin] = self._updated_lambda(
                    loads[chosen_bin], eff_caps[chosen_bin], lambda_scale
                )
            elif fallback_idx >= 0 and chosen_bin == fallback_idx:
                loads[fallback_idx] += volume

        runtime = time.perf_counter() - start_time
        obj_value = self._calculate_objective(assigned_bin, inst)

        state = AssignmentState(
            load=loads,
            assigned_bin=assigned_bin,
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
        self, assigned_bin: Dict[int, int], inst: Instance
    ) -> float:
        total_cost = 0.0
        fallback_idx = len(inst.bins)
        for item_idx, bin_idx in assigned_bin.items():
            if bin_idx < fallback_idx:
                total_cost += float(inst.costs.assignment_costs[item_idx, bin_idx])
            else:
                total_cost += inst.costs.huge_fallback
        return total_cost
