from __future__ import annotations

import time
import math
from typing import Dict, Tuple, List

import numpy as np

from generic.general_utils import effective_capacity, scalarize_vector, vector_fits, residual_vector
from generic.models import AssignmentState, Instance
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
        fallback_idx = regular_bins
        dims = inst.bins[0].capacity.shape[0] if inst.bins else 1
        loads = np.zeros((regular_bins + 1, dims))
        assigned_bin: Dict[int, int] = {}

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
            avg_cost = 1.0
        lambda_scale = avg_cost
        if self.cfg.util_pricing.vector_prices:
            price_cache = np.zeros((regular_bins, dims))
        else:
            price_cache = np.zeros(regular_bins)

        for item_idx, volume in items_sorted:
            best_bin = None
            best_score = float("inf")
            best_residual = float("-inf")

            for bin_idx in range(regular_bins):
                if inst.offline_feasible.feasible[item_idx, bin_idx] != 1:
                    continue
                cap = eff_caps[bin_idx]
                if np.any(cap <= 0.0):
                    continue
                if not vector_fits(loads[bin_idx], volume, cap, 1e-9):
                    continue

                if self.cfg.util_pricing.vector_prices:
                    lam_vec = price_cache[bin_idx]
                    score = float(inst.costs.assignment_costs[item_idx, bin_idx]) + float(np.dot(lam_vec, volume))
                else:
                    lam = float(price_cache[bin_idx])
                    score = float(inst.costs.assignment_costs[item_idx, bin_idx]) + lam * scalarize_vector(volume, size_key)
                residual = residual_vector(loads[bin_idx], volume, cap)
                residual_score = scalarize_vector(residual, self.cfg.heuristics.residual_scalarization)
                if (
                    score < best_score - 1e-9
                    or (abs(score - best_score) <= 1e-9 and residual_score > best_residual)
                ):
                    best_score = score
                    best_residual = residual_score
                    best_bin = bin_idx

            if best_bin is not None:
                loads[best_bin] += volume
                assigned_bin[item_idx] = best_bin
                price_cache[best_bin] = self._updated_lambda(
                    loads[best_bin], eff_caps[best_bin], lambda_scale
                )
                continue

            if (
                self.cfg.problem.fallback_is_enabled and self.cfg.problem.fallback_allowed_offline
                and inst.offline_feasible.feasible[item_idx, fallback_idx] == 1
            ):
                loads[fallback_idx] += volume
                assigned_bin[item_idx] = fallback_idx
                continue

            raise ValueError(f"Item {item_idx} cannot be assigned to any feasible bin.")

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
