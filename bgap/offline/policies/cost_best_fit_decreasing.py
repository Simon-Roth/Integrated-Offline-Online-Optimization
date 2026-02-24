from __future__ import annotations

import time
from typing import Dict, Tuple, List

from generic.core.utils import (
    option_is_feasible,
    scalarize_vector,
    vector_fits,
    residual_vector,
)
from generic.core.models import AssignmentState, Instance
from generic.offline.policies import BaseOfflinePolicy
from generic.core.models import OfflineSolutionInfo
from bgap.offline.policies._policy_utils import (
    init_loads_and_caps,
    objective_with_fallback,
    sorted_steps_by_volume,
)


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
        steps_sorted = sorted_steps_by_volume(inst, size_key)

        regular_options = inst.n
        fallback_idx = inst.fallback_option_index
        loads, eff_caps = init_loads_and_caps(inst, self.cfg)
        assigned_option: Dict[int, int] = {}

        for step_idx, volume in steps_sorted:
            best_bin = None
            best_cost = float("inf")
            best_residual = float("inf")

            for bin_idx in range(regular_options):
                if not option_is_feasible(
                    inst.offline_steps[step_idx].feas_matrix,
                    inst.offline_steps[step_idx].feas_rhs,
                    bin_idx,
                ):
                    continue
                if not vector_fits(loads[bin_idx], volume, eff_caps[bin_idx], 1e-9):
                    continue
                cost = float(inst.costs.assignment_costs[step_idx, bin_idx])
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
                assigned_option[step_idx] = best_bin
                continue

            if (
                fallback_idx >= 0
                and option_is_feasible(
                    inst.offline_steps[step_idx].feas_matrix,
                    inst.offline_steps[step_idx].feas_rhs,
                    fallback_idx,
                )
            ):
                assigned_option[step_idx] = fallback_idx
                continue

            raise ValueError(f"Step {step_idx} cannot be assigned to any feasible bin.")

        runtime = time.perf_counter() - start_time
        obj_value = objective_with_fallback(assigned_option, inst)

        state = AssignmentState(
            load=loads.reshape(-1),
            assigned_option=assigned_option,
            offline_evicted_steps=set(),
        )
        info = OfflineSolutionInfo(
            algorithm="CostAwareBestFitDecreasing",
            status="FEASIBLE",
            runtime=runtime,
            obj_value=obj_value,
            feasible=True,
        )
        return state, info
