from __future__ import annotations

import time
import numpy as np
from typing import List, Tuple

from generic.config import Config
from generic.models import AssignmentState, Instance, Decision
from generic.online.models import OnlineSolutionInfo
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from generic.online import state_utils


class OnlineSolver:
    """
    Orchestrates the online phase given a policy implementation.
    """

    def __init__(
        self,
        cfg: Config,
        policy: BaseOnlinePolicy,
        *,
        log_to_console: bool = False,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.log_to_console = log_to_console

    def run(
        self,
        instance: Instance,
        initial_state: AssignmentState,
    ) -> Tuple[AssignmentState, OnlineSolutionInfo]:
        """
        Execute the online phase starting from 'initial_state'.
        """
        if not instance.online_items:
            state_copy = state_utils.clone_state(initial_state)
            info = OnlineSolutionInfo(
                status="NO_ITEMS",
                runtime=0.0,
                total_objective=0.0,
                fallback_items=state_utils.count_fallback_items(state_copy, instance),
                evicted_offline=0,
                decisions=[],
            )
            return state_copy, info

        feasible_matrix = None
        if instance.online_feasible is not None:
            feasible_matrix = instance.online_feasible.feasible

        state = state_utils.clone_state(initial_state)
        if (
            instance.fallback_bin_index >= 0
            and state.load.shape[0] == len(instance.bins)
        ):
            # Ensure fallback row exists when a fallback bin is enabled.
            zeros = np.zeros((1, state.load.shape[1]), dtype=state.load.dtype)
            state.load = np.vstack([state.load, zeros])
        volume_lookup = state_utils.build_volume_lookup(instance)
        decisions: List[Decision] = []
        total_objective = 0.0
        eviction_events = 0

        start_time = time.perf_counter()

        for idx, item in enumerate(instance.online_items):
            feasible_row = None
            if feasible_matrix is not None:
                if idx >= feasible_matrix.shape[0]:
                    raise IndexError(
                        f"Feasibility matrix has fewer rows ({feasible_matrix.shape[0]}) "
                        f"than online items ({len(instance.online_items)})."
                    )
                feasible_row = feasible_matrix[idx, :]

            decision = None
            try:
                decision = self.policy.select_bin(item, state, instance, feasible_row)
            except PolicyInfeasibleError:
                decision = None

            needs_fallback = decision is None or (
                not self.cfg.problem.binpacking
                and (decision.evicted_offline or decision.reassignments)
            )
            if needs_fallback:
                decision = self._fallback_decision(instance, item, feasible_row)
                if decision is None:
                    info = OnlineSolutionInfo(
                        status="INFEASIBLE",
                        runtime=time.perf_counter() - start_time,
                        total_objective=total_objective,
                        fallback_items=state_utils.count_fallback_items(state, instance),
                        evicted_offline=len(state.offline_evicted),
                        decisions=decisions,
                    )
                    return state, info
            state_utils.apply_decision(
                decision,
                item,
                state,
                instance,
                volume_lookup,
            )
            decisions.append(decision)
            total_objective += decision.incremental_cost
            eviction_events += len(decision.evicted_offline)

        runtime = time.perf_counter() - start_time

        info = OnlineSolutionInfo(
            status="COMPLETED",
            runtime=runtime,
            total_objective=total_objective,
            fallback_items=state_utils.count_fallback_items(state, instance),
            evicted_offline=eviction_events,
            decisions=decisions,
        )
        return state, info

    def _fallback_decision(
        self,
        instance: Instance,
        item,
        feasible_row,
    ) -> Decision | None:
        if not self._can_fallback_online(instance, feasible_row):
            return None
        fallback_idx = instance.fallback_bin_index
        fallback_cost = self._fallback_cost(instance, item.id, fallback_idx)
        return Decision(
            placed_item=(item.id, fallback_idx),
            evicted_offline=[],
            reassignments=[],
            incremental_cost=fallback_cost,
        )

    def _can_fallback_online(
        self,
        instance: Instance,
        feasible_row,
    ) -> bool:
        if not self.cfg.problem.fallback_is_enabled:
            return False
        if not self.cfg.problem.fallback_allowed_online:
            return False
        fallback_idx = instance.fallback_bin_index
        if fallback_idx < 0:
            return False
        if feasible_row is None:
            return True
        if fallback_idx >= feasible_row.shape[0]:
            return False
        return bool(feasible_row[fallback_idx] == 1)

    def _fallback_cost(self, instance: Instance, item_id: int, fallback_idx: int) -> float:
        costs = instance.costs.assignment_costs
        if (
            costs is not None
            and costs.size
            and fallback_idx < costs.shape[1]
            and item_id < costs.shape[0]
        ):
            return float(costs[item_id, fallback_idx])
        return float(self.cfg.costs.huge_fallback)
