from __future__ import annotations

import time
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
                total_cost=0.0,
                fallback_items=state_utils.count_fallback_items(state_copy, instance),
                evicted_offline=0,
                decisions=[],
            )
            return state_copy, info

        feasible_matrix = None
        if instance.online_feasible is not None:
            feasible_matrix = instance.online_feasible.feasible

        state = state_utils.clone_state(initial_state)
        volume_lookup = state_utils.build_volume_lookup(instance)
        decisions: List[Decision] = []
        total_cost = 0.0
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

            try:
                decision = self.policy.select_bin(item, state, instance, feasible_row)
            except PolicyInfeasibleError:
                info = OnlineSolutionInfo(
                    status="INFEASIBLE",
                    runtime=time.perf_counter() - start_time,
                    total_cost=total_cost,
                    fallback_items=state_utils.count_fallback_items(state, instance),
                    evicted_offline=len(state.offline_evicted),
                    decisions=decisions,
                )
                return state, info
            if not self.cfg.problem.binpacking:
                if decision.evicted_offline or decision.reassignments:
                    raise PolicyInfeasibleError(
                        "Evictions/reassignments are disabled when binpacking is False."
                    )
            state_utils.apply_decision(
                decision,
                item,
                state,
                instance,
                volume_lookup,
            )
            decisions.append(decision)
            total_cost += decision.incremental_cost
            eviction_events += len(decision.evicted_offline)

        runtime = time.perf_counter() - start_time

        info = OnlineSolutionInfo(
            status="COMPLETED",
            runtime=runtime,
            total_cost=total_cost,
            fallback_items=state_utils.count_fallback_items(state, instance),
            evicted_offline=eviction_events,
            decisions=decisions,
        )
        return state, info
