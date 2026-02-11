from __future__ import annotations

import time
from typing import List, Tuple

from generic.core.config import Config
from generic.core.models import AssignmentState, Instance, Decision
from generic.core.models import OnlineSolutionInfo
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from generic.online import state_utils
from generic.core.utils import option_is_feasible


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
        if not instance.online_steps:
            # Nothing to process: return a clean clone and a zero-cost summary
            state_copy = state_utils.clone_state(initial_state)
            info = OnlineSolutionInfo(
                status="NO_STEPS",
                runtime=0.0,
                total_objective=0.0,
                fallback_steps=state_utils.count_fallback_steps(state_copy, instance),
                evicted_offline_steps=0,
                decisions=[],
            )
            return state_copy, info

        # Work on a cloned state so the offline allocation remains intact.
        state = state_utils.clone_state(initial_state)
        # Cache A_t^{cap} lookups for all steps to speed up state updates.
        cap_lookup = state_utils.build_cap_lookup(instance)
        decisions: List[Decision] = []
        total_objective = 0.0
        eviction_events = 0

        start_time = time.perf_counter()

        for step in instance.online_steps:
            decision = None
            try:
                # Let the policy pick an option based on the current state.
                decision = self.policy.select_action(step, state, instance)
            except PolicyInfeasibleError:
                decision = None

            # Regarding the below check: 
            # Generic policies do not need that check as feasibility is encoded in A_t^{feas} x_t = b_t.
            # However, bp. specific policies try 1) regular bins 2) evictions if needed and allowed, and 3) fallback as last resort.
            # So below: If bp. policy failed or produced disallowed evictions/reassigned_offline_steps, try fallback.
            # I decided to unify this behaviour here to make the bp. policies cleaner -> less duplicate code.
            needs_fallback = decision is None or (
                not self.cfg.problem.allow_reassignment
                and (decision.evicted_offline_steps or decision.reassigned_offline_steps)
            )
            if needs_fallback:
                decision = self._fallback_decision(instance, step)
                if decision is None:
                    info = OnlineSolutionInfo(
                        status="INFEASIBLE",
                        runtime=time.perf_counter() - start_time,
                        total_objective=total_objective,
                        fallback_steps=state_utils.count_fallback_steps(state, instance),
                        evicted_offline_steps=len(state.offline_evicted_steps),
                        decisions=decisions,
                    )
                    return state, info
            # Apply the decision to the live state (load update + assignments).
            state_utils.apply_decision(
                decision,
                step,
                state,
                instance,
                cap_lookup,
            )
            decisions.append(decision)
            total_objective += decision.incremental_cost
            eviction_events += len(decision.evicted_offline_steps)

        runtime = time.perf_counter() - start_time

        info = OnlineSolutionInfo(
            status="COMPLETED",
            runtime=runtime,
            total_objective=total_objective,
            fallback_steps=state_utils.count_fallback_steps(state, instance),
            evicted_offline_steps=eviction_events,
            decisions=decisions,
        )
        return state, info

    def _fallback_decision(
        self,
        instance: Instance,
        step,
    ) -> Decision | None:
        # Construct a fallback decision if enabled and feasible for this step.
        if not self._can_fallback_online(instance, step):
            return None
        fallback_idx = instance.fallback_option_index
        fallback_cost = self._fallback_cost(instance, step.step_id, fallback_idx)
        return Decision(
            placed_step=(step.step_id, fallback_idx),
            evicted_offline_steps=[],
            reassigned_offline_steps=[],
            incremental_cost=fallback_cost,
        )

    def _can_fallback_online(
        self,
        instance: Instance,
        step,
    ) -> bool:
        # Fallback is controlled by config flags and the local feasibility constraints.
        if not self.cfg.problem.fallback_is_enabled:
            return False
        if not self.cfg.problem.fallback_allowed_online:
            return False
        fallback_idx = instance.fallback_option_index
        if fallback_idx < 0:
            return False
        return option_is_feasible(step.feas_matrix, step.feas_rhs, fallback_idx)

    def _fallback_cost(self, instance: Instance, step_id: int, fallback_idx: int) -> float:
        # Prefer explicit fallback cost if present in the cost matrix.
        costs = instance.costs.assignment_costs
        if (
            costs is not None
            and costs.size
            and fallback_idx < costs.shape[1]
            and step_id < costs.shape[0]
        ):
            return float(costs[step_id, fallback_idx])
        return float(self.cfg.costs.huge_fallback)
