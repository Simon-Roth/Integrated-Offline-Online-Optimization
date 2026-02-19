from __future__ import annotations

from typing import List, Optional

import numpy as np

from generic.core.config import Config
from generic.core.models import AssignmentState, Decision, Instance, StepSpec
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from binpacking.online.state_utils import (
    build_context,
    candidate_bins,
    eviction_order_desc,
    execute_placement,
    select_reassignment_bin,
    TOLERANCE,
)
from generic.core.utils import (
    scalarize_vector,
)


class CostAwareBestFitOnlinePolicy(BaseOnlinePolicy):
    """
    Selects the feasible bin that minimises incremental cost (assignment plus potential eviction
    penalties). Uses residual capacity as a tie-breaker to avoid fragmentation.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def begin_instance(
        self,
        instance: Instance,
        initial_state: AssignmentState,
    ) -> None:
        # Stateless heuristic: nothing to initialize per instance.
        return

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        bins = candidate_bins(step, instance)
        if not bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online step {step.step_id}")

        allow_reshuffle = self.cfg.problem.allow_reassignment
        size_key = self.cfg.heuristics.size_key
        residual_mode = self.cfg.heuristics.residual_scalarization
        eviction_fn = lambda bin_id, ctx: eviction_order_desc(bin_id, ctx, size_key=size_key)
        destination_fn = lambda offline_id, origin_bin, ctx: select_reassignment_bin(
            offline_id,
            origin_bin,
            ctx,
            mode="cost",
            residual_mode=residual_mode,
        )
        best_decision: Optional[Decision] = None
        best_cost = float("inf")
        best_residual = float("inf")

        # First, try placements without evictions.
        # We dont just immediatly pick the bin in candidate_bins with min c(j,i) because of the "best-fit" logic as tie break (-> complete loop needed) 
        for bin_id in bins:
            ctx = build_context(self.cfg, instance, state)
            decision = execute_placement(
                bin_id,
                step,
                ctx,
                eviction_order_fn=eviction_fn,
                destination_fn=destination_fn,
                allow_eviction=False,
            )
            if decision is None:
                continue

            incremental_cost = decision.incremental_cost
            residual_vec = ctx.effective_caps[bin_id] - ctx.loads[bin_id]
            residual_score = scalarize_vector(residual_vec, self.cfg.heuristics.residual_scalarization)
            if not np.all(residual_vec >= -TOLERANCE):
                residual_score = float("inf")

            if (
                incremental_cost < best_cost - 1e-9
                or (
                    abs(incremental_cost - best_cost) <= 1e-9
                    and residual_score < best_residual - 1e-9
                )
            ):
                best_cost = incremental_cost
                best_residual = residual_score
                best_decision = decision

        if best_decision is not None:
            return best_decision

        if not allow_reshuffle:
            raise PolicyInfeasibleError(
                f"CostAwareBestFitOnlinePolicy could not place step {step.step_id}"
            )

        # Allow evictions if no feasible bin remained.
        for bin_id in bins:
            ctx = build_context(self.cfg, instance, state)
            decision = execute_placement(
                bin_id,
                step,
                ctx,
                eviction_order_fn=eviction_fn,
                destination_fn=destination_fn,
                allow_eviction=True,
            )
            if decision is None:
                continue

            incremental_cost = decision.incremental_cost
            residual_vec = ctx.effective_caps[bin_id] - ctx.loads[bin_id]
            residual_score = scalarize_vector(residual_vec, self.cfg.heuristics.residual_scalarization)
            if not np.all(residual_vec >= -TOLERANCE):
                residual_score = float("inf")

            if (
                incremental_cost < best_cost - 1e-9
                or (
                    abs(incremental_cost - best_cost) <= 1e-9
                    and residual_score < best_residual - 1e-9
                )
            ):
                best_cost = incremental_cost
                best_residual = residual_score
                best_decision = decision

        if best_decision is None:
            raise PolicyInfeasibleError(
                f"CostAwareBestFitOnlinePolicy could not place step {step.step_id}"
            )
        return best_decision

 
