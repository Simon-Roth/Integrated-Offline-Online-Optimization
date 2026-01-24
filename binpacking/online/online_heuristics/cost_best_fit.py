from __future__ import annotations

from typing import List, Optional

import numpy as np

from generic.config import Config
from generic.models import AssignmentState, Decision, Instance, OnlineItem
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from binpacking.online.state_utils import (
    PlacementContext,
    build_context,
    execute_placement,
    TOLERANCE,
)
from generic.general_utils import (
    action_is_feasible,
    feasible_action_indices,
    scalarize_vector,
    residual_vector,
    vector_fits,
)


class CostAwareBestFitOnlinePolicy(BaseOnlinePolicy):
    """
    Selects the feasible bin that minimises incremental cost (assignment plus potential eviction
    penalties). Uses residual capacity as a tie-breaker to avoid fragmentation.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def select_action(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        candidate_bins = self._candidate_bins(item, instance)
        if not candidate_bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online item {item.id}")

        allow_reshuffle = self.cfg.problem.allow_reassignment
        best_decision: Optional[Decision] = None
        best_cost = float("inf")
        best_residual = float("inf")

        # First, try placements without evictions.
        # We dont just immediatly pick the bin in candidate_bins with min c(j,i) because of the "best-fit" logic as tie break (That is why we go through all for each) 
        for bin_id in candidate_bins:
            ctx = build_context(self.cfg, instance, state)
            decision = execute_placement(
                bin_id,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
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
                f"CostAwareBestFitOnlinePolicy could not place item {item.id}"
            )

        # Allow evictions if no feasible bin remained.
        for bin_id in candidate_bins:
            ctx = build_context(self.cfg, instance, state)
            decision = execute_placement(
                bin_id,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
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
                f"CostAwareBestFitOnlinePolicy could not place item {item.id}"
            )
        return best_decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_bins(
        self,
        item: OnlineItem,
        instance: Instance,
    ) -> List[int]:
        regular_actions = list(range(instance.n))
        return feasible_action_indices(item.feas_matrix, item.feas_rhs, action_ids=regular_actions)

    def _eviction_order_desc(
        self,
        bin_id: int,
        ctx: PlacementContext,
    ) -> List[int]:
        regular_bins = ctx.instance.n
        offline_ids = [
            itm_id
            for itm_id, assigned_bin in ctx.assignments.items()
            if assigned_bin == bin_id and itm_id < len(ctx.instance.offline_items)
        ]
        size_key = self.cfg.heuristics.size_key
        zero_vec = np.zeros_like(ctx.effective_caps[0])
        offline_ids.sort(
            key=lambda oid: scalarize_vector(ctx.offline_volumes.get(oid, zero_vec), size_key),
            reverse=True,
        )
        return offline_ids

    def _select_reassignment_bin(
        self,
        offline_id: int,
        origin_bin: int,
        ctx: PlacementContext,
    ) -> Optional[int]:
        zero_vec = np.zeros_like(ctx.effective_caps[0])
        volume = ctx.offline_volumes.get(offline_id, zero_vec)
        instance = ctx.instance
        regular_bins = instance.n
        fallback_idx = instance.fallback_action_index
        offline_item = instance.offline_items[offline_id]

        best_candidate: Optional[int] = None
        best_cost = float("inf")
        best_residual = float("inf")

        for candidate in range(regular_bins):
            if candidate == origin_bin or not action_is_feasible(
                offline_item.feas_matrix, offline_item.feas_rhs, candidate
            ):
                continue
            residual_vec = residual_vector(ctx.loads[candidate], volume, ctx.effective_caps[candidate])
            if not vector_fits(ctx.loads[candidate], volume, ctx.effective_caps[candidate], TOLERANCE):
                continue
            cost = instance.costs.assignment_costs[offline_id, candidate]
            residual_score = scalarize_vector(residual_vec, self.cfg.heuristics.residual_scalarization)
            if cost < best_cost - 1e-9 or (
                abs(cost - best_cost) <= 1e-9 and residual_score < best_residual
            ):
                best_cost = cost
                best_residual = residual_score
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

        if action_is_feasible(offline_item.feas_matrix, offline_item.feas_rhs, fallback_idx):
            return fallback_idx
        return None
