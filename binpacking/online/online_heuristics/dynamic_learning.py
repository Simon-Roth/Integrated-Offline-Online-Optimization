from __future__ import annotations

"""
Dynamic Learning Algorithm (binpacking extension)

This policy inherits the generic Dynamic Learning core and adds binpacking-
specific eviction/reshuffling logic when allow_reassignment is enabled.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from generic.config import Config
from generic.general_utils import (
    action_is_feasible,
    scalarize_vector,
    vector_fits,
    residual_vector,
)
from generic.models import AssignmentState, Decision, Instance, OnlineItem
from generic.online.policies import GenericDynamicLearningPolicy
from generic.online.policies import PolicyInfeasibleError
from binpacking.online.state_utils import (
    PlacementContext,
    build_context,
    execute_placement,
    TOLERANCE,
)


class DynamicLearningPolicy(GenericDynamicLearningPolicy):
    """
    Dynamic price-updating policy with binpacking-specific evictions.
    """

    def __init__(self, cfg: Config, price_path: Optional[Path] = None) -> None:
        super().__init__(cfg, price_path)

    def select_action(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        # Lazily build phase schedule on the first call.
        if not self._schedule:
            self._schedule = self._build_schedule(len(instance.online_items))

        # Update prices if a phase boundary was reached after the last item.
        self._maybe_update_prices(state, instance)

        candidate_actions = self._candidate_actions(item, instance)
        if not candidate_actions:
            raise PolicyInfeasibleError(f"No feasible regular action for online item {item.id}")

        # First, try to place without evictions via a one-item MILP (regular actions only).
        milp_decision = self._milp_no_eviction(item, state, instance)
        if milp_decision is not None:
            self._mark_processed(item)
            return milp_decision

        if not self.cfg.problem.allow_reassignment:
            raise PolicyInfeasibleError(
                f"DynamicLearningPolicy could not place item {item.id}"
            )

        # Allow evictions if no feasible bin remained.
        best_decision: Optional[Decision] = None
        best_score = float("inf")
        for action_id in candidate_actions:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            score = self._score(action_id, item, instance)
            if score >= best_score - 1e-12:
                continue
            decision = execute_placement(
                action_id,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
                allow_eviction=True,
            )
            if decision is not None:
                best_score = score
                best_decision = decision

        if best_decision is not None:
            self._mark_processed(item)
            return best_decision

        # No feasible regular action (even with evictions) -> signal infeasibility.
        raise PolicyInfeasibleError(f"DynamicLearningPolicy could not place item {item.id}")

    def _eviction_order_desc(
        self,
        bin_id: int,
        ctx: PlacementContext,
    ) -> List[int]:
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
