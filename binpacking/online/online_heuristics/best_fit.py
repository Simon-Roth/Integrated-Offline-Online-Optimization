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
from binpacking.block_utils import extract_volume


class BestFitOnlinePolicy(BaseOnlinePolicy):
    """
    Greedy online heuristic: choose the feasible bin that leaves the smallest
    remaining capacity after placing the item. Falls back to the dedicated
    fallback bin only if no regular bin can accommodate the item.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def select_action(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        ctx = build_context(self.cfg, instance, state)
        candidate_bins = self._candidate_bins(item, instance)
        if not candidate_bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online item {item.id}")

        allow_reshuffle = self.cfg.problem.allow_reassignment
        residual_mode = self.cfg.heuristics.residual_scalarization
        volume = extract_volume(item.cap_matrix, instance.n, instance.m)
        fits: List[tuple[float, int]] = []
        overflows: List[tuple[float, int]] = []
        for bin_id in candidate_bins:
            residual_vec = residual_vector(ctx.loads[bin_id], volume, ctx.effective_caps[bin_id])
            residual_score = scalarize_vector(residual_vec, residual_mode)
            if vector_fits(ctx.loads[bin_id], volume, ctx.effective_caps[bin_id], TOLERANCE):
                fits.append((residual_score, bin_id))
            else:
                overflows.append((residual_score, bin_id))

        # Try bins that currently fit without evictions, ordered by tightest fit
        for residual, target_bin in sorted(fits, key=lambda pair: (pair[0], pair[1])):
            decision = execute_placement(
                target_bin,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
                allow_eviction=False,
            )
            if decision is not None:
                return decision

        # Otherwise, attempt bins in order of smallest overflow, allowing evictions
        if allow_reshuffle:
            for residual, target_bin in sorted(overflows, key=lambda pair: (pair[0], pair[1]), reverse=True):
                decision = execute_placement(
                    target_bin,
                    item,
                    ctx,
                    eviction_order_fn=self._eviction_order_desc,
                    destination_fn=self._select_reassignment_bin,
                    allow_eviction=True,
                )
                if decision is not None:
                    return decision

        raise PolicyInfeasibleError(f"BestFitOnlinePolicy could not place item {item.id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_bins(
        self,
        item: OnlineItem,
        instance: Instance,
    ) -> List[int]:
        regular_bins = list(range(instance.n))
        return feasible_action_indices(item.feas_matrix, item.feas_rhs, action_ids=regular_bins)

    def _eviction_order_desc(
        self,
        bin_id: int,
        ctx: PlacementContext,
    ) -> List[int]:
        regular_bins = ctx.instance.n
        fallback_idx = ctx.instance.fallback_action_index
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
        best_residual = float("inf")

        for candidate in range(regular_bins):
            if candidate == origin_bin or not action_is_feasible(
                offline_item.feas_matrix, offline_item.feas_rhs, candidate
            ):
                continue
            residual_vec = residual_vector(ctx.loads[candidate], volume, ctx.effective_caps[candidate])
            residual_score = scalarize_vector(residual_vec, self.cfg.heuristics.residual_scalarization)
            if vector_fits(ctx.loads[candidate], volume, ctx.effective_caps[candidate], TOLERANCE) and residual_score < best_residual:
                best_residual = residual_score
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

        if action_is_feasible(offline_item.feas_matrix, offline_item.feas_rhs, fallback_idx):
            return fallback_idx

        return None
