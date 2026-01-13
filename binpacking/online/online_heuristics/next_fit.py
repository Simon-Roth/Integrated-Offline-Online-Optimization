from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy

from generic.config import Config
from generic.models import AssignmentState, Decision, Instance, OnlineItem
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from generic.online.state_utils import (
    PlacementContext,
    build_context,
    execute_placement,
    TOLERANCE,
)
from generic.general_utils import vector_fits


class NextFitOnlinePolicy(BaseOnlinePolicy):
    """
    Online Next-Fit heuristic. Attempts to keep placing items into the same bin
    as long as it remains feasible; otherwise cycles to the next feasible bin.
    Falls back to the fallback bin if no feasible regular bin has sufficient space.
    """

    def __init__(self, cfg: Config, *, k: int = 1) -> None:
        self.cfg = cfg
        self.k = max(1, k)
        self._last_bin: Optional[int] = None

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional["numpy.ndarray"],
    ) -> Decision:
        ctx = build_context(self.cfg, instance, state)
        feasible_bins = self._feasible_bins(item, instance, feasible_row)
        if not feasible_bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online item {item.id}")

        ordered_bins = self._ordered_bins(feasible_bins)
        window = ordered_bins[: min(self.k, len(ordered_bins))]

        # Try to place without eviction inside the active window
        for bin_id in window:
            if vector_fits(ctx.loads[bin_id], item.volume, ctx.effective_caps[bin_id], TOLERANCE):
                decision = execute_placement(
                    bin_id,
                    item,
                    ctx,
                    eviction_order_fn=self._eviction_order_fifo,
                    destination_fn=self._choose_destination,
                    allow_eviction=False,
                )
                if decision is None:
                    continue
                self._last_bin = bin_id
                return decision

        # No bin in the window can accommodate directly; open the next feasible bin (if any).
        candidate_after_window = [bin_id for bin_id in ordered_bins if bin_id not in window]
        eviction_targets = candidate_after_window + window

        for target_bin in eviction_targets:
            decision = execute_placement(
                target_bin,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_fifo,
                destination_fn=self._choose_destination,
                allow_eviction=True,
            )
            if decision is not None:
                self._last_bin = target_bin
                return decision

        raise PolicyInfeasibleError(
            f"NextFitOnlinePolicy could not place item {item.id}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _feasible_bins(
        self,
        item: OnlineItem,
        instance: Instance,
        feasible_row: Optional["numpy.ndarray"],
    ) -> List[int]:
        bins: Set[int] = set(item.feasible_bins)
        if feasible_row is not None:
            regular_bins = len(instance.bins)
            for idx, allowed in enumerate(feasible_row[:regular_bins]):
                if allowed:
                    bins.add(int(idx))
        return sorted(b for b in bins if 0 <= b < len(instance.bins))

    def _ordered_bins(self, bins: List[int]) -> List[int]:
        if not bins:
            return []
        if self._last_bin is None or self._last_bin not in bins:
            return bins
        idx = bins.index(self._last_bin)
        return bins[idx + 1 :] + bins[: idx + 1]

    def _eviction_order_fifo(
        self,
        bin_id: int,
        ctx: PlacementContext,
    ) -> List[int]:
        offline_ids = [
            itm_id
            for itm_id, assigned_bin in ctx.assignments.items()
            if assigned_bin == bin_id and itm_id < len(ctx.instance.offline_items)
        ]
        return offline_ids  # FIFO order by default


    def _choose_destination(
        self,
        offline_id: int,
        origin_bin: int,
        ctx: PlacementContext,
    ) -> Optional[int]:
        volume = ctx.offline_volumes.get(offline_id, 0.0)
        instance = ctx.instance
        feasible_row = instance.feasible.feasible[offline_id]
        regular_bins = len(instance.bins)
        fallback_idx = instance.fallback_bin_index

        feasible_bins = self._feasible_bins_for_offline(offline_id, instance)
        ordered_bins = self._ordered_bins(feasible_bins)
        if not ordered_bins:
            ordered_bins = feasible_bins

        for candidate in ordered_bins:
            if candidate == origin_bin or feasible_row[candidate] != 1:
                continue
            residual = ctx.effective_caps[candidate] - (ctx.loads[candidate] + volume)
            if residual + TOLERANCE >= 0:
                return candidate

        if (
            self.cfg.problem.fallback_is_enabled
            and fallback_idx < feasible_row.shape[0]
            and feasible_row[fallback_idx] == 1
        ):
            return fallback_idx

        return None

    def _feasible_bins_for_offline(self, offline_id: int, instance: Instance) -> List[int]:
        row = instance.feasible.feasible[offline_id]
        regular_bins = len(instance.bins)
        return [idx for idx in range(regular_bins) if row[idx] == 1]
