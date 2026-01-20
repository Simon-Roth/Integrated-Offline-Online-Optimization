from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, List, Dict, Set

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
from generic.general_utils import scalarize_vector, residual_vector, vector_fits
from binpacking.block_utils import extract_volume

class SimBasePolicy(BaseOnlinePolicy):
    """
    Cost-minimizing Lagrangian policy:
    - score(i) = c_{ji} + λ_i * volume_j
    - choose feasible regular bin with minimum score, allowing evictions if needed
    - if none works, raise PolicyInfeasibleError so the caller can handle fallback.
    """

    def __init__(self, cfg: Config, price_path: Path = Path("binpacking/results/sim_base.json")):
        self.cfg = cfg
        with open(price_path) as f:
            data = json.load(f)
        # per-regular-bin price λ_i (scalar or vector)
        self.lam: Dict[int, np.ndarray] = {}
        for k, v in data["prices"].items():
            if isinstance(v, list):
                self.lam[int(k)] = np.asarray(v, dtype=float)
            else:
                self.lam[int(k)] = np.asarray([float(v)], dtype=float)

    def select_action(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        fallback_idx = instance.fallback_action_index
        candidate_bins = self._candidate_bins(item, instance, feasible_row)
        if not candidate_bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online item {item.id}")

        allow_reshuffle = self.cfg.problem.allow_reassignment
        best_decision: Optional[Decision] = None
        best_score = float("inf")

        # First, try to place without evictions.
        for bin_id in candidate_bins:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            volume = extract_volume(item.cap_matrix, instance.n, instance.m)
            if not vector_fits(ctx.loads[bin_id], volume, ctx.effective_caps[bin_id], TOLERANCE):
                continue

            score = self._score(bin_id, item, instance)
            if score >= best_score - 1e-12:
                continue

            decision = execute_placement(
                bin_id,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
                allow_eviction=False,
            )
            if decision is not None:
                best_score = score
                best_decision = decision

        if best_decision is not None:
            return best_decision

        if not allow_reshuffle:
            raise PolicyInfeasibleError(f"SimBasePolicy could not place item {item.id}")

        # Allow evictions if no feasible bin remained.
        for bin_id in candidate_bins:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            score = self._score(bin_id, item, instance)
            if score >= best_score - 1e-12:
                continue
            decision = execute_placement(
                bin_id,
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
            return best_decision

        # No feasible regular bin (even with evictions) -> signal infeasibility so
        # the caller can handle fallback placement consistently with other policies.
        raise PolicyInfeasibleError(f"SimBasePolicy could not place item {item.id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_bins(
        self,
        item: OnlineItem,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> List[int]:
        regular_bins = instance.n
        candidate_bins: Set[int] = set(item.feasible_actions)
        if feasible_row is not None:
            for idx, allowed in enumerate(feasible_row[:regular_bins]):
                if allowed:
                    candidate_bins.add(int(idx))
        return sorted(b for b in candidate_bins if 0 <= b < regular_bins)

    def _score(self, bin_id: int, item: OnlineItem, instance: Instance) -> float:
        c_ji = float(instance.costs.assignment_costs[item.id, bin_id])
        volume = extract_volume(item.cap_matrix, instance.n, instance.m)
        lam_i = self.lam.get(bin_id, np.zeros_like(volume))
        if self.cfg.util_pricing.vector_prices:
            return c_ji + float(np.dot(lam_i, volume))
        lam_scalar = scalarize_vector(lam_i, "max")
        return c_ji + lam_scalar * scalarize_vector(volume, self.cfg.heuristics.size_key)

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
        feasible_row = instance.offline_feasible.feasible[offline_id]
        regular_bins = instance.n
        fallback_idx = instance.fallback_action_index

        best_candidate: Optional[int] = None
        best_cost = float("inf")
        best_residual = float("inf")

        for candidate in range(regular_bins):
            if candidate == origin_bin or feasible_row[candidate] != 1:
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

        if (
            self.cfg.problem.fallback_is_enabled and self.cfg.problem.fallback_allowed_offline
            and fallback_idx < feasible_row.shape[0]
            and feasible_row[fallback_idx] == 1
        ):
            return fallback_idx
        return None
