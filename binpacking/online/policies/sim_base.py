from __future__ import annotations
from pathlib import Path
import json
from typing import Optional, List, Dict

import numpy as np

from generic.core.config import Config
from generic.core.models import AssignmentState, Decision, Instance, StepSpec
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from binpacking.online.state_utils import (
    PlacementContext,
    build_context,
    candidate_bins,
    eviction_order_desc,
    execute_placement,
    TOLERANCE,
    select_reassignment_bin,
)
from generic.core.utils import (
    scalarize_vector,
    vector_fits,
)
from binpacking.core.block_utils import extract_volume

class SimBasePolicy(BaseOnlinePolicy):
    """
    Cost-minimizing Lagrangian policy:
    - score(i) = c_{ji} + λ_i * volume_j
    - choose feasible regular bin with minimum score, allowing evictions if needed
    - if none works, raise PolicyInfeasibleError so the caller can handle fallback.
    """

    def __init__(self, cfg: Config, price_path: Path = Path("outputs/binpacking/results/sim_base.json")):
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
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        fallback_idx = instance.fallback_option_index
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
        best_score = float("inf")

        # First, try to place without evictions.
        for bin_id in bins:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            volume = extract_volume(step.cap_matrix, instance.n, instance.m)
            if not vector_fits(ctx.loads[bin_id], volume, ctx.effective_caps[bin_id], TOLERANCE):
                continue

            score = self._score(bin_id, step, instance)
            if score >= best_score - 1e-12:
                continue

            decision = execute_placement(
                bin_id,
                step,
                ctx,
                eviction_order_fn=eviction_fn,
                destination_fn=destination_fn,
                allow_eviction=False,
            )
            if decision is not None:
                best_score = score
                best_decision = decision

        if best_decision is not None:
            return best_decision

        if not allow_reshuffle:
            raise PolicyInfeasibleError(f"SimBasePolicy could not place step {step.step_id}")

        # Allow evictions if no feasible bin remained.
        for bin_id in bins:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            score = self._score(bin_id, step, instance)
            if score >= best_score - 1e-12:
                continue
            decision = execute_placement(
                bin_id,
                step,
                ctx,
                eviction_order_fn=eviction_fn,
                destination_fn=destination_fn,
                allow_eviction=True,
            )
            if decision is not None:
                best_score = score
                best_decision = decision

        if best_decision is not None:
            return best_decision

        # No feasible regular bin (even with evictions) -> signal infeasibility so
        # the caller can handle fallback placement consistently with other policies.
        raise PolicyInfeasibleError(f"SimBasePolicy could not place step {step.step_id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score(self, bin_id: int, step: StepSpec, instance: Instance) -> float:
        c_ji = float(instance.costs.assignment_costs[step.step_id, bin_id])
        volume = extract_volume(step.cap_matrix, instance.n, instance.m)
        lam_i = self.lam.get(bin_id, np.zeros_like(volume))
        if self.cfg.util_pricing.vector_prices:
            return c_ji + float(np.dot(lam_i, volume))
        lam_scalar = scalarize_vector(lam_i, "max")
        return c_ji + lam_scalar * scalarize_vector(volume, self.cfg.heuristics.size_key)
