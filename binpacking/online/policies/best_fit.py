from __future__ import annotations

from typing import List

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
    residual_vector,
    vector_fits,
)
from binpacking.core.block_utils import extract_volume


class BestFitOnlinePolicy(BaseOnlinePolicy):
    """
    Greedy online heuristic: choose the feasible bin that leaves the smallest
    remaining capacity after placing the step. Falls back to the dedicated
    fallback bin only if no regular bin can accommodate the step.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        ctx = build_context(self.cfg, instance, state)
        bins = candidate_bins(step, instance)
        if not bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online step {step.step_id}")

        allow_reshuffle = self.cfg.problem.allow_reassignment
        residual_mode = self.cfg.heuristics.residual_scalarization
        size_key = self.cfg.heuristics.size_key
        eviction_fn = lambda bin_id, ctx: eviction_order_desc(bin_id, ctx, size_key=size_key)
        destination_fn = lambda offline_id, origin_bin, ctx: select_reassignment_bin(
            offline_id,
            origin_bin,
            ctx,
            mode="residual",
            residual_mode=residual_mode,
        )
        volume = extract_volume(step.cap_matrix, instance.n, instance.m)
        fits: List[tuple[float, int]] = []
        overflows: List[tuple[float, int]] = []
        for bin_id in bins:
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
                step,
                ctx,
                eviction_order_fn=eviction_fn,
                destination_fn=destination_fn,
                allow_eviction=False,
            )
            if decision is not None:
                return decision

        # Otherwise, attempt bins in order of smallest overflow, allowing evictions
        if allow_reshuffle:
            for residual, target_bin in sorted(overflows, key=lambda pair: (pair[0], pair[1]), reverse=True):
                decision = execute_placement(
                    target_bin,
                    step,
                    ctx,
                    eviction_order_fn=eviction_fn,
                    destination_fn=destination_fn,
                    allow_eviction=True,
                )
                if decision is not None:
                    return decision

        raise PolicyInfeasibleError(f"BestFitOnlinePolicy could not place step {step.step_id}")

 
