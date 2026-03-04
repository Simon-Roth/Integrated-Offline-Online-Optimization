from __future__ import annotations

"""
Dynamic Learning Algorithm (bgap extension)

This policy inherits the generic Dynamic Learning core and adds bgap-
specific eviction/reshuffling logic when allow_reassignment is enabled.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from generic.core.config import Config
from generic.core.models import AssignmentState, Decision, Instance, StepSpec
from generic.online.policies import GenericDynamicLearningPolicy
from generic.online.policies import PolicyInfeasibleError
from bgap.online.state_utils import (
    PlacementContext,
    build_context,
    eviction_order_desc,
    execute_placement,
    select_reassignment_bin,
    TOLERANCE,
)


class DynamicLearningPolicy(GenericDynamicLearningPolicy):
    """
    Dynamic price-updating policy with bgap-specific evictions.
    """

    def __init__(
        self,
        cfg: Config,
        price_path: Optional[Path] = None,
        *,
        pricing_sample_seed: int | None = None,
    ) -> None:
        super().__init__(cfg, price_path, pricing_sample_seed=pricing_sample_seed)

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        try:
            decision = super().select_action(step, state, instance)
        except PolicyInfeasibleError:
            decision = None

        if decision is not None:
            fallback_idx = instance.fallback_option_index
            if fallback_idx < 0 or decision.placed_step[1] != fallback_idx:
                return decision
            if not self.cfg.problem.allow_reassignment:
                return decision

        # Allow evictions if no feasible bin remained.
        best_decision: Optional[Decision] = None
        best_score = float("inf")
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
        candidate_options = self._candidate_options(step, instance)
        for option_id in candidate_options:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            score = self._score(option_id, step, instance)
            if score >= best_score - 1e-12:
                continue
            decision = execute_placement(
                option_id,
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
            self._mark_processed(step)
            return best_decision

        if decision is not None:
            return decision

        # No feasible regular option (even with evictions) -> signal infeasibility.
        raise PolicyInfeasibleError(
            f"DynamicLearningPolicy could not place step {step.step_id}"
        )
