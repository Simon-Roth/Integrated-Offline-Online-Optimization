from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Callable

import numpy as np

from generic.core.config import Config
from generic.core.utils import (
    option_is_feasible,
    effective_capacity,
    feasible_option_indices,
    residual_vector,
    scalarize_vector,
    usage_total,
    vector_fits,
)
from generic.core.models import AssignmentState, Instance, StepSpec, Decision
from bgap.core.block_utils import block_dim, extract_volume, split_capacities

TOLERANCE = 1e-9


@dataclass
class PlacementContext:
    """
    Lightweight simulation context used by online heuristics while planning.
    Mutations to `loads` and `assignments` affect only the context copy; the
    actual AssignmentState is updated later via `apply_decision`.
    """

    cfg: Config
    instance: Instance
    loads: np.ndarray
    assignments: Dict[int, int]
    effective_caps: np.ndarray
    offline_volumes: Dict[int, np.ndarray]


def build_context(cfg: Config, instance: Instance, state: AssignmentState) -> PlacementContext:
    """Create a placement context snapshot from the live state."""
    n = instance.n
    d = block_dim(n, instance.m)
    load_vec = np.asarray(state.load, dtype=float).reshape(-1)
    if load_vec.shape[0] != n * d:
        raise ValueError("State load does not match expected block structure.")
    loads = load_vec.reshape((n, d)).copy()
    assignments = dict(state.assigned_option)
    use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
    effective_caps = np.asarray(effective_capacities(instance, cfg, use_slack=use_slack))
    offline_vols = offline_volumes(instance)
    return PlacementContext(
        cfg=cfg,
        instance=instance,
        loads=loads,
        assignments=assignments,
        effective_caps=effective_caps,
        offline_volumes=offline_vols,
    )


EvictionOrderFn = Callable[[int, PlacementContext], List[int]]
DestinationFn = Callable[[int, int, PlacementContext], Optional[int]]


def offline_volumes(instance: Instance) -> Dict[int, np.ndarray]:
    """Map offline item_id -> volume vector."""
    n = instance.n
    m = instance.m
    return {item.step_id: extract_volume(item.cap_matrix, n, m) for item in instance.offline_steps}


def effective_capacities(
    instance: Instance,
    cfg: Config,
    *,
    use_slack: Optional[bool] = None,
) -> List[np.ndarray]:
    """Effective capacities for each regular bin, with optional slack override."""
    if use_slack is None:
        enforce_slack = cfg.slack.enforce_slack
    else:
        enforce_slack = use_slack
    slack_fraction = cfg.slack.fraction if enforce_slack else 0.0
    return list(
        effective_capacity(split_capacities(instance.b, instance.n), enforce_slack, slack_fraction)
    )


def eviction_penalty(usage: np.ndarray, cfg: Config) -> float:
    """Penalty incurred when evicting an offline item of given usage."""
    if cfg.costs.penalty_mode == "per_usage":
        return cfg.costs.per_usage_scale * usage_total(usage)
    return cfg.costs.reassignment_penalty


def execute_placement(
    target_bin: int,
    item: StepSpec,
    ctx: PlacementContext,
    *,
    eviction_order_fn: EvictionOrderFn,
    destination_fn: DestinationFn,
    allow_eviction: bool = True,
) -> Optional[Decision]:
    """
    Attempt to place `item` into `target_bin`, optionally evicting offline items using
    policy-defined helpers. Returns a fully specified Decision on success, None otherwise.
    
    -> Used for "simulating" a placement. Does not directly place like apply_decision in generic/state_utils.py
    """
    base_loads = ctx.loads
    base_assignments = ctx.assignments

    # Work on copies so failed attempts do not leak mutations.
    loads = base_loads.copy()
    assignments = dict(base_assignments)
    instance = ctx.instance
    cfg = ctx.cfg
    fallback_idx = instance.fallback_option_index
    regular_bins = instance.n
    # Mutable context so destination_fn sees up-to-date loads during evictions.
    ctx_mut = PlacementContext(
        cfg=ctx.cfg,
        instance=instance,
        loads=loads,
        assignments=assignments,
        effective_caps=ctx.effective_caps,
        offline_volumes=ctx.offline_volumes,
    )

    evicted_pairs: List[tuple[int, int]] = []
    reassigned_offline_steps: List[tuple[int, int]] = []
    incremental_cost = 0.0

    def _current_load(bin_id: int) -> np.ndarray:
        if fallback_idx >= 0 and bin_id == fallback_idx:
            return np.zeros_like(loads[0])
        return loads[bin_id]

    capacity = ctx.effective_caps[target_bin]
    required_volume = extract_volume(item.cap_matrix, instance.n, instance.m)

    if vector_fits(_current_load(target_bin), required_volume, capacity, TOLERANCE):
        # No eviction needed, simply reserve the space and pay assignment cost.
        loads[target_bin] += required_volume
        assignment_cost = float(instance.costs.assignment_costs[item.step_id, target_bin])
        decision = Decision(
            placed_step=(item.step_id, target_bin),
            evicted_offline_steps=[],
            reassigned_offline_steps=[],
            incremental_cost=assignment_cost,
        )
        # Commit simulated changes back to the shared context.
        base_loads[:] = loads
        base_assignments.clear()
        base_assignments.update(assignments)
        return decision

    if not allow_eviction:
        return None

    offline_candidates = eviction_order_fn(target_bin, ctx)
    if not offline_candidates:
        return None

    for offline_id in offline_candidates:
        if vector_fits(_current_load(target_bin), required_volume, capacity, TOLERANCE):
            break

        if offline_id not in assignments:
            continue
        origin = assignments[offline_id]
        if origin != target_bin:
            continue

        dest_bin = destination_fn(offline_id, origin, ctx_mut)
        if dest_bin is None:
            continue

        if dest_bin == fallback_idx and fallback_idx < 0:
            continue

        volume = ctx.offline_volumes.get(offline_id)
        if volume is None:
            continue
        if dest_bin < regular_bins:
            dest_cap = ctx_mut.effective_caps[dest_bin]
            if not vector_fits(loads[dest_bin], volume, dest_cap, TOLERANCE):
                continue
        loads[target_bin] -= volume
        if dest_bin != fallback_idx:
            loads[dest_bin] += volume
        assignments[offline_id] = dest_bin

        old_cost = instance.costs.assignment_costs[offline_id, origin]
        new_cost = instance.costs.assignment_costs[offline_id, dest_bin]
        penalty = eviction_penalty(volume, cfg)
        incremental_cost += new_cost - old_cost + penalty

        evicted_pairs.append((offline_id, origin))
        reassigned_offline_steps.append((offline_id, dest_bin))

    if not vector_fits(_current_load(target_bin), required_volume, capacity, TOLERANCE):
        return None

    loads[target_bin] += required_volume
    incremental_cost += float(instance.costs.assignment_costs[item.step_id, target_bin])
    decision = Decision(
        placed_step=(item.step_id, target_bin),
        evicted_offline_steps=evicted_pairs,
        reassigned_offline_steps=reassigned_offline_steps,
        incremental_cost=incremental_cost,
    )
    # Commit simulated changes on success.
    base_loads[:] = loads
    base_assignments.clear()
    base_assignments.update(assignments)
    return decision


def candidate_bins(item: StepSpec, instance: Instance) -> List[int]:
    """Feasible regular bins for this item (excludes fallback)."""
    regular_bins = list(range(instance.n))
    return feasible_option_indices(item.feas_matrix, item.feas_rhs, option_ids=regular_bins)


def eviction_order_desc(bin_id: int, ctx: PlacementContext, *, size_key: str) -> List[int]:
    """Offline items in a bin, sorted by size (descending)."""
    offline_ids = [
        itm_id
        for itm_id, assigned_bin in ctx.assignments.items()
        if assigned_bin == bin_id and itm_id < len(ctx.instance.offline_steps)
    ]
    zero_vec = np.zeros_like(ctx.effective_caps[0])
    offline_ids.sort(
        key=lambda oid: scalarize_vector(ctx.offline_volumes.get(oid, zero_vec), size_key),
        reverse=True,
    )
    return offline_ids


def select_reassignment_bin(
    offline_id: int,
    origin_bin: int,
    ctx: PlacementContext,
    *,
    mode: str,
    residual_mode: str,
) -> Optional[int]:
    """Choose a reassignment bin for an evicted offline item (cost or residual)."""
    zero_vec = np.zeros_like(ctx.effective_caps[0])
    volume = ctx.offline_volumes.get(offline_id, zero_vec)
    instance = ctx.instance
    regular_bins = instance.n
    fallback_idx = instance.fallback_option_index
    offline_item = instance.offline_steps[offline_id]

    best_candidate: Optional[int] = None
    best_cost = float("inf")
    best_residual = float("inf")

    for candidate in range(regular_bins):
        if candidate == origin_bin or not option_is_feasible(
            offline_item.feas_matrix, offline_item.feas_rhs, candidate
        ):
            continue
        residual_vec = residual_vector(ctx.loads[candidate], volume, ctx.effective_caps[candidate])
        if not vector_fits(ctx.loads[candidate], volume, ctx.effective_caps[candidate], TOLERANCE):
            continue
        residual_score = scalarize_vector(residual_vec, residual_mode)
        if mode == "residual":
            if residual_score < best_residual:
                best_residual = residual_score
                best_candidate = candidate
        else:
            cost = float(instance.costs.assignment_costs[offline_id, candidate])
            if cost < best_cost - 1e-9 or (
                abs(cost - best_cost) <= 1e-9 and residual_score < best_residual
            ):
                best_cost = cost
                best_residual = residual_score
                best_candidate = candidate

    if best_candidate is not None:
        return best_candidate

    if option_is_feasible(offline_item.feas_matrix, offline_item.feas_rhs, fallback_idx):
        return fallback_idx

    return None
