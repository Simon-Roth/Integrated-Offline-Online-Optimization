from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Callable

import numpy as np

from generic.config import Config
from generic.general_utils import effective_capacity, vector_fits, volume_total
from generic.models import AssignmentState, Instance, OnlineItem, Decision

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
    loads = state.load.copy()
    assignments = dict(state.assigned_bin)
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
    return {item.id: item.volume for item in instance.offline_items}


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
    return [
        effective_capacity(bin_spec.capacity, enforce_slack, slack_fraction)
        for bin_spec in instance.bins
    ]


def eviction_penalty(volume: np.ndarray, cfg: Config) -> float:
    """Penalty incurred when evicting an offline item of given volume."""
    if cfg.costs.penalty_mode == "per_volume":
        return cfg.costs.per_volume_scale * volume_total(volume)
    return cfg.costs.reassignment_penalty


def execute_placement(
    target_bin: int,
    item: OnlineItem,
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
    fallback_idx = instance.fallback_bin_index
    regular_bins = len(instance.bins)
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
    reassignments: List[tuple[int, int]] = []
    incremental_cost = 0.0

    def _current_load(bin_id: int) -> np.ndarray:
        if fallback_idx >= 0 and bin_id == fallback_idx:
            return loads[fallback_idx]
        return loads[bin_id]

    capacity = ctx.effective_caps[target_bin]
    required_volume = item.volume

    if vector_fits(_current_load(target_bin), required_volume, capacity, TOLERANCE):
        # No eviction needed, simply reserve the space and pay assignment cost.
        loads[target_bin] += required_volume
        assignment_cost = float(instance.costs.assignment_costs[item.id, target_bin])
        decision = Decision(
            placed_item=(item.id, target_bin),
            evicted_offline=[],
            reassignments=[],
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
        if dest_bin == fallback_idx and fallback_idx >= 0:
            loads[fallback_idx] += volume
        else:
            loads[dest_bin] += volume
        assignments[offline_id] = dest_bin

        old_cost = instance.costs.assignment_costs[offline_id, origin]
        new_cost = instance.costs.assignment_costs[offline_id, dest_bin]
        penalty = eviction_penalty(volume, cfg)
        incremental_cost += new_cost - old_cost + penalty

        evicted_pairs.append((offline_id, origin))
        reassignments.append((offline_id, dest_bin))

    if not vector_fits(_current_load(target_bin), required_volume, capacity, TOLERANCE):
        return None

    loads[target_bin] += required_volume
    incremental_cost += float(instance.costs.assignment_costs[item.id, target_bin])
    decision = Decision(
        placed_item=(item.id, target_bin),
        evicted_offline=evicted_pairs,
        reassignments=reassignments,
        incremental_cost=incremental_cost,
    )
    # Commit simulated changes on success.
    base_loads[:] = loads
    base_assignments.clear()
    base_assignments.update(assignments)
    return decision
