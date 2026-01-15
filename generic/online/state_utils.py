from __future__ import annotations

from typing import Dict

import numpy as np

from generic.models import AssignmentState, Instance, OnlineItem, Decision


def clone_state(state: AssignmentState) -> AssignmentState:
    """Create a deep-ish copy of the assignment state."""
    return AssignmentState(
        load=state.load.copy(),
        assigned_bin=dict(state.assigned_bin),
        offline_evicted=set(state.offline_evicted),
    )


def build_volume_lookup(instance: Instance) -> Dict[int, np.ndarray]:
    """Map item_id -> volume vector for both offline and online items."""
    lookup: Dict[int, np.ndarray] = {item.id: item.volume for item in instance.offline_items}
    for online_item in instance.online_items or []:
        lookup[online_item.id] = online_item.volume
    return lookup


def apply_decision(
    decision: Decision,
    arriving_item: OnlineItem,
    state: AssignmentState,
    instance: Instance,
    volume_lookup: Dict[int, np.ndarray],
) -> None:
    """Mutate 'state' according to the decision taken for the arriving item."""
    regular_bins = len(instance.bins)
    fallback_idx = instance.fallback_bin_index

    # Evict offline items if requested
    for item_id, from_bin in decision.evicted_offline:
        if item_id not in state.assigned_bin:
            continue
        current_bin = state.assigned_bin[item_id]
        if current_bin != from_bin:
            from_bin = current_bin
        if item_id not in volume_lookup:
            raise KeyError(f"Unknown volume for item {item_id}")
        volume = volume_lookup[item_id]
        remove_from_bin(state, from_bin, volume)
        state.offline_evicted.add(item_id)
        del state.assigned_bin[item_id]

    # Reassign items
    for item_id, target_bin in decision.reassignments:
        if item_id not in volume_lookup:
            raise KeyError(f"Unknown volume for item {item_id}")
        volume = volume_lookup[item_id]
        add_to_bin(state, target_bin, volume, regular_bins, fallback_idx)
        state.assigned_bin[item_id] = target_bin

    # Place the arriving online item
    item_id, target_bin = decision.placed_item
    if item_id != arriving_item.id:
        raise ValueError(
            f"Decision item id {item_id} does not match arriving item {arriving_item.id}"
        )
    add_to_bin(state, target_bin, arriving_item.volume, regular_bins, fallback_idx)
    state.assigned_bin[item_id] = target_bin


def add_to_bin(
    state: AssignmentState,
    bin_id: int,
    volume: np.ndarray,
    regular_bins: int,
    fallback_idx: int,
) -> None:
    """Increase load of bin_id (or fallback) by volume."""
    if bin_id < 0:
        raise ValueError(f"Negative bin index {bin_id}")
    if bin_id < regular_bins:
        state.load[bin_id] += volume
    elif fallback_idx >= 0 and bin_id == fallback_idx:
        state.load[fallback_idx] += volume
    else:
        raise ValueError(
            f"Bin id {bin_id} is invalid for instance with {regular_bins} bins "
            f"and fallback index {fallback_idx}"
        )


def remove_from_bin(
    state: AssignmentState,
    bin_id: int,
    volume: np.ndarray,
) -> None:
    """Decrease load of bin_id by volume (saturating at zero)."""
    if bin_id < 0 or bin_id >= len(state.load):
        raise ValueError(f"Cannot remove volume from invalid bin {bin_id}")
    state.load[bin_id] = np.maximum(0.0, state.load[bin_id] - volume)


def count_fallback_items(state: AssignmentState, instance: Instance) -> int:
    """Count how many items are currently assigned to the fallback bin."""
    fallback_idx = instance.fallback_bin_index
    if fallback_idx < 0:
        return 0
    return sum(
        1 for bin_id in state.assigned_bin.values() if bin_id >= fallback_idx
    )
