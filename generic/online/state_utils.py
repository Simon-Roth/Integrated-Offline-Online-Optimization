from __future__ import annotations

from typing import Dict

import numpy as np

from generic.models import AssignmentState, Instance, OnlineItem, Decision


def clone_state(state: AssignmentState) -> AssignmentState:
    """Create a deep-ish copy of the assignment state."""
    return AssignmentState(
        load=state.load.copy(),
        assigned_action=dict(state.assigned_action),
        offline_evicted=set(state.offline_evicted),
    )


def build_cap_lookup(instance: Instance) -> Dict[int, np.ndarray]:
    """Map item_id -> A_t^{cap} for both offline and online items."""
    lookup: Dict[int, np.ndarray] = {item.id: item.cap_matrix for item in instance.offline_items}
    for online_item in instance.online_items or []:
        lookup[online_item.id] = online_item.cap_matrix
    return lookup


def apply_decision(
    decision: Decision,
    arriving_item: OnlineItem,
    state: AssignmentState,
    instance: Instance,
    cap_lookup: Dict[int, np.ndarray],
) -> None:
    """Mutate 'state' according to the decision taken for the arriving item."""
    n = instance.n
    fallback_idx = instance.fallback_action_index

    # Evict offline items if requested
    for item_id, from_action in decision.evicted_offline:
        if item_id not in state.assigned_action:
            continue
        current_action = state.assigned_action[item_id]
        if current_action != from_action:
            from_action = current_action
        if item_id not in cap_lookup:
            raise KeyError(f"Unknown A_cap for item {item_id}")
        cap_matrix = cap_lookup[item_id]
        remove_from_load(state, from_action, cap_matrix, n, fallback_idx)
        state.offline_evicted.add(item_id)
        del state.assigned_action[item_id]

    # Reassign items
    for item_id, target_action in decision.reassignments:
        if item_id not in cap_lookup:
            raise KeyError(f"Unknown A_cap for item {item_id}")
        cap_matrix = cap_lookup[item_id]
        add_to_load(state, target_action, cap_matrix, n, fallback_idx)
        state.assigned_action[item_id] = target_action

    # Place the arriving online item
    item_id, target_action = decision.placed_item
    if item_id != arriving_item.id:
        raise ValueError(
            f"Decision item id {item_id} does not match arriving item {arriving_item.id}"
        )
    add_to_load(state, target_action, arriving_item.cap_matrix, n, fallback_idx)
    state.assigned_action[item_id] = target_action


def add_to_load(
    state: AssignmentState,
    action_id: int,
    cap_matrix: np.ndarray,
    n: int,
    fallback_idx: int,
) -> None:
    """Increase resource usage by the column A_t^{cap}[:, action_id]."""
    if action_id < 0:
        raise ValueError(f"Negative action index {action_id}")
    if action_id < n:
        state.load += cap_matrix[:, action_id]
    elif fallback_idx >= 0 and action_id == fallback_idx:
        return
    else:
        raise ValueError(
            f"Action id {action_id} is invalid for instance with n={n} "
            f"and fallback index {fallback_idx}"
        )


def remove_from_load(
    state: AssignmentState,
    action_id: int,
    cap_matrix: np.ndarray,
    n: int,
    fallback_idx: int,
) -> None:
    """Decrease resource usage by A_t^{cap}[:, action_id] (saturating at zero)."""
    if action_id < 0:
        raise ValueError(f"Cannot remove from invalid action {action_id}")
    if action_id < n:
        state.load = np.maximum(0.0, state.load - cap_matrix[:, action_id])
        return
    if fallback_idx >= 0 and action_id == fallback_idx:
        return
    raise ValueError(f"Cannot remove from invalid action {action_id}")


def count_fallback_items(state: AssignmentState, instance: Instance) -> int:
    """Count how many items are currently assigned to the fallback action."""
    fallback_idx = instance.fallback_action_index
    if fallback_idx < 0:
        return 0
    return sum(
        1 for action_id in state.assigned_action.values() if action_id >= fallback_idx
    )
