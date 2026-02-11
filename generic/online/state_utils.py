from __future__ import annotations

from typing import Dict

import numpy as np

from generic.core.models import AssignmentState, Instance, StepSpec, Decision


def clone_state(state: AssignmentState) -> AssignmentState:
    """Create a deep-ish copy of the assignment state."""
    return AssignmentState(
        load=state.load.copy(),
        assigned_option=dict(state.assigned_option),
        offline_evicted_steps=set(state.offline_evicted_steps),
    )


def build_cap_lookup(instance: Instance) -> Dict[int, np.ndarray]:
    """Map step_id -> A_t^{cap} for both offline and online steps."""
    lookup: Dict[int, np.ndarray] = {step.step_id: step.cap_matrix for step in instance.offline_steps}
    for online_step in instance.online_steps or []:
        lookup[online_step.step_id] = online_step.cap_matrix
    return lookup


def apply_decision(
    decision: Decision,
    arriving_step: StepSpec,
    state: AssignmentState,
    instance: Instance,
    cap_lookup: Dict[int, np.ndarray],
) -> None:
    """Mutate 'state' according to the decision taken for the arriving step."""
    n = instance.n
    fallback_idx = instance.fallback_option_index

    # Evict offline steps if requested
    for step_id, from_option in decision.evicted_offline_steps:
        if step_id not in state.assigned_option:
            continue
        current_option = state.assigned_option[step_id]
        if current_option != from_option:
            from_option = current_option
        if step_id not in cap_lookup:
            raise KeyError(f"Unknown A_cap for step {step_id}")
        cap_matrix = cap_lookup[step_id]
        remove_from_load(state, from_option, cap_matrix, n, fallback_idx)
        state.offline_evicted_steps.add(step_id)
        del state.assigned_option[step_id]

    # Reassign offline steps
    for step_id, target_option in decision.reassigned_offline_steps:
        if step_id not in cap_lookup:
            raise KeyError(f"Unknown A_cap for step {step_id}")
        cap_matrix = cap_lookup[step_id]
        add_to_load(state, target_option, cap_matrix, n, fallback_idx)
        state.assigned_option[step_id] = target_option

    # Place the arriving online step
    step_id, target_option = decision.placed_step
    if step_id != arriving_step.step_id:
        raise ValueError(
            f"Decision step id {step_id} does not match arriving step {arriving_step.step_id}"
        )
    add_to_load(state, target_option, arriving_step.cap_matrix, n, fallback_idx)
    state.assigned_option[step_id] = target_option


def add_to_load(
    state: AssignmentState,
    option_id: int,
    cap_matrix: np.ndarray,
    n: int,
    fallback_idx: int,
) -> None:
    """Increase resource usage by the column A_t^{cap}[:, option_id]."""
    if option_id < 0:
        raise ValueError(f"Negative option index {option_id}")
    if option_id < n:
        state.load += cap_matrix[:, option_id]
    elif fallback_idx >= 0 and option_id == fallback_idx:
        return
    else:
        raise ValueError(
            f"Option id {option_id} is invalid for instance with n={n} "
            f"and fallback index {fallback_idx}"
        )


def remove_from_load(
    state: AssignmentState,
    option_id: int,
    cap_matrix: np.ndarray,
    n: int,
    fallback_idx: int,
) -> None:
    """Decrease resource usage by A_t^{cap}[:, option_id] (saturating at zero)."""
    if option_id < 0:
        raise ValueError(f"Cannot remove from invalid option {option_id}")
    if option_id < n:
        state.load = np.maximum(0.0, state.load - cap_matrix[:, option_id])
        return
    if fallback_idx >= 0 and option_id == fallback_idx:
        return
    raise ValueError(f"Cannot remove from invalid option {option_id}")


def count_fallback_steps(state: AssignmentState, instance: Instance) -> int:
    """Count how many steps are currently assigned to the fallback option."""
    fallback_idx = instance.fallback_option_index
    if fallback_idx < 0:
        return 0
    return sum(
        1 for option_id in state.assigned_option.values() if option_id >= fallback_idx
    )
