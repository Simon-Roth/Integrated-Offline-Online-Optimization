from __future__ import annotations

from typing import Optional

import numpy as np

from generic.config import Config
from generic.general_utils import effective_capacity
from generic.models import AssignmentState, Instance, OnlineItem
class PolicyInfeasibleError(Exception):
    """Raised when a policy cannot produce a feasible placement for the arriving item."""

    pass


def current_cost_row(
    cfg: Config,
    instance: Instance,
    item_id: int,
    cols: int,
) -> np.ndarray:
    """Return the assignment cost row for the current item."""
    row = np.full((cols,), cfg.costs.huge_fallback, dtype=float)
    costs = instance.costs.assignment_costs
    if costs is None or not costs.size:
        return row
    if item_id >= costs.shape[0]:
        return row
    row[: min(cols, costs.shape[1])] = costs[item_id, : min(cols, costs.shape[1])]
    return row


def current_feasible_row(
    cfg: Config,
    item: OnlineItem,
    feasible_row: Optional[np.ndarray],
    n: int,
    cols: int,
    fallback_idx: int,
) -> np.ndarray:
    """Return feasibility row for the current item (with fallback column if enabled)."""
    if feasible_row is not None:
        row = np.asarray(feasible_row, dtype=int).reshape(-1)
        if row.size == cols:
            if row.sum() == 0:
                raise PolicyInfeasibleError("No feasible action for current item.")
            return row.reshape(1, -1)
        if row.size == n:
            if cols > n:
                fallback_val = 1 if cfg.problem.fallback_allowed_online else 0
                row = np.concatenate([row, np.array([fallback_val], dtype=int)])
            if row.sum() == 0:
                raise PolicyInfeasibleError("No feasible action for current item.")
            return row.reshape(1, -1)
        raise ValueError(f"Feasible row has length {row.size}, expected {n} or {cols}.")

    row = np.zeros((cols,), dtype=int)
    for action_id in item.feasible_actions:
        if 0 <= action_id < n:
            row[action_id] = 1
    if cols > n:
        row[fallback_idx] = 1 if cfg.problem.fallback_allowed_online else 0
    if row.sum() == 0:
        raise PolicyInfeasibleError("No feasible action for current item.")
    return row.reshape(1, -1)


def remaining_capacities(
    cfg: Config,
    state: AssignmentState,
    instance: Instance,
    effective_caps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return remaining resource capacities (b - usage) for the current state."""
    load = np.asarray(state.load, dtype=float).reshape(-1)
    b_vec = np.asarray(instance.b, dtype=float).reshape(-1)
    if load.shape[0] != b_vec.shape[0]:
        raise ValueError("State load and b must have the same length.")
    if effective_caps is None:
        enforce_slack = bool(cfg.slack.enforce_slack and cfg.slack.apply_to_online)
        effective_caps = effective_capacity(b_vec, enforce_slack, cfg.slack.fraction)
    remaining = np.maximum(0.0, effective_caps - load)
    return remaining


def lookup_assignment_cost(
    cfg: Config,
    instance: Instance,
    item_id: int,
    action_id: int,
) -> float:
    """Return the assignment cost for item->action (fallback uses huge_fallback)."""
    costs = instance.costs.assignment_costs
    if (
        costs is not None
        and costs.size
        and item_id < costs.shape[0]
        and action_id < costs.shape[1]
    ):
        return float(costs[item_id, action_id])
    if action_id == instance.fallback_action_index:
        return float(cfg.costs.huge_fallback)
    return 0.0
