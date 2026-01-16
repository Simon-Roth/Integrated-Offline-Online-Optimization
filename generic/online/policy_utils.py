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
    N: int,
    cols: int,
    fallback_idx: int,
) -> np.ndarray:
    """Return feasibility row for the current item (with fallback column if enabled)."""
    if feasible_row is not None:
        row = np.asarray(feasible_row, dtype=int).reshape(-1)
        if row.size == cols:
            if row.sum() == 0:
                raise PolicyInfeasibleError("No feasible bin for current item.")
            return row.reshape(1, -1)
        if row.size == N:
            if cols > N:
                fallback_val = 1 if cfg.problem.fallback_allowed_online else 0
                row = np.concatenate([row, np.array([fallback_val], dtype=int)])
            if row.sum() == 0:
                raise PolicyInfeasibleError("No feasible bin for current item.")
            return row.reshape(1, -1)
        raise ValueError(f"Feasible row has length {row.size}, expected {N} or {cols}.")

    row = np.zeros((cols,), dtype=int)
    for bin_id in item.feasible_bins:
        if 0 <= bin_id < N:
            row[bin_id] = 1
    if cols > N:
        row[fallback_idx] = 1 if cfg.problem.fallback_allowed_online else 0
    if row.sum() == 0:
        raise PolicyInfeasibleError("No feasible bin for current item.")
    return row.reshape(1, -1)


def remaining_capacities(
    cfg: Config,
    state: AssignmentState,
    instance: Instance,
    effective_caps: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Return remaining regular-bin and fallback capacities for the current state."""
    N = len(instance.bins)
    load = np.asarray(state.load, dtype=float)
    if load.ndim == 1:
        load = load.reshape((load.shape[0], -1))
    if load.shape[0] < N:
        raise ValueError("State load has fewer rows than number of bins.")

    if effective_caps is None:
        caps = np.asarray([b.capacity for b in instance.bins], dtype=float)
        if caps.ndim == 1:
            caps = caps.reshape((N, -1))
        enforce_slack = bool(cfg.slack.enforce_slack and cfg.slack.apply_to_online)
        effective_caps = effective_capacity(caps, enforce_slack, cfg.slack.fraction)
    remaining = np.maximum(0.0, effective_caps - load[:N])

    fallback_idx = instance.fallback_bin_index
    if fallback_idx < 0:
        return remaining, 0.0
    dims = remaining.shape[1] if remaining.size else int(load.shape[1])
    use_slack = bool(cfg.slack.enforce_slack and cfg.slack.apply_to_online)
    slack_fraction = cfg.slack.fraction if use_slack else 0.0
    fallback_cap = np.asarray(cfg.problem.fallback_capacity_online, dtype=float)
    if fallback_cap.size == 1:
        fallback_cap = np.full((dims,), float(fallback_cap))
    fallback_cap = fallback_cap.reshape((dims,))
    fallback_effective = effective_capacity(
        fallback_cap,
        use_slack,
        slack_fraction,
    )
    fallback_load = (
        load[fallback_idx] if load.shape[0] > fallback_idx else np.zeros((dims,))
    )
    fallback_remaining = np.maximum(0.0, fallback_effective - fallback_load)
    return remaining, fallback_remaining


def lookup_assignment_cost(
    cfg: Config,
    instance: Instance,
    item_id: int,
    bin_id: int,
) -> float:
    """Return the assignment cost for item->bin (fallback uses huge_fallback)."""
    costs = instance.costs.assignment_costs
    if (
        costs is not None
        and costs.size
        and item_id < costs.shape[0]
        and bin_id < costs.shape[1]
    ):
        return float(costs[item_id, bin_id])
    if bin_id == instance.fallback_bin_index:
        return float(cfg.costs.huge_fallback)
    return 0.0
