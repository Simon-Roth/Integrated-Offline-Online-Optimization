# generic/data/offline_milp_assembly.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from generic.config import Config
from generic.models import Instance
from generic.general_utils import effective_capacity


@dataclass(frozen=True)
class OfflineMILPData:
    """
    Canonical MILP data for the offline stage: min c^T x s.t. A x <= b.
    """
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    var_shape: Tuple[int, int]
    fallback_idx: int
    m: int
    cap_matrices: np.ndarray
    feasible: Optional[np.ndarray] = None


def _as_length_vector(value: float | Sequence[float], length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(length, float(arr[0]))
    if arr.size != length:
        raise ValueError(f"Expected {length} values, got {arr.size}.")
    return arr


def build_offline_milp_data_from_arrays(
    *,
    cap_matrices: np.ndarray,
    costs: np.ndarray,
    feasible: np.ndarray,
    b: np.ndarray,
    fallback_idx: int,
    slack_enforce: bool,
    slack_fraction: float,
) -> OfflineMILPData:
    """
    Build offline MILP data from raw arrays. Independent of instance generation.
    cap_matrices: array of shape (M, m, n) with A_t^{cap} per item.
    """
    cap = np.asarray(cap_matrices, dtype=float)
    if cap.ndim == 2:
        cap = cap.reshape((1, cap.shape[0], cap.shape[1]))
    if cap.ndim != 3:
        raise ValueError("cap_matrices must have shape (M, m, n).")

    M = cap.shape[0]
    m = cap.shape[1]
    n = cap.shape[2]

    costs = np.asarray(costs, dtype=float)
    feas = np.asarray(feasible, dtype=int)
    b_vec = _as_length_vector(b, m)
    b_eff = effective_capacity(b_vec, slack_enforce, slack_fraction)

    if costs.ndim == 1 and M > 0:
        costs = costs.reshape((M, -1))
    if feas.ndim == 1 and M > 0:
        feas = feas.reshape((M, -1))

    cols = costs.shape[1] if costs.size else (n + (1 if fallback_idx >= 0 else 0))
    if cols not in (n, n + 1):
        raise ValueError(f"Cost matrix has {cols} columns, expected {n} or {n + 1}.")
    has_fallback = cols == n + 1
    if has_fallback:
        if fallback_idx < 0:
            fallback_idx = n
        if fallback_idx != n:
            raise ValueError("Fallback index must be n when fallback is enabled.")
    else:
        fallback_idx = -1

    if feas.size and feas.shape[1] != cols:
        raise ValueError("Feasibility matrix column count must match costs.")

    var_shape = (M, cols)
    num_vars = M * cols
    if M == 0:
        return OfflineMILPData(
            c=np.empty((0,), dtype=float),
            A=np.empty((0, 0), dtype=float),
            b=np.empty((0,), dtype=float),
            var_shape=var_shape,
            fallback_idx=fallback_idx,
            m=m,
            cap_matrices=cap,
            feasible=feas,
        )

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    # Capacity constraints (resource rows).
    for r in range(m):
        row = np.zeros(num_vars, dtype=float)
        for j in range(M):
            for i in range(n):
                row[j * cols + i] = cap[j, r, i]
        rows.append(row)
        rhs.append(float(b_eff[r]))

    # Assignment constraints (sum over all actions == 1).
    for j in range(M):
        row = np.zeros(num_vars, dtype=float)
        start = j * cols
        row[start : start + cols] = 1.0
        rows.append(row)
        rhs.append(1.0)
        rows.append(-row)
        rhs.append(-1.0)

    # Feasibility constraints (sum of infeasible edges == 0).
    for j in range(M):
        infeas_idx = np.flatnonzero(feas[j] == 0)
        if infeas_idx.size == 0:
            continue
        row = np.zeros(num_vars, dtype=float)
        start = j * cols
        for i in infeas_idx:
            row[start + i] = 1.0
        rows.append(row)
        rhs.append(0.0)

    A = np.vstack(rows) if rows else np.empty((0, num_vars), dtype=float)
    b_arr = np.asarray(rhs, dtype=float)
    c = costs.reshape(-1)

    return OfflineMILPData(
        c=c,
        A=A,
        b=b_arr,
        var_shape=var_shape,
        fallback_idx=fallback_idx,
        m=m,
        cap_matrices=cap,
        feasible=feas,
    )


def build_offline_milp_data(
    inst: Instance,
    cfg: Config,
) -> OfflineMILPData:
    """
    Convenience wrapper that assembles MILP data from an Instance + Config.
    """
    if inst.offline_items:
        cap_matrices = np.asarray([it.cap_matrix for it in inst.offline_items], dtype=float)
    else:
        cap_matrices = np.empty((0, int(inst.m), int(inst.n)), dtype=float)
    costs = np.asarray(inst.costs.assignment_costs[: len(inst.offline_items), :], dtype=float)
    feasible = np.asarray(inst.offline_feasible.feasible[: len(inst.offline_items), :], dtype=int)
    return build_offline_milp_data_from_arrays(
        cap_matrices=cap_matrices,
        costs=costs,
        feasible=feasible,
        b=inst.b,
        fallback_idx=inst.fallback_action_index,
        slack_enforce=cfg.slack.enforce_slack,
        slack_fraction=cfg.slack.fraction,
    )
