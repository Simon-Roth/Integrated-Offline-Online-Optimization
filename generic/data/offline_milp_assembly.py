# generic/data/offline_milp_assembly.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from generic.core.config import Config
from generic.core.models import Instance
from generic.core.utils import effective_capacity
from generic.data.generator_utils import _as_length_vector


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


def build_offline_milp_data_from_arrays(
    *,
    cap_matrices: np.ndarray,
    costs: np.ndarray,
    feas_matrices: Sequence[np.ndarray],
    feas_rhs: Sequence[np.ndarray],
    b: np.ndarray,
    fallback_idx: int,
    slack_enforce: bool,
    slack_fraction: float,
) -> OfflineMILPData:
    """
    Build offline MILP data from raw arrays. Independent of instance generation.
    cap_matrices: array of shape (M, m, n) with A_t^{cap} per step.
    feas_matrices: list of A_t^{feas} matrices (shape p_t x n' (n' is n or n+1 dependent on fallback | pt is num of local feas constraints for step t))
    feas_rhs: list of b_t vectors (shape p_t,)
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
    b_vec = _as_length_vector(b, m)
    b_eff = effective_capacity(b_vec, slack_enforce, slack_fraction)

    if costs.ndim == 1 and M > 0:
        costs = costs.reshape((M, -1))
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

    if len(feas_matrices) != M or len(feas_rhs) != M:
        raise ValueError("feas_matrices and feas_rhs must have length M.")

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
    
    #\sum_{t=1}^M \sum_{i=1}^n A_t^{cap}[r,i] , x_{t,i} \le b_r
    # M Zeilen


    # Local feasibility constraints A_t^{feas} x_t = b_t (added as two inequalities).
    for j in range(M):
        A_feas = np.asarray(feas_matrices[j], dtype=float)
        b_t = np.asarray(feas_rhs[j], dtype=float).reshape(-1)
        if A_feas.ndim != 2:
            raise ValueError("Each feas_matrix must be 2D.")
        if A_feas.shape[1] != cols:
            raise ValueError("Feasibility matrix column count must match costs.")
        if A_feas.shape[0] != b_t.size:
            raise ValueError("Feasibility rhs length must match number of rows.")
        start = j * cols
        for row_idx in range(A_feas.shape[0]):
            row = np.zeros(num_vars, dtype=float)
            row[start : start + cols] = A_feas[row_idx]
            rows.append(row)
            rhs.append(float(b_t[row_idx]))
            rows.append(-row)
            rhs.append(float(-b_t[row_idx]))

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
    )
    # A: m capacity rows + 2 * sum_t p_t feasibility rows | M * n (or n+1) columns
    
    


def build_offline_milp_data(
    inst: Instance,
    cfg: Config,
) -> OfflineMILPData:
    """
    Convenience wrapper that assembles MILP data from an Instance + Config.
    """
    if inst.offline_steps:
        cap_matrices = np.asarray([it.cap_matrix for it in inst.offline_steps], dtype=float)
    else:
        cap_matrices = np.empty((0, int(inst.m), int(inst.n)), dtype=float)
    costs = np.asarray(inst.costs.assignment_costs[: len(inst.offline_steps), :], dtype=float)
    return build_offline_milp_data_from_arrays(
        cap_matrices=cap_matrices,
        costs=costs,
        feas_matrices=[it.feas_matrix for it in inst.offline_steps],
        feas_rhs=[it.feas_rhs for it in inst.offline_steps],
        b=inst.b,
        fallback_idx=inst.fallback_option_index,
        slack_enforce=cfg.slack.enforce_slack,
        slack_fraction=cfg.slack.fraction,
    )
