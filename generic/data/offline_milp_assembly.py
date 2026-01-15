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
    dimensions: int
    volumes: np.ndarray
    feasible: Optional[np.ndarray] = None


def _as_dim_vector(value: float | Sequence[float], dims: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return np.full(dims, float(arr))
    if arr.size != dims:
        raise ValueError(f"Expected {dims} values, got {arr.size}.")
    return arr.reshape((dims,))


def build_offline_milp_data_from_arrays(
    *,
    volumes: np.ndarray,
    costs: np.ndarray,
    feasible: np.ndarray,
    capacities: np.ndarray,
    fallback_idx: int,
    fallback_capacity: float | Sequence[float],
    slack_enforce: bool,
    slack_fraction: float,
) -> OfflineMILPData:
    """
    Build offline MILP data from raw arrays. Independent of instance generation.
    """
    vols = np.asarray(volumes, dtype=float)
    if vols.ndim == 1:
        vols = vols.reshape((-1, 1))
    costs = np.asarray(costs, dtype=float)
    feas = np.asarray(feasible, dtype=int)
    caps = np.asarray(capacities, dtype=float)
    if caps.ndim == 1:
        caps = caps.reshape((-1, 1))

    M = vols.shape[0]
    if costs.ndim == 1 and M > 0:
        costs = costs.reshape((M, -1))
    if feas.ndim == 1 and M > 0:
        feas = feas.reshape((M, -1))
    dims = vols.shape[1] if vols.size else (caps.shape[1] if caps.size else 1)
    N = caps.shape[0] if caps.size else 0
    if costs.size:
        cols = costs.shape[1]
    else:
        cols = N + (1 if fallback_idx >= 0 else 0)
    has_fallback = cols == N + 1
    if cols not in (N, N + 1):
        raise ValueError(f"Cost matrix has {cols} columns, expected {N} or {N + 1}.")
    if feas.size and feas.shape[1] != cols:
        raise ValueError("Feasibility matrix column count must match costs.")
    if has_fallback:
        fallback_idx_used = N if fallback_idx < 0 else fallback_idx
        if fallback_idx_used != N:
            raise ValueError("Fallback index must be N when fallback is enabled.")
    else:
        fallback_idx_used = -1
    var_shape = (M, cols)
    num_vars = M * cols
    if M == 0:
        return OfflineMILPData(
            c=np.empty((0,), dtype=float),
            A=np.empty((0, 0), dtype=float),
            b=np.empty((0,), dtype=float),
            var_shape=var_shape,
            fallback_idx=fallback_idx_used,
            dimensions=dims,
            volumes=vols,
            feasible=feas,
        )

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    # Capacity constraints (regular bins).
    for i in range(N):
        cap_i = effective_capacity(
            caps[i],
            slack_enforce,
            slack_fraction,
        )
        for d in range(dims):
            row = np.zeros(num_vars, dtype=float)
            for j in range(M):
                row[j * cols + i] = vols[j, d]
            rows.append(row)
            rhs.append(float(cap_i[d]))

    # Fallback capacity constraints (explicit, large by config).
    if has_fallback:
        fallback_cap = _as_dim_vector(fallback_capacity, dims)
        for d in range(dims):
            row = np.zeros(num_vars, dtype=float)
            for j in range(M):
                row[j * cols + fallback_idx_used] = vols[j, d]
            rows.append(row)
            rhs.append(float(fallback_cap[d]))

    # Assignment constraints (sum over all bins == 1).
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
    b = np.asarray(rhs, dtype=float)
    c = costs.reshape(-1)

    return OfflineMILPData(
        c=c,
        A=A,
        b=b,
        var_shape=var_shape,
        fallback_idx=fallback_idx_used,
        dimensions=dims,
        volumes=vols,
        feasible=feas,
    )


def build_offline_milp_data(
    inst: Instance,
    cfg: Config,
) -> OfflineMILPData:
    """
    Convenience wrapper that assembles MILP data from an Instance + Config.
    """
    volumes = np.array([it.volume for it in inst.offline_items], dtype=float)
    costs = np.asarray(inst.costs.assignment_costs[: len(inst.offline_items), :], dtype=float)
    feasible = np.asarray(inst.offline_feasible.feasible[: len(inst.offline_items), :], dtype=int)
    capacities = np.asarray([b.capacity for b in inst.bins], dtype=float)
    return build_offline_milp_data_from_arrays(
        volumes=volumes,
        costs=costs,
        feasible=feasible,
        capacities=capacities,
        fallback_idx=inst.fallback_bin_index,
        fallback_capacity=cfg.problem.fallback_capacity_offline,
        slack_enforce=cfg.slack.enforce_slack,
        slack_fraction=cfg.slack.fraction,
    )
