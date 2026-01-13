# binpacking/data/generators.py
from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import numpy as np
from generic.config import Config
from generic.models import BinSpec, ItemSpec, Instance, Costs, FeasibleGraph, OnlineItem
from generic.general_utils import validate_capacities, make_rng, validate_mask


def _as_dim_vector(value: float | Sequence[float], dims: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return np.full(dims, float(arr))
    if arr.size != dims:
        raise ValueError(f"Expected {dims} values, got {arr.size}.")
    return arr.reshape((dims,))


def _normalize_beta_params(beta, dims: int) -> np.ndarray:
    """
    Return array of shape (dims, 2) with alpha/beta per dimension.
    """
    arr = np.asarray(beta, dtype=float)
    if arr.shape == (2,):
        return np.tile(arr, (dims, 1))
    if arr.shape == (dims, 2):
        return arr
    raise ValueError("beta parameters must be length-2 or shape (dims, 2).")


def _normalize_bounds(bounds, dims: int) -> np.ndarray:
    """
    Return array of shape (dims, 2) with lower/upper bounds per dimension.
    """
    arr = np.asarray(bounds, dtype=float)
    if arr.shape == (2,):
        return np.tile(arr, (dims, 1))
    if arr.shape == (dims, 2):
        return arr
    raise ValueError("bounds must be length-2 or shape (dims, 2).")


def _coerce_capacities(
    cfg: Config,
    rng: np.random.Generator,
    N: int,
    dims: int,
) -> np.ndarray:
    base_caps = np.array(cfg.problem.capacities, dtype=float) if cfg.problem.capacities else np.array([], dtype=float)
    if base_caps.size > 0 and base_caps.ndim == 1 and dims > 1:
        if base_caps.size == N:
            base_caps = np.tile(base_caps.reshape((N, 1)), (1, dims))
        elif base_caps.size % dims == 0:
            base_caps = base_caps.reshape((-1, dims))
        else:
            raise ValueError("capacity list must be length N or N*d for vector capacities.")

    if base_caps.ndim == 1 and dims == 1:
        base_caps = base_caps.reshape((-1, 1))

    if base_caps.size < N * dims:
        mean = _as_dim_vector(getattr(cfg.problem, "capacity_mean", 1.0), dims)
        std = _as_dim_vector(getattr(cfg.problem, "capacity_std", 0.0), dims)
        extra = N - (base_caps.shape[0] if base_caps.size else 0)
        sampled = rng.normal(loc=mean, scale=np.maximum(std, 0.0), size=(extra, dims)) if extra > 0 else np.empty((0, dims))
        if np.all(std == 0.0):
            sampled[:] = mean
        sampled = np.clip(sampled, 1e-6, None)
        capacities = np.vstack([base_caps, sampled]) if base_caps.size else sampled
    else:
        capacities = base_caps[:N, :]
    validate_capacities(capacities)
    cfg.problem.capacities = capacities.tolist()
    return capacities


def _sample_assignment_costs(
    cfg: Config,
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Sample assignment costs using the configured beta distribution and bounds.
    """
    if rows <= 0 or cols <= 0:
        return np.empty((rows, cols), dtype=float)
    alpha_cost, beta_cost = cfg.costs.assign_beta
    lo_cost, hi_cost = cfg.costs.assign_bounds
    assert alpha_cost > 0 and beta_cost > 0, "assign_beta must be positive."
    assert lo_cost < hi_cost, "assign_bounds must satisfy lower < upper."
    draws = rng.beta(alpha_cost, beta_cost, size=(rows, cols)).astype(float)
    return (lo_cost + draws * (hi_cost - lo_cost)).astype(float)

def generate_offline_instance(cfg: Config, seed: int) -> Instance:
    """
    Create an offline instance at t=0:
    - N regular bins with capacities
    - M_off offline items with volumes ~ Beta(a,b)
    - G_off feasibility mask via Erdos-Renyi with p_off
    - Cost matrix with explicit fallback column for offline items
    """
    rng = make_rng(seed)
    N = cfg.problem.N
    M_off = cfg.problem.M_off
    dims = max(1, int(getattr(cfg.problem, "dimensions", 1)))
    capacities = _coerce_capacities(cfg, rng, N, dims)

    # Bins (0..N-1). Fallback's "index" will be N (no capacity).
    bins = [BinSpec(id=i, capacity=np.array(capacities[i], dtype=float)) for i in range(N)]
    fallback_idx = N  # IMPORTANT: 0-based indexing, fallback is the (N)-th column

    # Offline volumes ~ Beta(a, b)
    beta_params = _normalize_beta_params(cfg.volumes.offline_beta, dims)
    bounds = _normalize_bounds(cfg.volumes.offline_bounds, dims)
    offline_volumes = np.zeros((M_off, dims), dtype=float)
    for d in range(dims):
        alpha_vol_off, beta_vol_off = beta_params[d]
        lo, hi = bounds[d]
        assert alpha_vol_off > 0 and beta_vol_off > 0, "offline_beta must be positive."
        assert 0.0 < lo < hi, "offline_bounds must satisfy 0 < lower < upper."
        u = rng.beta(alpha_vol_off, beta_vol_off, size=M_off).astype(float)
        offline_volumes[:, d] = (lo + u * (hi - lo)).astype(float)
    offline_items = [ItemSpec(id=j, volume=offline_volumes[j]) for j in range(M_off)]

    # Feasibility mask for OFFLINE items: shape (M_off, N+1)
    # Last column (fallback) is always feasible if fallback enabled; else 0.
    p_off = cfg.graphs.p_off
    feas_mask = (rng.uniform(size=(M_off, N)) < p_off).astype(int)
    if cfg.problem.fallback_is_enabled:
        fallback_col = np.ones((M_off, 1), dtype=int)
    else:
        fallback_col = np.zeros((M_off, 1), dtype=int)
    feas_full = np.hstack([feas_mask, fallback_col])  # (M_off, N+1)
    
    validate_mask(feas_full)
    
    # Assignment costs for OFFLINE items to all N+1 bins (fallback very large)
    base_costs = _sample_assignment_costs(cfg, rng, M_off, N)
    fallback_costs = np.full((M_off, 1), cfg.costs.huge_fallback, dtype=float)
    assign = np.hstack([base_costs, fallback_costs])

    # Infeasible edges get zeroed in feasibility mask (solver will enforce x[j,i]=0 later).
    costs = Costs(
        assign=assign,
        reassignment_penalty=cfg.costs.reassignment_penalty,
        penalty_mode=cfg.costs.penalty_mode,
        per_volume_scale=cfg.costs.per_volume_scale,
        huge_fallback=cfg.costs.huge_fallback,
    )

    feasible = FeasibleGraph(feasible=feas_full)
    return Instance(
        bins=bins,
        offline_items=offline_items,
        costs=costs,
        feasible=feasible,
        fallback_bin_index=fallback_idx,
    )

def _sample_online_volumes(cfg: Config, rng: np.random.Generator, count: int) -> np.ndarray:
    """
    Draw 'count' online item volumes using the configured beta distribution
    and absolute bounds.
    """
    dims = max(1, int(getattr(cfg.problem, "dimensions", 1)))
    beta_params = _normalize_beta_params(cfg.volumes.online_beta, dims)
    bounds = _normalize_bounds(cfg.volumes.online_bounds, dims)
    volumes = np.zeros((count, dims), dtype=float)
    for d in range(dims):
        alpha_vol_onl, beta_vol_onl = beta_params[d]
        lo, hi = bounds[d]
        assert alpha_vol_onl > 0 and beta_vol_onl > 0, "online_beta must be positive."
        assert 0.0 < lo < hi, "online_bounds must satisfy 0 < lower < upper."
        draws = rng.beta(alpha_vol_onl, beta_vol_onl, size=count).astype(float)
        volumes[:, d] = lo + draws * (hi - lo)
    return volumes

def _ensure_row_feasible(mask: np.ndarray, rng: np.random.Generator) -> None:
    """
    Ensure each row contains at least one feasible bin by activating a random bin if needed.
    """
    if mask.size == 0:
        return
    row_count, col_count = mask.shape
    for idx in range(row_count):
        if mask[idx].sum() == 0:
            j = int(rng.integers(0, col_count))
            mask[idx, j] = 1

def generate_online_sequence(
    cfg: Config,
    seed: int,
    *,
    start_id: Optional[int] = None,
    horizon: Optional[int] = None,
) -> Tuple[List[OnlineItem], FeasibleGraph, np.ndarray]:
    """
    Generate a sequence of online arrivals together with their feasibility mask.

    Returns
    -------
    online_items: list of OnlineItem descriptors (ids start at M_off unless overridden)
    online_feasible: FeasibleGraph with shape (M_on, N+1) and fallback column fixed to 0
    """
    if cfg.stoch.horizon_dist != "fixed":
        raise NotImplementedError("Only fixed horizon is currently supported.")

    M_on = horizon if horizon is not None else cfg.stoch.horizon
    rng = make_rng(seed)
    N = cfg.problem.N
    base_mask = (rng.uniform(size=(M_on, N)) < cfg.graphs.p_onl).astype(int)
    _ensure_row_feasible(base_mask, rng)

    fallback_col = np.zeros((M_on, 1), dtype=int)
    feas_full = np.hstack([base_mask, fallback_col])
    validate_mask(feas_full)

    volumes = _sample_online_volumes(cfg, rng, M_on) if M_on > 0 else np.empty((0, max(1, int(getattr(cfg.problem, "dimensions", 1)))), dtype=float)
    first_id = cfg.problem.M_off if start_id is None else start_id

    online_items: List[OnlineItem] = []
    if M_on > 0:
        base_costs = _sample_assignment_costs(cfg, rng, M_on, N)
        fallback_costs = np.full((M_on, 1), cfg.costs.huge_fallback, dtype=float)
        assign = np.hstack([base_costs, fallback_costs])
    else:
        assign = np.empty((0, N + 1), dtype=float)

    for offset in range(M_on):
        feasible_bins = [int(bin_id) for bin_id in np.flatnonzero(base_mask[offset])]
        online_items.append(
            OnlineItem(
                id=first_id + offset,
                volume=volumes[offset],
                feasible_bins=feasible_bins,
            )
        )

    return online_items, FeasibleGraph(feasible=feas_full), assign

def generate_instance_with_online(
    cfg: Config,
    seed: int,
    *,
    online_seed: Optional[int] = None,
    horizon: Optional[int] = None,
) -> Instance:
    """
    Convenience wrapper that augments an offline instance with generated online arrivals.
    """
    inst = generate_offline_instance(cfg, seed)
    M_off = cfg.problem.M_off
 
    on_seed = seed if online_seed is None else online_seed
    online_items, online_feasible, online_costs = generate_online_sequence(
        cfg,
        on_seed,
        start_id=len(inst.offline_items),
        horizon=horizon,
    )
    inst.online_items = online_items
    inst.online_feasible = online_feasible
    if online_costs.size:
        inst.costs.assign = np.vstack([inst.costs.assign, online_costs])
    return inst

def sample_online_item(
    cfg: Config,
    next_id: int,
    rng: np.random.Generator,
    capacities: np.ndarray,
) -> Tuple[int, np.ndarray, List[int]]:
    """
    Generate one ONLINE item:
    - volume ~ Beta(alpha, beta) scaled to the absolute 'online_bounds'
    - feasible bins via Erdos-Renyi with p_onl (NO fallback)
    """
    # capacities retained in signature for compatibility; volumes are independent of it now.
    dims = max(1, int(getattr(cfg.problem, "dimensions", 1)))
    beta_params = _normalize_beta_params(cfg.volumes.online_beta, dims)
    bounds = _normalize_bounds(cfg.volumes.online_bounds, dims)
    vol = np.zeros(dims, dtype=float)
    for d in range(dims):
        alpha, beta = beta_params[d]
        lo, hi = bounds[d]
        assert alpha > 0 and beta > 0, "online_beta must be positive."
        assert 0.0 < lo < hi, "online_bounds must satisfy 0 < lower < upper."
        vol[d] = float(lo + rng.beta(alpha, beta) * (hi - lo))

    N = cfg.problem.N
    p_onl = cfg.graphs.p_onl
    feas = (rng.uniform(size=N) < p_onl)
    feasible_bins = [i for i in range(N) if feas[i]]

    # Guarantee at least one feasible bin to avoid degenerate arrivals
    # Note: this is about the FEASIBILITY GRAPH (compatibility), not capacity.
    # Capacity may still be insufficient; the online policy can evict offline items to free space.
    if not feasible_bins:
        feasible_bins = [int(rng.integers(0, N))]

    return next_id, vol, feasible_bins
