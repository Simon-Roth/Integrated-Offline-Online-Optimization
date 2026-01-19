# generic/data/instance_generators.py
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
from generic.config import Config
from generic.models import BinSpec, ItemSpec, Instance, Costs, FeasibleGraph, OnlineItem
from generic.general_utils import (
    validate_capacities,
    make_rng,
    validate_mask,
)




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
    Convenience wrapper for an offline-only instance (M_onl = 0).
    """
    return generate_instance_with_online(cfg, seed, M_onl=0)

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

def _generate_online_phase(
    cfg: Config,
    rng_onl: np.random.Generator,
    M_onl: int,
    N: int,
    M_off: int,
) -> tuple[List[OnlineItem], np.ndarray, np.ndarray]:
    base_mask_onl = (rng_onl.uniform(size=(M_onl, N)) < cfg.graphs.p_onl).astype(int)
    _ensure_row_feasible(base_mask_onl, rng_onl)

    if cfg.problem.fallback_is_enabled:
        fallback_val = 1 if cfg.problem.fallback_allowed_online else 0
        fallback_col = np.full((M_onl, 1), fallback_val, dtype=int)
        feas_onl = np.hstack([base_mask_onl, fallback_col])
    else:
        feas_onl = base_mask_onl
    validate_mask(feas_onl)

    volumes_onl = _sample_online_volumes(cfg, rng_onl, M_onl)
    online_items: List[OnlineItem] = []
    for offset in range(M_onl):
        feasible_bins = [int(bin_id) for bin_id in np.flatnonzero(base_mask_onl[offset])]
        online_items.append(
            OnlineItem(
                id=M_off + offset,
                volume=volumes_onl[offset],
                feasible_bins=feasible_bins,
            )
        )

    base_costs_onl = _sample_assignment_costs(cfg, rng_onl, M_onl, N)
    if cfg.problem.fallback_is_enabled:
        fallback_costs = np.full((M_onl, 1), cfg.costs.huge_fallback, dtype=float)
        assign_onl = np.hstack([base_costs_onl, fallback_costs])
    else:
        assign_onl = base_costs_onl

    return online_items, feas_onl, assign_onl

def generate_instance_with_online(
    cfg: Config,
    seed: int,
    *,
    online_seed: Optional[int] = None,
    M_onl: Optional[int] = None,
) -> Instance:
    """
    Generate a full instance (offline + online) in one pass.
    """
    if cfg.stoch.horizon_dist != "fixed":
        raise NotImplementedError("Only fixed horizon is currently supported.")

    rng = make_rng(seed)
    N = cfg.problem.N
    M_off = cfg.problem.M_off
    dims = max(1, int(getattr(cfg.problem, "dimensions", 1)))
    capacities = _coerce_capacities(cfg, rng, N, dims)

    # Bins (0..N-1). Fallback uses index N when enabled.
    bins = [BinSpec(id=i, capacity=np.array(capacities[i], dtype=float)) for i in range(N)]
    fallback_idx = N if cfg.problem.fallback_is_enabled else -1

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

    # Feasibility mask for OFFLINE items: shape (M_off, N) or (M_off, N+1)
    p_off = cfg.graphs.p_off
    feas_mask = (rng.uniform(size=(M_off, N)) < p_off).astype(int)
    if cfg.problem.fallback_is_enabled:
        fallback_val = 1 if cfg.problem.fallback_allowed_offline else 0
        fallback_col = np.full((M_off, 1), fallback_val, dtype=int)
        feas_full = np.hstack([feas_mask, fallback_col])  # (M_off, N+1)
    else:
        feas_full = feas_mask  # (M_off, N)
    validate_mask(feas_full)

    # Assignment costs for OFFLINE items to all bins
    base_costs = _sample_assignment_costs(cfg, rng, M_off, N)
    if cfg.problem.fallback_is_enabled:
        fallback_costs = np.full((M_off, 1), cfg.costs.huge_fallback, dtype=float)
        assign = np.hstack([base_costs, fallback_costs])
    else:
        assign = base_costs

    costs = Costs(
        assignment_costs=assign,
        reassignment_penalty=cfg.costs.reassignment_penalty,
        penalty_mode=cfg.costs.penalty_mode,
        per_volume_scale=cfg.costs.per_volume_scale,
        huge_fallback=cfg.costs.huge_fallback,
    )
    feasible = FeasibleGraph(feasible=feas_full)

    inst = Instance(
        bins=bins,
        offline_items=offline_items,
        costs=costs,
        offline_feasible=feasible,
        fallback_bin_index=fallback_idx,
    )

    # Online items (optional).
    M_onl = int(M_onl) if M_onl is not None else int(cfg.stoch.horizon)
    if M_onl <= 0:
        inst.online_items = []
        inst.online_feasible = None
        return inst

    on_seed = seed if online_seed is None else online_seed
    rng_onl = make_rng(on_seed)
    online_items, feas_onl, assign_onl = _generate_online_phase(
        cfg, rng_onl, M_onl, N, len(inst.offline_items)
    )

    inst.online_items = online_items
    inst.online_feasible = FeasibleGraph(feasible=feas_onl)
    inst.costs.assignment_costs = np.vstack([inst.costs.assignment_costs, assign_onl])
    return inst


def resample_online_items(
    cfg: Config,
    instance: Instance,
    *,
    seed: Optional[int] = None,
    M_onl: Optional[int] = None,
) -> Instance:
    """
    Return a copy of `instance` with the same offline part but freshly sampled online items.
    """
    M_off = len(instance.offline_items)
    N = len(instance.bins)
    M_onl = len(instance.online_items) if M_onl is None else int(M_onl)

    if M_onl <= 0:
        offline_costs = np.asarray(
            instance.costs.assignment_costs[:M_off, :], dtype=float
        )
        costs = Costs(
            assignment_costs=offline_costs,
            reassignment_penalty=instance.costs.reassignment_penalty,
            penalty_mode=instance.costs.penalty_mode,
            per_volume_scale=instance.costs.per_volume_scale,
            huge_fallback=instance.costs.huge_fallback,
        )
        return Instance(
            bins=instance.bins,
            offline_items=instance.offline_items,
            costs=costs,
            offline_feasible=instance.offline_feasible,
            fallback_bin_index=instance.fallback_bin_index,
            online_items=[],
            online_feasible=None,
        )

    rng_onl = make_rng(seed)
    online_items, feas_onl, assign_onl = _generate_online_phase(
        cfg, rng_onl, M_onl, N, M_off
    )

    offline_costs = np.asarray(instance.costs.assignment_costs[:M_off, :], dtype=float)
    assign = np.vstack([offline_costs, assign_onl])
    costs = Costs(
        assignment_costs=assign,
        reassignment_penalty=instance.costs.reassignment_penalty,
        penalty_mode=instance.costs.penalty_mode,
        per_volume_scale=instance.costs.per_volume_scale,
        huge_fallback=instance.costs.huge_fallback,
    )
    return Instance(
        bins=instance.bins,
        offline_items=instance.offline_items,
        costs=costs,
        offline_feasible=instance.offline_feasible,
        fallback_bin_index=instance.fallback_bin_index,
        online_items=online_items,
        online_feasible=FeasibleGraph(feasible=feas_onl),
    )
