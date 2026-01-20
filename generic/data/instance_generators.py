# generic/data/instance_generators.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from generic.config import Config
from generic.models import Costs, FeasibleGraph, Instance, ItemSpec, OnlineItem
from generic.general_utils import (
    make_rng,
    validate_capacities,
    validate_mask,
)


def _as_length_vector(value: float | Sequence[float], length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(length, float(arr[0]))
    if arr.size != length:
        raise ValueError(f"Expected {length} values, got {arr.size}.")
    return arr


def _normalize_beta_params(beta, length: int) -> np.ndarray:
    """
    Return array of shape (length, 2) with alpha/beta per component.
    """
    arr = np.asarray(beta, dtype=float)
    if arr.shape == (2,):
        return np.tile(arr, (length, 1))
    if arr.shape == (length, 2):
        return arr
    raise ValueError("beta parameters must be length-2 or shape (length, 2).")


def _normalize_bounds(bounds, length: int) -> np.ndarray:
    """
    Return array of shape (length, 2) with lower/upper bounds per component.
    """
    arr = np.asarray(bounds, dtype=float)
    if arr.shape == (2,):
        return np.tile(arr, (length, 1))
    if arr.shape == (length, 2):
        return arr
    raise ValueError("bounds must be length-2 or shape (length, 2).")


def _coerce_b(cfg: Config, rng: np.random.Generator, m: int) -> np.ndarray:
    """
    Build capacity vector b (length m). If cfg.problem.b is shorter than m,
    fill the remainder by sampling from b_mean/b_std.
    """
    base_b = np.asarray(cfg.problem.b, dtype=float).reshape(-1) if cfg.problem.b else np.array([], dtype=float)
    if base_b.size > m:
        base_b = base_b[:m]

    b_full = np.zeros((m,), dtype=float)
    if base_b.size:
        b_full[: base_b.size] = base_b
    if base_b.size < m:
        mean = _as_length_vector(cfg.problem.b_mean, m)
        std = _as_length_vector(cfg.problem.b_std, m)
        start = base_b.size
        extra = m - start
        sampled = rng.normal(
            loc=mean[start:],
            scale=np.maximum(std[start:], 0.0),
            size=(extra,),
        )
        if np.all(std[start:] == 0.0):
            sampled = mean[start:]
        sampled = np.clip(sampled, 1e-6, None)
        b_full[start:] = sampled
    validate_capacities(b_full)
    return b_full


def _require_block_structure(n: int, m: int) -> int:
    """
    As this is for binpacking problems, for now we use the binpacking block structure: m = n * d.
    Returns d and raises if m is not a multiple of n.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if m % n != 0:
        raise ValueError(f"Block-structured A_cap requires m % n == 0 (got m={m}, n={n}).")
    return m // n


def _sample_cap_vectors(
    cfg: Config,
    rng: np.random.Generator,
    count: int,
    *,
    beta_params: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """
    Sample coefficient vectors (length d) used to build block-structured A_t^{cap}.
    """
    d = beta_params.shape[0]
    samples = np.zeros((count, d), dtype=float)
    for k in range(d):
        alpha, beta = beta_params[k]
        lo, hi = bounds[k]
        assert alpha > 0 and beta > 0, "beta parameters must be positive."
        assert 0.0 < lo < hi, "bounds must satisfy 0 < lower < upper."
        draws = rng.beta(alpha, beta, size=count).astype(float)
        samples[:, k] = lo + draws * (hi - lo)
    return samples


def _build_block_cap_matrix(size_vec: np.ndarray, n: int, d: int) -> np.ndarray:
    """
    Build A_t^{cap} for the binpacking block structure (m = n * d).
    Each action i consumes size_vec in its own resource block.
    """
    m = n * d
    mat = np.zeros((m, n), dtype=float)
    for i in range(n):
        mat[i * d : (i + 1) * d, i] = size_vec
    return mat


def _ensure_row_feasible(mask: np.ndarray, rng: np.random.Generator) -> None:
    """
    Ensure each row contains at least one feasible action by activating a random action if needed.
    """
    if mask.size == 0:
        return
    row_count, col_count = mask.shape
    for idx in range(row_count):
        if mask[idx].sum() == 0:
            j = int(rng.integers(0, col_count))
            mask[idx, j] = 1


def generate_offline_instance(cfg: Config, seed: int) -> Instance:
    """
    Convenience wrapper for an offline-only instance (M_onl = 0).
    """
    return generate_instance_with_online(cfg, seed, M_onl=0)


def _generate_online_phase(
    cfg: Config,
    rng_onl: np.random.Generator,
    M_onl: int,
    n: int,
    d: int,
    M_off: int,
) -> tuple[List[OnlineItem], np.ndarray, np.ndarray]:
    base_mask_onl = (rng_onl.uniform(size=(M_onl, n)) < cfg.feasibility.p_onl).astype(int)
    _ensure_row_feasible(base_mask_onl, rng_onl)

    if cfg.problem.fallback_is_enabled:
        fallback_val = 1 if cfg.problem.fallback_allowed_online else 0
        fallback_col = np.full((M_onl, 1), fallback_val, dtype=int)
        feas_onl = np.hstack([base_mask_onl, fallback_col])
    else:
        feas_onl = base_mask_onl
    validate_mask(feas_onl)

    beta_params = _normalize_beta_params(cfg.cap_coeffs.online_beta, d)
    bounds = _normalize_bounds(cfg.cap_coeffs.online_bounds, d)
    cap_vectors_onl = _sample_cap_vectors(cfg, rng_onl, M_onl, beta_params=beta_params, bounds=bounds)

    online_items: List[OnlineItem] = []
    for offset in range(M_onl):
        cap_matrix = _build_block_cap_matrix(cap_vectors_onl[offset], n, d)
        feasible_actions = [int(a_id) for a_id in np.flatnonzero(base_mask_onl[offset])]
        online_items.append(
            OnlineItem(
                id=M_off + offset,
                cap_matrix=cap_matrix,
                feasible_actions=feasible_actions,
            )
        )

    base_costs_onl = _sample_assignment_costs(cfg, rng_onl, M_onl, n)
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
    n = cfg.problem.n
    m = cfg.problem.m
    d = _require_block_structure(n, m)
    M_off = cfg.problem.M_off

    b = _coerce_b(cfg, rng, m)

    beta_params = _normalize_beta_params(cfg.cap_coeffs.offline_beta, d)
    bounds = _normalize_bounds(cfg.cap_coeffs.offline_bounds, d)
    cap_vectors_off = _sample_cap_vectors(cfg, rng, M_off, beta_params=beta_params, bounds=bounds)
    offline_items = [
        ItemSpec(id=j, cap_matrix=_build_block_cap_matrix(cap_vectors_off[j], n, d))
        for j in range(M_off)
    ]

    # Feasibility mask for OFFLINE items: shape (M_off, n) or (M_off, n+1)
    p_off = cfg.feasibility.p_off
    feas_mask = (rng.uniform(size=(M_off, n)) < p_off).astype(int)
    if cfg.problem.fallback_is_enabled:
        fallback_val = 1 if cfg.problem.fallback_allowed_offline else 0
        fallback_col = np.full((M_off, 1), fallback_val, dtype=int)
        feas_full = np.hstack([feas_mask, fallback_col])  # (M_off, n+1)
    else:
        feas_full = feas_mask  # (M_off, n)
    validate_mask(feas_full)

    # Assignment costs for OFFLINE items to all actions
    base_costs = _sample_assignment_costs(cfg, rng, M_off, n)
    if cfg.problem.fallback_is_enabled:
        fallback_costs = np.full((M_off, 1), cfg.costs.huge_fallback, dtype=float)
        assign = np.hstack([base_costs, fallback_costs])
    else:
        assign = base_costs

    costs = Costs(
        assignment_costs=assign,
        reassignment_penalty=cfg.costs.reassignment_penalty,
        penalty_mode=cfg.costs.penalty_mode,
        per_usage_scale=cfg.costs.per_usage_scale,
        huge_fallback=cfg.costs.huge_fallback,
    )
    feasible = FeasibleGraph(feasible=feas_full)

    fallback_idx = n if cfg.problem.fallback_is_enabled else -1
    inst = Instance(
        n=n,
        m=m,
        b=b,
        offline_items=offline_items,
        costs=costs,
        offline_feasible=feasible,
        fallback_action_index=fallback_idx,
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
        cfg, rng_onl, M_onl, n, d, len(inst.offline_items)
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
    n = instance.n
    m = instance.m
    d = _require_block_structure(n, m)
    M_onl = len(instance.online_items) if M_onl is None else int(M_onl)

    if M_onl <= 0:
        offline_costs = np.asarray(
            instance.costs.assignment_costs[:M_off, :], dtype=float
        )
        costs = Costs(
            assignment_costs=offline_costs,
            reassignment_penalty=instance.costs.reassignment_penalty,
            penalty_mode=instance.costs.penalty_mode,
            per_usage_scale=instance.costs.per_usage_scale,
            huge_fallback=instance.costs.huge_fallback,
        )
        return Instance(
            n=instance.n,
            m=instance.m,
            b=instance.b.copy(),
            offline_items=instance.offline_items,
            costs=costs,
            offline_feasible=instance.offline_feasible,
            fallback_action_index=instance.fallback_action_index,
            online_items=[],
            online_feasible=None,
        )

    rng_onl = make_rng(seed)
    online_items, feas_onl, assign_onl = _generate_online_phase(
        cfg, rng_onl, M_onl, n, d, M_off
    )

    offline_costs = np.asarray(instance.costs.assignment_costs[:M_off, :], dtype=float)
    assign = np.vstack([offline_costs, assign_onl])
    costs = Costs(
        assignment_costs=assign,
        reassignment_penalty=instance.costs.reassignment_penalty,
        penalty_mode=instance.costs.penalty_mode,
        per_usage_scale=instance.costs.per_usage_scale,
        huge_fallback=instance.costs.huge_fallback,
    )
    return Instance(
        n=instance.n,
        m=instance.m,
        b=instance.b.copy(),
        offline_items=instance.offline_items,
        costs=costs,
        offline_feasible=instance.offline_feasible,
        fallback_action_index=instance.fallback_action_index,
        online_items=online_items,
        online_feasible=FeasibleGraph(feasible=feas_onl),
    )


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
