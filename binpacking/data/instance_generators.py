# binpacking/data/instance_generators.py
# Binpacking block-structured instance generator (m = n * d).
from __future__ import annotations

from typing import List, Optional

import numpy as np

from generic.core.config import Config
from generic.core.models import Costs, Instance, StepSpec
from generic.data.instance_generators import BaseInstanceGenerator
from generic.core.utils import make_rng
from generic.data.generator_utils import (
    _normalize_beta_params,
    _normalize_bounds,
    _cap_params_for_phase,
    _coerce_b,
    _ensure_row_feasible,
    _build_feas_constraints,
    _sample_feas_mask_by_option,
    _sample_assignment_costs,
)
from generic.data.offline_milp_assembly import build_offline_milp_data

def _require_block_structure(n: int, m: int) -> int:
    """
    Enforce the binpacking block structure: m = n * d.
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


def _generate_online_phase(
    cfg: Config,
    rng_onl: np.random.Generator,
    T_onl: int,
    n: int,
    d: int,
    T_off: int,
) -> tuple[List[StepSpec], np.ndarray]:
    base_mask_onl = _sample_feas_mask_by_option(cfg, rng_onl, T_onl, n, phase="online")
    _ensure_row_feasible(base_mask_onl, rng_onl)

    fallback_idx = n if cfg.problem.fallback_is_enabled else -1
    cols = n + (1 if fallback_idx >= 0 else 0)

    beta_params = _normalize_beta_params(cfg.cap_coeffs.online_beta, d)
    bounds = _normalize_bounds(cfg.cap_coeffs.online_bounds, d)
    cap_vectors_onl = _sample_cap_vectors(cfg, rng_onl, T_onl, beta_params=beta_params, bounds=bounds)

    online_steps: List[StepSpec] = []
    for offset in range(T_onl):
        cap_matrix = _build_block_cap_matrix(cap_vectors_onl[offset], n, d)
        feas_matrix, feas_rhs = _build_feas_constraints(
            base_mask_onl[offset],
            fallback_allowed=cfg.problem.fallback_allowed_online,
            fallback_idx=fallback_idx,
            cols=cols,
        )
        online_steps.append(
            StepSpec(
                step_id=T_off + offset,
                cap_matrix=cap_matrix,
                feas_matrix=feas_matrix,
                feas_rhs=feas_rhs,
            )
        )

    base_costs_onl = _sample_assignment_costs(cfg, rng_onl, T_onl, n)
    if cfg.problem.fallback_is_enabled:
        fallback_costs = np.full((T_onl, 1), cfg.costs.huge_fallback, dtype=float)
        assign_onl = np.hstack([base_costs_onl, fallback_costs])
    else:
        assign_onl = base_costs_onl

    return online_steps, assign_onl


def generate_full_instance(
    cfg: Config,
    seed: int,
    *,
    online_seed: Optional[int] = None,
    T_onl: Optional[int] = None,
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
    T_off = cfg.problem.T_off

    b = _coerce_b(cfg, rng, m)

    beta_params = _normalize_beta_params(cfg.cap_coeffs.offline_beta, d)
    bounds = _normalize_bounds(cfg.cap_coeffs.offline_bounds, d)
    cap_vectors_off = _sample_cap_vectors(cfg, rng, T_off, beta_params=beta_params, bounds=bounds)
    feas_mask = _sample_feas_mask_by_option(cfg, rng, T_off, n, phase="offline")
    _ensure_row_feasible(feas_mask, rng)
    fallback_idx = n if cfg.problem.fallback_is_enabled else -1
    cols = n + (1 if fallback_idx >= 0 else 0)
    offline_steps: List[StepSpec] = []
    for j in range(T_off):
        feas_matrix, feas_rhs = _build_feas_constraints(
            feas_mask[j],
            fallback_allowed=cfg.problem.fallback_allowed_offline,
            fallback_idx=fallback_idx,
            cols=cols,
        )
        offline_steps.append(
            StepSpec(
                step_id=j,
                cap_matrix=_build_block_cap_matrix(cap_vectors_off[j], n, d),
                feas_matrix=feas_matrix,
                feas_rhs=feas_rhs,
            )
        )

    base_costs = _sample_assignment_costs(cfg, rng, T_off, n)
    if cfg.problem.fallback_is_enabled:
        fallback_costs = np.full((T_off, 1), cfg.costs.huge_fallback, dtype=float)
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
    fallback_idx = n if cfg.problem.fallback_is_enabled else -1
    inst = Instance(
        n=n,
        m=m,
        b=b,
        offline_steps=offline_steps,
        costs=costs,
        fallback_option_index=fallback_idx,
    )

    T_onl = int(T_onl) if T_onl is not None else int(cfg.stoch.T_onl)
    if T_onl <= 0:
        inst.online_steps = []
        return inst

    on_seed = seed if online_seed is None else online_seed
    rng_onl = make_rng(on_seed)
    online_steps, assign_onl = _generate_online_phase(
        cfg, rng_onl, T_onl, n, d, len(inst.offline_steps)
    )

    inst.online_steps = online_steps
    inst.costs.assignment_costs = np.vstack([inst.costs.assignment_costs, assign_onl])
    return inst


def resample_online_phase(
    cfg: Config,
    instance: Instance,
    *,
    seed: Optional[int] = None,
    T_onl: Optional[int] = None,
) -> Instance:
    """
    Return a copy of `instance` with the same offline part but freshly sampled online items.
    """
    T_off = len(instance.offline_steps)
    n = instance.n
    m = instance.m
    d = _require_block_structure(n, m)
    T_onl = len(instance.online_steps) if T_onl is None else int(T_onl)

    if T_onl <= 0:
        offline_costs = np.asarray(
            instance.costs.assignment_costs[:T_off, :], dtype=float
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
            offline_steps=instance.offline_steps,
            costs=costs,
            fallback_option_index=instance.fallback_option_index,
            online_steps=[],
        )

    rng_onl = make_rng(seed)
    online_steps, assign_onl = _generate_online_phase(
        cfg, rng_onl, T_onl, n, d, T_off
    )

    offline_costs = np.asarray(instance.costs.assignment_costs[:T_off, :], dtype=float)
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
        offline_steps=instance.offline_steps,
        costs=costs,
        fallback_option_index=instance.fallback_option_index,
        online_steps=online_steps,
    )


class BinpackingInstanceGenerator(BaseInstanceGenerator):
    """
    Binpacking generator that enforces the block structure (m = n * d).
    """

    def generate_full_instance(
        self,
        cfg: Config,
        seed: int,
        *,
        online_seed: Optional[int] = None,
        T_onl: Optional[int] = None,
    ) -> Instance:
        return generate_full_instance(
            cfg,
            seed,
            online_seed=online_seed,
            T_onl=T_onl,
        )

    def resample_online_phase(
        self,
        cfg: Config,
        instance: Instance,
        *,
        seed: Optional[int] = None,
        T_onl: Optional[int] = None,
    ) -> Instance:
        return resample_online_phase(
            cfg,
            instance,
            seed=seed,
            T_onl=T_onl,
        )

    def sample_cap_matrices(
        self,
        cfg: Config,
        rng: np.random.Generator,
        count: int,
        n: int,
        m: int,
        *,
        phase: str,
    ) -> np.ndarray:
        beta, bounds = _cap_params_for_phase(cfg, phase)
        d = _require_block_structure(n, m)
        beta_params = _normalize_beta_params(beta, d)
        bounds = _normalize_bounds(bounds, d)
        vectors = _sample_cap_vectors(cfg, rng, count, beta_params=beta_params, bounds=bounds)
        mats = np.zeros((count, m, n), dtype=float)
        for idx in range(count):
            mats[idx] = _build_block_cap_matrix(vectors[idx], n, d)
        return mats


__all__ = [
    "build_offline_milp_data",
    "generate_full_instance",
    "resample_online_phase",
    "BinpackingInstanceGenerator",
]
