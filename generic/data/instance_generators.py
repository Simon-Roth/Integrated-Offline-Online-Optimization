# generic/data/instance_generators.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from generic.core.config import Config
from generic.core.models import Costs, Instance, StepSpec
from generic.core.utils import make_rng
from generic.data.generator_utils import (
    _as_length_vector,
    _normalize_beta_params,
    _normalize_bounds,
    _cap_params_for_phase,
    _coerce_b,
    _ensure_row_feasible,
    _build_feas_constraints,
    _sample_feas_mask_by_option,
    _sample_assignment_costs,
)


class BaseInstanceGenerator(ABC):
    """
    Base interface for instance generators.
    """

    @abstractmethod
    def generate_full_instance(
        self,
        cfg: Config,
        seed: int,
        *,
        online_seed: Optional[int] = None,
        T_onl: Optional[int] = None,
    ) -> Instance:
        """
        Generate a full instance (offline + online) in one pass.
        """

    def generate_offline_instance(self, cfg: Config, seed: int) -> Instance:
        """
        Convenience wrapper for an offline-only instance (T_onl = 0).
        """
        return self.generate_full_instance(cfg, seed, T_onl=0)

    @abstractmethod
    def resample_online_phase(
        self,
        cfg: Config,
        instance: Instance,
        *,
        seed: Optional[int] = None,
        T_onl: Optional[int] = None,
    ) -> Instance:
        """
        Return a copy of `instance` with the same offline part but freshly sampled online steps (e.g., used for pricing)
        """
        ...

    @abstractmethod
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
        """
        Sample capacity matrices for the given phase.
        """
        ...

    @classmethod # makes it easy to use the correct way of sampling by just specifying the config (e.g., bp needs different sampling than generic due to block-structure of matrices)
    def from_config(cls, cfg: Config) -> "BaseInstanceGenerator":
        name = getattr(getattr(cfg, "generation", None), "generator", "generic")
        if name == "generic":
            return GenericInstanceGenerator()
        if name == "bgap":
            from bgap.data.instance_generators import BGAPInstanceGenerator

            return BGAPInstanceGenerator()
        raise ValueError(f"Unknown generator '{name}'. Expected 'generic' or 'bgap'.")


def _sample_cap_matrices_generic(
    rng: np.random.Generator,
    count: int,
    m: int,
    n: int,
    *,
    beta_params: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """
    Sample full cap matrices A_t^{cap} with shape (m, n) per step.
    """
    mats = np.zeros((count, m, n), dtype=float)
    for k in range(m):
        alpha, beta = beta_params[k]
        lo, hi = bounds[k]
        assert alpha > 0 and beta > 0, "beta parameters must be positive."
        assert 0.0 < lo < hi, "bounds must satisfy 0 < lower < upper."
        draws = rng.beta(alpha, beta, size=(count, n)).astype(float)
        mats[:, k, :] = lo + draws * (hi - lo)
    return mats


def sample_cap_matrices(
    cfg: Config,
    rng: np.random.Generator,
    count: int,
    n: int,
    m: int,
    *,
    phase: str,
) -> np.ndarray:
    """
    Sample capacity matrices for the given phase.
    Generic generator -> full m x n.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if count <= 0:
        return np.empty((0, m, n), dtype=float)
    if m <= 0 or n <= 0:
        return np.zeros((count, m, n), dtype=float)

    beta, bounds = _cap_params_for_phase(cfg, phase)
    beta_params = _normalize_beta_params(beta, m)
    bounds = _normalize_bounds(bounds, m)
    return _sample_cap_matrices_generic(
        rng,
        count,
        m,
        n,
        beta_params=beta_params,
        bounds=bounds,
    )


def _generate_online_phase(
    cfg: Config,
    rng_onl: np.random.Generator,
    T_onl: int,
    n: int,
    m: int,
    T_off: int,
) -> tuple[List[StepSpec], np.ndarray]:
    base_mask_onl = _sample_feas_mask_by_option(cfg, rng_onl, T_onl, n, phase="online")
    _ensure_row_feasible(base_mask_onl, rng_onl)

    fallback_idx = n if cfg.problem.fallback_is_enabled else -1
    cols = n + (1 if fallback_idx >= 0 else 0)

    cap_matrices_onl = sample_cap_matrices(cfg, rng_onl, T_onl, n, m, phase="online")

    online_steps: List[StepSpec] = []
    for offset in range(T_onl):
        cap_matrix = cap_matrices_onl[offset]
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


class GenericInstanceGenerator(BaseInstanceGenerator):
    """
    Generator for the generic problem family (full m x n capacities).
    """

    def generate_full_instance(
        self,
        cfg: Config,
        seed: int,
        *,
        online_seed: Optional[int] = None,
        T_onl: Optional[int] = None,
    ) -> Instance:
        if cfg.stoch.horizon_dist != "fixed":
            raise NotImplementedError("Only fixed horizon is currently supported.")

        rng = make_rng(seed)
        n = cfg.problem.n
        m = cfg.problem.m
        T_off = cfg.problem.T_off

        b = _coerce_b(cfg, rng, m)

        cap_matrices_off = sample_cap_matrices(cfg, rng, T_off, n, m, phase="offline")
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
                    cap_matrix=cap_matrices_off[j],
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
            cfg, rng_onl, T_onl, n, m, len(inst.offline_steps)
        )

        inst.online_steps = online_steps
        inst.costs.assignment_costs = np.vstack([inst.costs.assignment_costs, assign_onl])
        return inst

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
        return sample_cap_matrices(
            cfg,
            rng,
            count,
            n,
            m,
            phase=phase,
        )

    def resample_online_phase(
        self,
        cfg: Config,
        instance: Instance,
        *,
        seed: Optional[int] = None,
        T_onl: Optional[int] = None,
    ) -> Instance:
        T_off = len(instance.offline_steps)
        n = instance.n
        m = instance.m
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
            cfg, rng_onl, T_onl, n, m, T_off
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
