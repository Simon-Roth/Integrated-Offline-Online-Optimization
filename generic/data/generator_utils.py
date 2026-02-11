from __future__ import annotations

from typing import Sequence

import numpy as np

from generic.core.config import Config
from generic.core.utils import validate_capacities


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


def _cap_params_for_phase(cfg: Config, phase: str) -> tuple:
    if phase == "offline":
        return cfg.cap_coeffs.offline_beta, cfg.cap_coeffs.offline_bounds
    if phase == "online":
        return cfg.cap_coeffs.online_beta, cfg.cap_coeffs.online_bounds
    raise ValueError("phase must be 'offline' or 'online'.")


def _coerce_b(cfg: Config, rng: np.random.Generator, m: int) -> np.ndarray:
    """
    Build capacity vector b (length m). If cfg.problem.b is shorter than m,
    fill the remainder by sampling from b_mean/b_std.
    """
    base_b = (
        np.asarray(cfg.problem.b, dtype=float).reshape(-1)
        if cfg.problem.b
        else np.array([], dtype=float)
    )
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


def _ensure_row_feasible(mask: np.ndarray, rng: np.random.Generator) -> None:
    """
    Ensure each row contains at least one feasible option by activating a random option if needed.
    """
    if mask.size == 0:
        return
    row_count, col_count = mask.shape
    for idx in range(row_count):
        if mask[idx].sum() == 0:
            j = int(rng.integers(0, col_count))
            mask[idx, j] = 1


def _build_feas_constraints(
    mask_row: np.ndarray,
    *,
    fallback_allowed: bool,
    fallback_idx: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build A_t^{feas} and b_t from a 0/1 feasibility mask on regular options.
    Encodes one-hot selection and forbidden options as explicit equalities.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # One-hot constraint across all options (including fallback column if present).
    one_hot = np.ones((cols,), dtype=float)
    rows.append(one_hot)
    rhs.append(1.0)

    # Forbid regular options where mask_row[i] == 0 via x_i = 0.
    for idx in np.flatnonzero(mask_row == 0):
        row = np.zeros((cols,), dtype=float)
        row[int(idx)] = 1.0
        rows.append(row)
        rhs.append(0.0)

    # Forbid fallback if it exists but is not allowed for this phase.
    if fallback_idx >= 0 and not fallback_allowed:
        row = np.zeros((cols,), dtype=float)
        row[int(fallback_idx)] = 1.0
        rows.append(row)
        rhs.append(0.0)

    return np.vstack(rows), np.asarray(rhs, dtype=float)


def _option_probs_exp(n: int, p_min: float, p_max: float, alpha: float) -> np.ndarray:
    if n <= 1:
        return np.asarray([p_max], dtype=float)
    t = np.linspace(0.0, 1.0, n)
    probs = p_min + (p_max - p_min) * np.exp(-alpha * t)
    return np.clip(probs, 0.0, 1.0)


def _sample_feas_mask_by_option(
    cfg: Config, rng: np.random.Generator, M: int, n: int, *, phase: str
) -> np.ndarray:
    mode = getattr(cfg.feasibility, "mode", "uniform")
    if mode == "uniform":
        p = cfg.feasibility.p_off if phase == "offline" else cfg.feasibility.p_onl
        probs = np.full((n,), float(p), dtype=float)
    elif mode == "exp_bin":
        if phase == "offline":
            params = getattr(cfg.feasibility, "exp_bin_offline", None) or {}
        else:
            params = getattr(cfg.feasibility, "exp_bin_online", None) or {}
        p_min = float(params.get("p_min", 0.05))
        p_max = float(params.get("p_max", 0.8))
        alpha = float(params.get("alpha", 2.0))
        probs = _option_probs_exp(n, p_min, p_max, alpha)
    else:
        raise ValueError(f"Unknown feasibility mode '{mode}'.")

    # Broadcast per-option probabilities to steps.
    return (rng.uniform(size=(M, n)) < probs).astype(int)


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
