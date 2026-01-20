# generic/general_utils.py
from __future__ import annotations
from typing import Iterable, Sequence, Optional
import random
import numpy as np

def set_global_seed(seed: int) -> None:
    """
    Set seeds across Python's random and NumPy for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Return a modern NumPy RNG (PCG64). If seed=None, it is non-deterministic.
    """
    return np.random.default_rng(seed)

def safe_argmax(arr: np.ndarray) -> int:
    """
    Argmax that breaks ties by the first index deterministically.
    """
    return int(np.argmax(arr))

def validate_capacities(capacities: Sequence[float] | np.ndarray) -> None:
    arr = np.asarray(capacities, dtype=float)
    if arr.size == 0:
        return
    assert np.all(arr > 0), "All capacities must be positive."

def validate_mask(mask: np.ndarray) -> None:
    assert mask.dtype == np.int_ or mask.dtype == np.bool_, "Feasibility mask must be int/bool."
    if mask.size == 0:
        return
    assert mask.min() >= 0 and mask.max() <= 1, "Feasibility mask must be 0/1."

def effective_capacity(
    capacity: float | np.ndarray,
    enforce_slack: bool,
    slack_fraction: float,
) -> float | np.ndarray:
    """
    Apply slack if enabled. If enforce_slack=True, return (1 - slack_fraction) * capacity.
    Otherwise return capacity.
    """
    if enforce_slack:
        assert 0.0 <= slack_fraction < 1.0, "Slack fraction must be in [0,1)."
        return (1.0 - slack_fraction) * capacity
    return capacity


def scalarize_vector(vec: np.ndarray, mode: str) -> float:
    """
    Map a vector to a scalar for ordering/comparison.
    mode: "max" | "l1" | "l2"
    """
    arr = np.asarray(vec, dtype=float)
    mode = mode.lower()
    if mode == "max":
        return float(np.max(arr))
    if mode == "l1":
        return float(np.sum(arr))
    if mode == "l2":
        return float(np.linalg.norm(arr))
    raise ValueError(f"Unknown scalarization mode: {mode}")


def vector_fits(load: np.ndarray, usage: np.ndarray, capacity: np.ndarray, tol: float = 0.0) -> bool:
    """Return True if load + usage <= capacity (component-wise)."""
    return bool(np.all(load + usage <= capacity + tol))


def residual_vector(load: np.ndarray, usage: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Residual capacity after adding usage into load."""
    return capacity - (load + usage)


def usage_total(usage: np.ndarray) -> float:
    """Total usage for per-usage penalties (L1)."""
    return float(np.sum(np.asarray(usage, dtype=float)))
