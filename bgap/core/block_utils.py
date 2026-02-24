from __future__ import annotations

import numpy as np


def block_dim(n: int, m: int) -> int:
    """Return block dimension d for bgap (m = n * d)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if m % n != 0:
        raise ValueError(f"Expected m % n == 0 for block structure (m={m}, n={n}).")
    return m // n


def split_capacities(b: np.ndarray, n: int) -> np.ndarray:
    """Reshape global capacity vector b into per-action blocks (n x d)."""
    b_vec = np.asarray(b, dtype=float).reshape(-1)
    d = block_dim(n, b_vec.shape[0])
    return b_vec.reshape((n, d))


def extract_volume(cap_matrix: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Recover the size vector (length d) from a block-structured A_t^{cap}.
    Uses the first action's block.
    """
    d = block_dim(n, m)
    cap = np.asarray(cap_matrix, dtype=float)
    return cap[:d, 0].copy()
