from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from generic.core.config import Config
from generic.core.models import Instance
from generic.core.utils import effective_capacity, scalarize_vector
from bgap.core.block_utils import block_dim, extract_volume, split_capacities


def sorted_steps_by_volume(inst: Instance, size_key: str) -> List[Tuple[int, np.ndarray]]:
    """Return (step_idx, volume_vec) sorted by scalarized volume (descending)."""
    n = inst.n
    m = inst.m
    steps: List[Tuple[int, np.ndarray]] = [
        (idx, extract_volume(step.cap_matrix, n, m))
        for idx, step in enumerate(inst.offline_steps)
    ]
    return sorted(steps, key=lambda pair: scalarize_vector(pair[1], size_key), reverse=True)


def init_loads_and_caps(inst: Instance, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Initialize per-bin loads and effective capacities (with slack if enabled)."""
    n = inst.n
    d = block_dim(n, inst.m)
    loads = np.zeros((n, d))
    eff_caps = effective_capacity(
        split_capacities(inst.b, n),
        cfg.slack.enforce_slack,
        cfg.slack.fraction,
    )
    return loads, np.asarray(eff_caps, dtype=float)


def objective_with_fallback(assigned_option: Dict[int, int], inst: Instance) -> float:
    """Compute total assignment cost, charging huge_fallback for fallback options."""
    total_cost = 0.0
    fallback_idx = inst.fallback_option_index
    for step_idx, option_idx in assigned_option.items():
        if option_idx < fallback_idx or fallback_idx < 0:
            total_cost += float(inst.costs.assignment_costs[step_idx, option_idx])
        else:
            total_cost += inst.costs.huge_fallback
    return total_cost
