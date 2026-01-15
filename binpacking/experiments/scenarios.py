from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from generic.config import Config


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    overrides: Dict[str, Any]
    seeds: Sequence[int] | None = None
    description: str | None = None


def apply_config_overrides(base_cfg: Config, overrides: Dict[str, Any]) -> Config:
    cfg = copy.deepcopy(base_cfg)

    def _apply(target, patch: Dict[str, Any]) -> None:
        for key, value in patch.items():
            if not hasattr(target, key):
                raise KeyError(f"Config object {target} has no attribute '{key}'")
            current = getattr(target, key)
            if isinstance(value, dict) and hasattr(current, "__dict__"):
                _apply(current, value)
            else:
                setattr(target, key, copy.deepcopy(value))

    _apply(cfg, overrides)
    return cfg


# -----------------------------
# Helpers for "clean" scenarios
# -----------------------------

def beta_from_mean_kappa(m: float, kappa: float) -> Tuple[float, float]:
    """
    Beta(α,β) with mean m and concentration κ = α+β.
    Var decreases as κ increases. Mean stays fixed.
    """
    if not (0.0 < m < 1.0):
        raise ValueError("m must be in (0,1)")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    return (m * kappa, (1.0 - m) * kappa)


def mean_scaled_beta(bounds: Tuple[float, float], alpha_beta: Tuple[float, float]) -> float:
    """
    If X ~ Beta(α,β) on [0,1], then V = lo + (hi-lo) X on [lo,hi].
    E[V] = lo + (hi-lo) * α/(α+β)
    """
    lo, hi = bounds
    a, b = alpha_beta
    m = a / (a + b)
    return lo + (hi - lo) * m


# -----------------------------
# Canonical scenario knobs
# -----------------------------

TOTAL_ITEMS = 300   # always keep M_off + M_onl = 300 for interpretability

# Keep bounds fixed in most families for clean attribution
DEFAULT_BOUNDS = (0.5, 1.7)

# Choose target mean volume via m=1/3 so that E[V]=0.9 on bounds [0.5,1.7]
# (shown in the "check" section below)
TARGET_M = 1.0 / 3.0

# Volume variance levels (same mean, different variance)
VOL_LOW_VAR   = beta_from_mean_kappa(TARGET_M, 60.0)  # (20,40)
VOL_MID_VAR   = beta_from_mean_kappa(TARGET_M, 15.0)  # (5,10)
VOL_HIGH_VAR  = beta_from_mean_kappa(TARGET_M, 6.0)   # (2,4)

# Uniform baseline on [0,1] is Beta(1,1)
VOL_UNIFORM_01 = (1.0, 1.0)

# Ratio sweep
RATIO_SWEEP = [
    ("off0_on100",    0),
    #("off10_on90",   30),
    ("off20_on80",   60),
    #("off30_on70",   90),
    ("off40_on60",  120),
    ("off50_on50",  150),
    ("off60_on40",  180),
    #("off70_on30",  210),
    ("off80_on20",  240),
    #("off90_on10",  270),
    #("off100_on0",  300),
]


def ratio_overrides(M_off: int) -> Dict[str, Any]:
    M_onl = TOTAL_ITEMS - M_off
    return {"problem": {"M_off": M_off}, "stoch": {"horizon": M_onl}}


def base_cost_graph_overrides() -> Dict[str, Any]:
    return {
        "graphs": {"p_off": 0.8, "p_onl": 0.5},
        "costs": {
            "assign_beta": [2.0, 5.0],
            "assign_bounds": [1.0, 5.0],
            "huge_fallback": 50.0,
            "reassignment_penalty": 10.0,
            "penalty_mode": "per_item",
            "per_volume_scale": 10.0,
        },
    }


def volume_overrides(alpha_beta: Tuple[float, float], bounds: Tuple[float, float]) -> Dict[str, Any]:
    a, b = alpha_beta
    lo, hi = bounds
    return {
        "volumes": {
            "offline_beta": [a, b],
            "offline_bounds": [lo, hi],
            "online_beta": [a, b],
            "online_bounds": [lo, hi],
        }
    }


SCENARIO_SWEEP: List[ScenarioConfig] = []

# ========= FAMILY 1: BASELINE (mid variance), ratio sweep =========
for suffix, M_off in RATIO_SWEEP:
    SCENARIO_SWEEP.append(
        ScenarioConfig(
            name=f"baseline_midvar_{suffix}",
            overrides={
                **ratio_overrides(M_off),
                **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
                **base_cost_graph_overrides(),
                # baseline load regime stays at capacity_mean=30 (default yaml)
            },
            description="Baseline: bounded volumes with fixed mean (E[V]=0.9) and medium variance; ratio sweep.",
        )
    )

# ========= FAMILY 2: VOLUME VARIANCE (ceteris paribus mean & bounds), fixed ratio (50/50) =========
for tag, ab in [("lowvar", VOL_LOW_VAR), ("highvar", VOL_HIGH_VAR)]:
    SCENARIO_SWEEP.append(
        ScenarioConfig(
            name=f"vol_{tag}_off50_on50",
            overrides={
                **ratio_overrides(150),
                **volume_overrides(ab, DEFAULT_BOUNDS),
                **base_cost_graph_overrides(),
            },
            description="Volume dispersion test at fixed mean/bounds (only variance changes).",
        )
    )

# ========= FAMILY 3: GRAPH SPARSITY (only p_onl changes), fixed ratio (50/50) =========
for tag, p_onl in [("dense", 0.8), ("sparse", 0.2)]:
    SCENARIO_SWEEP.append(
        ScenarioConfig(
            name=f"graph_{tag}_off50_on50",
            overrides={
                **ratio_overrides(150),
                **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
                **base_cost_graph_overrides(),
                "graphs": {"p_off": 0.8, "p_onl": p_onl},
            },
            description="Graph feasibility test (online sparsity) at fixed volumes/costs/load.",
        )
    )

# ========= FAMILY 4: LOAD REGIME (capacity_mean changes), fixed ratio (50/50) =========
for tag, cap_mean in [("underload", 40.0), ("overload", 18.0)]:
    base = ratio_overrides(150)
    # Preserve the ratio overrides (M_off + M_onl) while tweaking capacity_mean.
    base["problem"]["capacity_mean"] = cap_mean
    SCENARIO_SWEEP.append(
        ScenarioConfig(
            name=f"load_{tag}_off50_on50",
            overrides={
                **base,
                **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
                **base_cost_graph_overrides(),
            },
            description="Load regime test by varying capacity_mean only (volumes/costs fixed).",
        )
        
    )

# ========= FAMILY 5: UNIFORM SHAPE CHECK via Beta(1,1) but SAME GLOBAL LOAD =========
# Uniform on [0,1] has mean 0.5 -> mean volume changes. To keep rho comparable,
# we adjust capacity_mean to match the baseline rho.
baseline_Ev = mean_scaled_beta(DEFAULT_BOUNDS, VOL_MID_VAR)     # 0.9
uniform_Ev  = mean_scaled_beta(DEFAULT_BOUNDS, VOL_UNIFORM_01)  # 1.1
baseline_cap_mean = 30.0
uniform_cap_mean = baseline_cap_mean * (uniform_Ev / baseline_Ev)  # keeps rho fixed

SCENARIO_SWEEP.append(
    ScenarioConfig(
        name="uniform_beta11_same_load_off50_on50",
        overrides={
            **ratio_overrides(150),
            **volume_overrides(VOL_UNIFORM_01, DEFAULT_BOUNDS),
            **base_cost_graph_overrides(),
            "problem": {"capacity_mean": float(uniform_cap_mean)},
        },
        description="Uniform volumes (Beta(1,1)) with adjusted capacity_mean to keep global load comparable.",
    )
)


def select_scenarios(names: Iterable[str] | None) -> List[ScenarioConfig]:
    if names is None:
        return SCENARIO_SWEEP
    name_set = set(names)
    filtered = [s for s in SCENARIO_SWEEP if s.name in name_set]
    missing = name_set - {s.name for s in filtered}
    if missing:
        known = ", ".join(sorted(s.name for s in SCENARIO_SWEEP))
        raise KeyError(f"Unknown scenario(s): {missing}. Known: {known}")
    return filtered
