from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from generic.core.config import Config


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
# Helpers for clean scenarios
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

TOTAL_ITEMS = 200   

# Keep bounds fixed in most families for clean attribution
DEFAULT_BOUNDS = (30, 180)

# Usage variance levels (same mean, different variance)
VOL_LOW_VAR   = (9.0, 21.0)   # kappa=30 (low variance)
VOL_MID_VAR   = (3.0, 7.0)   # kappa=20 (medium variance)
VOL_HIGH_VAR_UNIFORM  = (0.3, 0.3)    # very high variance (U-shaped)

# Uniform baseline on [0,1] is Beta(1,1)
VOL_UNIFORM_01 = (1.0, 1.0)

# Feasibility variants (applied to all scenario families)
FEAS_VARIANTS = {
    "uniform": {"p_off": 0.5, "p_onl": 0.5},
    "expbin_a2": {"p_min": 0.3, "p_max": 0.9, "alpha_off": 2.0, "alpha_on": 2.0},
    #"expbin_a3": {"p_min": 0.1, "p_max": 0.6, "alpha_off": 3.0, "alpha_on": 3.0},
}

# Which feasibility variants to generate for each scenario
FEAS_VARIANTS_ACTIVE = ["uniform", "expbin_a2"]


# Ratio sweep
RATIO_SWEEP = [
    ("off0_on100",    0),
    #("off10_on90",   30),
    ("off20_on80",   40),
    #("off30_on70",   90),
    ("off40_on60",  80),
    #("off50_on50",  150),
    ("off60_on40",  120),
    #("off70_on30",  210),
    ("off80_on20",  160),
    #("off90_on10",  270),
    #("off100_on0",  300),
]


def ratio_overrides(T_off: int) -> Dict[str, Any]:
    T_onl = TOTAL_ITEMS - T_off
    return {"problem": {"T_off": T_off}, "stoch": {"T_onl": T_onl}}


def base_cost_graph_overrides() -> Dict[str, Any]:
    return {
        "costs": {
            "assign_beta": [2.0, 5.0],
            "assign_bounds": [1.0, 5.0],
            "huge_fallback": 50.0,
            "reassignment_penalty": 10.0,
            "penalty_mode": "per_item",
            "per_usage_scale": 10.0,
        },
    }




# -----------------------------
# Feasibility mode helpers
# -----------------------------


def _feas_variant_overrides(tag: str, extra: dict | None = None) -> Dict[str, Any]:
    if tag not in FEAS_VARIANTS:
        raise KeyError(f"Unknown feasibility variant '{tag}'")
    cfg = FEAS_VARIANTS[tag]
    if tag == "uniform":
        feas = {"mode": "uniform", "p_off": cfg["p_off"], "p_onl": cfg["p_onl"]}
        if extra:
            feas.update({k: v for k, v in extra.items() if k in {"p_off", "p_onl"}})
    else:
        feas = {
            "mode": "exp_bin",
            "exp_bin_offline": {"p_min": cfg["p_min"], "p_max": cfg["p_max"], "alpha": cfg["alpha_off"]},
            "exp_bin_online": {"p_min": cfg["p_min"], "p_max": cfg["p_max"], "alpha": cfg["alpha_on"]},
        }
        if extra:
            # Allow optional override of exp-bin params (if provided).
            if "exp_bin_offline" in extra:
                feas["exp_bin_offline"].update(extra["exp_bin_offline"])
            if "exp_bin_online" in extra:
                feas["exp_bin_online"].update(extra["exp_bin_online"])
    return {"feasibility": feas}


def add_scenario_with_feas_variants(
    *,
    name: str,
    overrides: Dict[str, Any],
    description: str,
    feasibility_extra: dict | None = None,
) -> None:
    for tag in FEAS_VARIANTS_ACTIVE:
        SCENARIO_SWEEP.append(
            ScenarioConfig(
                name=f"{name}_{tag}",
                overrides={
                    **overrides,
                    **_feas_variant_overrides(tag, feasibility_extra),
                },
                description=f"{description} (feasibility={tag})",
            )
        )



def volume_overrides(alpha_beta: Tuple[float, float], bounds: Tuple[float, float]) -> Dict[str, Any]:
    a, b = alpha_beta
    lo, hi = bounds
    return {
        "cap_coeffs": {
            "offline_beta": [a, b],
            "offline_bounds": [lo, hi],
            "online_beta": [a, b],
            "online_bounds": [lo, hi],
        }
    }


SCENARIO_SWEEP: List[ScenarioConfig] = []


# ========= FAMILY 1: BASELINE (mid variance), ratio sweep =========
for suffix, T_off in RATIO_SWEEP:
    add_scenario_with_feas_variants(
        name=f"baseline_midvar_{suffix}",
        overrides={
            **ratio_overrides(T_off),
            **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
            **base_cost_graph_overrides(),
        },
        description="Baseline: bounded coefficients with fixed mean and medium variance; ratio sweep.",
    )

# ========= FAMILY 2: COEFF VARIANCE (ceteris paribus mean & bounds), fixed ratio (60/40) =========
# Baseline is VOL_MID_VAR = Beta(3,7) on DEFAULT_BOUNDS.
add_scenario_with_feas_variants(
    name="vol_midvar_off60_on40",
    overrides={
        **ratio_overrides(120),
        **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
        **base_cost_graph_overrides(),
    },
    description="Coeff dispersion baseline (mid variance): Beta(3,7) on [30,180].",
)

add_scenario_with_feas_variants(
    name="vol_lowvar_off60_on40",
    overrides={
        **ratio_overrides(120),
        **volume_overrides(VOL_LOW_VAR, DEFAULT_BOUNDS),
        **base_cost_graph_overrides(),
    },
    description="Lower coeff variance than baseline (same mean): e.g., Beta(6,14) on [30,180].",
)

# High-uncertainty control: U-shaped Beta(0.3,0.3) but keep global load comparable by adjusting b_mean
baseline_Ev = mean_scaled_beta(DEFAULT_BOUNDS, VOL_MID_VAR)
uniform_Ev  = mean_scaled_beta(DEFAULT_BOUNDS, VOL_HIGH_VAR_UNIFORM)
baseline_cap_mean = 1680.0
uniform_cap_mean = baseline_cap_mean * (uniform_Ev / baseline_Ev)

uniform_base = ratio_overrides(120)
uniform_base["problem"]["b_mean"] = float(uniform_cap_mean)
add_scenario_with_feas_variants(
    name="vol_highuncert_uniform_same_load_off60_on40",
    overrides={
        **uniform_base,
        **volume_overrides(VOL_HIGH_VAR_UNIFORM, DEFAULT_BOUNDS),
        **base_cost_graph_overrides(),
    },
    description="High-uncertainty control: U-shaped Beta(0.3,0.3) coeffs with adjusted b_mean to match baseline load.",
)

# ========= FAMILY 3: GRAPH SPARSITY (only p_onl changes), fixed ratio (60/40) =========
for tag, p_onl in [("dense", 0.8), ("mid", 0.5), ("sparse", 0.2)]:
    add_scenario_with_feas_variants(
        name=f"graph_{tag}_off60_on40",
        overrides={
            **ratio_overrides(120),
            **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
            **base_cost_graph_overrides(),
        },
        description="Graph feasibility test (online sparsity) at fixed coeffs/costs/load.",
        feasibility_extra={"p_onl": p_onl, "p_off": 0.5},
    )

# ========= FAMILY 3A: GRAPH SPARSITY (mid only), fixed ratio (60/40) =========
add_scenario_with_feas_variants(
    name="graph_mid_off60_on40",
    overrides={
        **ratio_overrides(120),
        **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
        **base_cost_graph_overrides(),
    },
    description="Graph feasibility test (mid sparsity) at fixed coeffs/costs/load.",
    feasibility_extra={"p_onl": 0.5, "p_off": 0.5},
)

# ========= FAMILY 3B: GRAPH SPARSITY (only p_onl changes), purely online (0/100) =========
for tag, p_onl in [("dense", 0.8), ("mid", 0.5), ("sparse", 0.2)]:
    add_scenario_with_feas_variants(
        name=f"graph_{tag}_off0_on100",
        overrides={
            **ratio_overrides(0),
            **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
            **base_cost_graph_overrides(),
            "problem": {"fallback_allowed_online": False},
        },
        description="Graph feasibility test (online sparsity) for purely online setting without fallback for online.",
        feasibility_extra={"p_onl": p_onl, "p_off": 0.5},
    )

# ========= FAMILY 3C: PURELY ONLINE (0/400), horizon stress test =========
_online_h400_base = ratio_overrides(0)
_online_h400_base["problem"]["b_mean"] = 3360.0
_online_h400_base["problem"]["b_std"] = 40.0
add_scenario_with_feas_variants(
    name="online_h400_off0_on400",
    overrides={
        **_online_h400_base,
        **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
        **base_cost_graph_overrides(),
        "stoch": {"T_onl": 400},
    },
    description="Purely online horizon stress test: T_off=0, T_onl=400, mid-variance baseline.",
)

# ========= FAMILY 3D: FEASIBILITY SPARSITY w/ RESHUFFLING (20/80, 60/40) =========
for ratio_tag, T_off in [("off20_on80", 40), ("off60_on40", 120)]:
    base = ratio_overrides(T_off)
    base["problem"]["allow_reassignment"] = True
    base["problem"]["fallback_allowed_online"] = False
    for sparsity_tag, p_onl in [("dense", 0.8), ("mid", 0.5), ("sparse", 0.2)]:
        add_scenario_with_feas_variants(
            name=f"graph_{sparsity_tag}_{ratio_tag}_reshuffle",
            overrides={
                **base,
                **volume_overrides(VOL_MID_VAR, DEFAULT_BOUNDS),
                **base_cost_graph_overrides(),
            },
            description=(
                "Graph feasibility sparsity with reshuffling allowed "
                f"({ratio_tag}, p_onl={p_onl})."
            ),
            feasibility_extra={"p_onl": p_onl, "p_off": 0.5},
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
