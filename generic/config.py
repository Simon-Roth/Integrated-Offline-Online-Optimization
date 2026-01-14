# generic/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import yaml
from pathlib import Path

# ---- Problem & generation knobs ----

@dataclass
class ProblemConfig:
    """
    Core structural parameters for an instance.
    - N: number of regular bins
    - M_off: number of offline items
    - dimensions: number of dimensions for capacities/volumes
    - capacities: list of bin capacities (len == N) (optional if using distribution)
    - capacity_mean/std: parameters to synthesize capacities when list shorter than N
    - fallback_is_enabled: if False, no fallback bin exists in the instance
    - binpacking: enable binpacking-specific logic (evictions, fallback tracking)
    - fallback_capacity_offline: fallback bin capacity for offline MILP (scalar or per-dim)
    - fallback_capacity_online: fallback capacity placeholder for online (scalar or per-dim)
    """
    N: int
    M_off: int
    capacities: List[float]
    dimensions: int = 1
    capacity_mean: float = 1.0
    capacity_std: float = 0.1
    fallback_is_enabled: bool = True
    binpacking: bool = True
    fallback_capacity_offline: float | List[float] = 1e6
    fallback_capacity_online: float | List[float] = 1e6

@dataclass
class VolumeGenerationConfig:
    """
    Volume distributions for offline and online items.
    - offline_beta: Beta distribution parameters (shared or per-dimension).
    - offline_bounds: lower/upper bounds (shared or per-dimension).
    - online_beta: Beta distribution parameters (shared or per-dimension).
    - online_bounds: lower/upper bounds (shared or per-dimension).
    """
    offline_beta: Tuple[float, float] = (1, 1)
    offline_bounds: Tuple[float, float] = (0.05, 0.3)
    online_beta: Tuple[float, float] = (2.0, 5.0)
    online_bounds: Tuple[float, float] = (0.05, 0.3)

@dataclass
class GraphGenerationConfig:
    """
    Feasibility graph parameters:
    - p_off: edge prob for G_off (item j can be assigned to bin i)
    - p_onl: edge prob for G_onl^(k) per arrival
    """
    p_off: float = 0.7
    p_onl: float = 0.5

@dataclass
class CostConfig:
    """
    Cost model for assignments and evictions.
    - assign_beta: Beta distribution parameters for assignment costs
    - assign_bounds: lower/upper bounds applied to assignment costs
    - huge_fallback: large fallback cost to ensure feasibility but discourage use
    - reassignment_penalty: base penalty for evicting an OFFLINE item (per default PER-ITEM)
    - penalty_mode: 'per_item' | 'per_volume'  (we default to per_item but can switch later)
    - per_volume_scale: if penalty_mode == 'per_volume', use penalty = per_volume_scale * volume
    """
    base_assign_range: Tuple[float, float] = (1.0, 5.0)
    assign_beta: Tuple[float, float] = (1.0, 1.0)
    assign_bounds: Tuple[float, float] = (1.0, 5.0)
    huge_fallback: float = 1e6
    reassignment_penalty: float = 10.0
    penalty_mode: str = "per_item"     # or "per_volume"
    per_volume_scale: float = 10.0     # used only if penalty_mode == "per_volume"

@dataclass
class StochasticConfig:
    """
    Horizon + stochastic arrivals for online phase (already planned for later).
    - horizon_dist: 'fixed' (use 'horizon') or name of a distribution you might add later
    - horizon: default number of online items to generate
    """
    horizon_dist: str = "fixed"
    horizon: int = 100

@dataclass
class SlackConfig:
    """
    Slack control (even if default is 'no slack') so we can switch later without refactors.
    - enforce_slack: if True, enforce a global fraction of capacity to remain unused
    - fraction: fraction in [0,1); effective capacity is (1 - fraction) * C_i
    - apply_to_online: if False, only the offline stage honors slack and the online stage
      sees full physical capacities.
    """
    enforce_slack: bool = False
    fraction: float = 0.0
    apply_to_online: bool = True


@dataclass
class UtilizationPricingConfig:
    """
    Controls the utilization-based pricing heuristic (UtilizationPricedDecreasing).
    - update_rule: 'polynomial' (current behavior) or 'exponential'
    - price_exponent: exponent for the polynomial rule (lambda ∝ util^exp)
    - exp_rate: growth rate for the exponential rule (lambda ∝ exp(exp_rate*util) - 1)
    - vector_prices: if True, use per-dimension prices and dot products
    """
    update_rule: str = "polynomial"
    price_exponent: float = 2.0
    exp_rate: float = 4.0
    vector_prices: bool = True


@dataclass
class DLAConfig:
    """
    Dynamic Learning Algorithm (Agrawal et al. 2014) controls.
    - epsilon: base sampling fraction that sets geometric update times t_k = ε * M_onl * 2^k
    - log_prices: if True, dump per-phase prices/residuals for plotting/debugging
    - output_dir: directory where per-run DLA logs are stored
    - min_phase_len: optional lower bound on phase length to avoid tiny intervals
    - use_offline_slack: if True, respect cfg.slack settings when building residual LPs
    """
    epsilon: float = 0.1
    log_prices: bool = False
    output_dir: str = "binpacking/results/dla"
    min_phase_len: int = 1
    use_offline_slack: bool = True

@dataclass
class SolverConfig:
    """
    Solver-specific configuration options.
    - use_warm_start: whether to automatically generate warm start solutions
    - warm_start_heuristic: which heuristic to use for warm start ("FFD", "BFD", "CBFD", "PD", "none")
    """
    use_warm_start: bool = False
    warm_start_heuristic: str = "BFD"  # "FFD", "BFD", "CBFD", "PD", "none"


@dataclass
class HeuristicConfig:
    """
    Scalarization choices for vector bin packing.
    - size_key: "max" | "l1" | "l2" for item ordering (FFD/BFD).
    - residual_scalarization: "max" | "l1" | "l2" for residual scoring.
    """
    size_key: str = "max"
    residual_scalarization: str = "max"

@dataclass
class EvalConfig:
    """
    Reproducibility and evaluation bookkeeping.
    - seeds: list of RNG seeds for repeated runs
    """
    seeds: Tuple[int, ...] = (1, 2, 3)

@dataclass
class Config:
    problem: ProblemConfig
    volumes: VolumeGenerationConfig
    graphs: GraphGenerationConfig
    costs: CostConfig
    stoch: StochasticConfig
    slack: SlackConfig
    util_pricing: UtilizationPricingConfig
    dla: DLAConfig
    solver: SolverConfig
    heuristics: HeuristicConfig
    eval: EvalConfig


def load_config_data(data: dict) -> Config:
    """
    Build Config from a parsed YAML dictionary.
    """
    return Config(
        problem=ProblemConfig(**data["problem"]),
        volumes=VolumeGenerationConfig(**data["volumes"]),
        graphs=GraphGenerationConfig(**data["graphs"]),
        costs=CostConfig(**data["costs"]),
        stoch=StochasticConfig(**data["stoch"]),
        slack=SlackConfig(**data["slack"]),
        util_pricing=UtilizationPricingConfig(**data.get("util_pricing", {})),
        dla=DLAConfig(**data.get("dla", {})),
        solver=SolverConfig(**data.get("solver", {})),
        heuristics=HeuristicConfig(**data.get("heuristics", {})),
        eval=EvalConfig(tuple(data["eval"]["seeds"])),
    )


def load_config(path: str | Path) -> Config:
    """
    Load YAML into strongly-typed dataclasses. Fails early if keys are missing.
    """
    data = yaml.safe_load(Path(path).read_text())
    return load_config_data(data)
