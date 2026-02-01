# generic/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import yaml
from pathlib import Path

# ---- Problem & generation knobs ----

@dataclass
class ProblemConfig:
    """
    Core structural parameters for an instance.
    - n: number of actions per time step (length of x_t)
    - M_off: number of offline items
    - m: number of capacity constraints (length of b)
    - b: capacity vector b (len == m); if empty, sampled from b_mean/b_std
    - b_mean/std: parameters to synthesize b when list shorter than m
    - fallback_is_enabled: if False, no rejection action exists in the instance
    - fallback_allowed_offline: if True, offline items may use the rejection action
    - fallback_allowed_online: if True, online items may use the rejection action
    - allow_reassignment: if True, online policies may evict/reassign offline items
    """
    n: int
    M_off: int
    m: int
    b: List[float]
    b_mean: float = 1.0
    b_std: float = 0.1
    fallback_is_enabled: bool = True
    fallback_allowed_offline: bool = True
    fallback_allowed_online: bool = False
    allow_reassignment: bool = False

@dataclass
class CapCoeffGenerationConfig:
    """
    Coefficient distributions for A_t^{cap} (binpacking interprets these as sizes).
    - offline_beta: Beta distribution parameters for offline coefficients.
    - offline_bounds: lower/upper bounds for offline coefficients.
    - online_beta: Beta distribution parameters for online coefficients.
    - online_bounds: lower/upper bounds for online coefficients.
    """
    offline_beta: Tuple[float, float] = (1, 1)
    offline_bounds: Tuple[float, float] = (0.05, 0.3)
    online_beta: Tuple[float, float] = (2.0, 5.0)
    online_bounds: Tuple[float, float] = (0.05, 0.3)

@dataclass
class FeasibilityGenerationConfig:
    """
    Feasibility sampling parameters (used to build A_t^{feas} rows).
    - p_off: probability an action is feasible for an offline item (uniform mode)
    - p_onl: probability an action is feasible for an online item (uniform mode)
    - mode: 'uniform' | 'exp_bin'
    - exp_bin_offline/online: per-bin exp decay params (p_min, p_max, alpha)
    """
    p_off: float = 0.7
    p_onl: float = 0.5
    mode: str = "uniform"
    exp_bin_offline: dict = field(default_factory=dict)
    exp_bin_online: dict = field(default_factory=dict)

@dataclass
class CostConfig:
    """
    Cost model for assignments and evictions.
    - assign_beta: Beta distribution parameters for assignment costs
    - assign_bounds: lower/upper bounds applied to assignment costs
    - huge_fallback: large fallback cost to ensure feasibility but discourage use
    - reassignment_penalty: base penalty for evicting an OFFLINE item (per default PER-ITEM)
    - penalty_mode: 'per_item' | 'per_usage'  (we default to per_item but can switch later)
    - per_usage_scale: if penalty_mode == 'per_usage', use penalty = per_usage_scale * ||A_t^{cap}||_1
    - fail_penalty_per_item: penalty per unplaced item when a phase fails
    - fail_penalty_scale: multiplier applied to fail_penalty_per_item
    """
    base_assign_range: Tuple[float, float] = (1.0, 5.0)
    assign_beta: Tuple[float, float] = (1.0, 1.0)
    assign_bounds: Tuple[float, float] = (1.0, 5.0)
    huge_fallback: float = 1e6
    reassignment_penalty: float = 10.0
    penalty_mode: str = "per_item"     # or "per_usage"
    per_usage_scale: float = 10.0     # used only if penalty_mode == "per_usage"
    fail_penalty_per_item: float = 0.0
    fail_penalty_scale: float = 1.0

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
    output_dir: str = "generic/results/dla"
    min_phase_len: int = 1
    use_offline_slack: bool = True

@dataclass
class SimDualConfig:
    """
    Controls SAA settings for SimDual pricing.
    - saa_samples: number of samples used for SAA (>=1)
    - sample_online_caps: if True, sample online A_t^{cap} (and feasibility) for SAA
    - sample_online_costs: if True, sample online costs for SAA (else use realized c_t)
    """
    saa_samples: int = 1
    sample_online_caps: bool = True
    sample_online_costs: bool = False

@dataclass
class PrimalDualConfig:
    """
    Primal-dual online MILP controls.
    - eta_mode: "constant" | "linear" | "sqrt" | "exponential"
    - eta0: base step size
    - eta_decay: decay rate for linear/exponential schedules
    - eta_min: floor for linear/exponential schedules
    - normalize_update: if True, scale update by (usage - b/T) / (b/T)
    - normalize_costs: if True, divide costs by a scale factor
    - use_remaining_capacity_target: if True, target remaining capacity / remaining steps
    - cost_scale_mode: "assign_mean" | "assign_bounds"
    - cost_scale_min: lower bound for cost scale to avoid divide-by-zero
    """
    eta_mode: str = "constant"
    eta0: float = 0.1
    eta_decay: float = 0.0
    eta_min: float = 0.0
    normalize_update: bool = False
    normalize_costs: bool = False
    use_remaining_capacity_target: bool = False
    cost_scale_mode: str = "assign_mean"
    cost_scale_min: float = 1e-8

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
    Scalarization choices for vector-capacity ordering.
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
    track_offline_util_for_binpacking: bool = False

@dataclass
class Config:
    problem: ProblemConfig
    cap_coeffs: CapCoeffGenerationConfig
    feasibility: FeasibilityGenerationConfig
    costs: CostConfig
    stoch: StochasticConfig
    slack: SlackConfig
    util_pricing: UtilizationPricingConfig
    dla: DLAConfig
    sim_dual: SimDualConfig
    primal_dual: PrimalDualConfig
    solver: SolverConfig
    heuristics: HeuristicConfig
    eval: EvalConfig


def load_config_data(data: dict) -> Config:
    """
    Build Config from a parsed YAML dictionary.
    """
    return Config(
        problem=ProblemConfig(**data["problem"]),
        cap_coeffs=CapCoeffGenerationConfig(**data["cap_coeffs"]),
        feasibility=FeasibilityGenerationConfig(**data["feasibility"]),
        costs=CostConfig(**data["costs"]),
        stoch=StochasticConfig(**data["stoch"]),
        slack=SlackConfig(**data["slack"]),
        util_pricing=UtilizationPricingConfig(**data.get("util_pricing", {})),
        dla=DLAConfig(**data.get("dla", {})),
        sim_dual=SimDualConfig(**data.get("sim_dual", {})),
        primal_dual=PrimalDualConfig(**data.get("primal_dual", {})),
        solver=SolverConfig(**data.get("solver", {})),
        heuristics=HeuristicConfig(**data.get("heuristics", {})),
        eval=EvalConfig(
            tuple(data["eval"]["seeds"]),
            bool(data["eval"].get("track_offline_util_for_binpacking", False)),
        ),
    )


def load_config(path: str | Path) -> Config:
    """
    Load YAML into strongly-typed dataclasses. Fails early if keys are missing.
    """
    data = yaml.safe_load(Path(path).read_text())
    return load_config_data(data)
