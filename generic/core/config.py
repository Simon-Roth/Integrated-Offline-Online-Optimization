# generic/core/config.py
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
    - n: number of options per step (length of x_t)
    - T_off: number of offline steps
    - m: number of capacity constraints (length of b)
    - b: capacity vector b (len == m); if empty, sampled from b_mean/b_std
    - b_mean/std: parameters to synthesize b when list shorter than m
    - fallback_is_enabled: if False, no rejection option exists in the instance
    - fallback_allowed_offline: if True, offline steps may use the rejection option
    - fallback_allowed_online: if True, online steps may use the rejection option
    - allow_reassignment: if True, BGAP-specific online policies may evict/reassign offline steps
    """
    n: int = 10
    T_off: int = 40
    m: int = 10
    b: List[float] = field(default_factory=list)
    b_mean: float = 1680
    b_std: float = 20
    fallback_is_enabled: bool = True
    fallback_allowed_offline: bool = True
    fallback_allowed_online: bool = True
    allow_reassignment: bool = False

@dataclass
class GenerationConfig:
    """
    Instance generator selection.
    - generator: "generic" | "bgap"
    """
    generator: str = "generic"

@dataclass
class CapCoeffGenerationConfig:
    """
    Coefficient distributions for A_t^{cap} (bgap interprets these as sizes).
    - offline_beta: Beta distribution parameters for offline coefficients.
    - offline_bounds: lower/upper bounds for offline coefficients.
    - online_beta: Beta distribution parameters for online coefficients.
    - online_bounds: lower/upper bounds for online coefficients.
    """
    offline_beta: Tuple[float, float] = (3.0, 7.0)
    offline_bounds: Tuple[float, float] = (30.0, 180.0)
    online_beta: Tuple[float, float] = (3.0, 7.0)
    online_bounds: Tuple[float, float] = (30.0, 180.0)

@dataclass
class FeasibilityGenerationConfig:
    """
    Feasibility sampling parameters (used to build A_t^{feas} rows).
    - p_off: probability an option is feasible for an offline step (uniform mode)
    - p_onl: probability an option is feasible for an online step (uniform mode)
    - mode: 'uniform' | 'exp_bin'
    - exp_bin_offline/online: per-bin exp decay params (p_min, p_max, alpha)
    """
    p_off: float = 0.5
    p_onl: float = 0.5
    mode: str = "uniform"
    exp_bin_offline: dict = field(default_factory=lambda: {"p_max": 0.6, "p_min": 0.1, "alpha": 3.0})
    exp_bin_online: dict = field(default_factory=lambda: {"p_max": 0.6, "p_min": 0.1, "alpha": 3.0})

@dataclass
class CostConfig:
    """
    Cost model for assignments and evictions.
    - assign_beta: Beta distribution parameters for assignment costs
    - assign_bounds: lower/upper bounds applied to assignment costs
    - observe_future_online_costs: if True, policies may use realized future online costs
    - huge_fallback: large fallback cost to ensure feasibility but discourage use
    - reassignment_penalty: base penalty for evicting an OFFLINE step (per default PER-ITEM)
    - penalty_mode: 'per_item' | 'per_usage'  (per-step penalty, I kept "item" name for compatibility (instead of "step"))
    - per_usage_scale: if penalty_mode == 'per_usage', use penalty = per_usage_scale * ||A_t^{cap}||_1
    - fail_penalty_per_item: penalty per unplaced step when a phase fails
    - fail_penalty_scale: multiplier applied to fail_penalty_per_item
    - stop_online_on_first_failure: if True, stop online processing at first unplaced step;
      if False, continue and treat later steps as usual
    """
    base_assign_range: Tuple[float, float] = (1.0, 5.0)
    assign_beta: Tuple[float, float] = (2.0, 5.0)
    assign_bounds: Tuple[float, float] = (1.0, 5.0)
    observe_future_online_costs: bool = True
    huge_fallback: float = 7.5
    reassignment_penalty: float = 7.0
    penalty_mode: str = "per_item"     # or "per_usage"
    per_usage_scale: float = 10.0     # used only if penalty_mode == "per_usage"
    fail_penalty_per_item: float = 7.0
    fail_penalty_scale: float = 1.0
    stop_online_on_first_failure: bool = True

@dataclass
class StochasticConfig:
    """
    Horizon + stochastic arrivals for online phase (already planned for later).
    - horizon_dist: 'fixed' (use 'T_onl') or name of a distribution you might add later
    - T_onl: default number of online steps to generate
    """
    horizon_dist: str = "fixed"
    T_onl: int = 160

@dataclass
class SlackConfig:
    """
    Slack control.
    - enforce_slack: if True, enforce a global fraction of capacity to remain unused
    - fraction: fraction in [0,1); effective capacity is (1 - fraction) * C_i
    - apply_to_online: if False, only the offline stage honors slack and the online stage
      sees full physical capacities.
    """
    enforce_slack: bool = False
    fraction: float = 0.1
    apply_to_online: bool = False


@dataclass
class UtilizationPricingConfig:
    """
    Controls the utilization-based pricing heuristic (UtilizationPricedDecreasing).
    - update_rule: 'polynomial' (current behavior) or 'exponential'
    - price_exponent: exponent for the polynomial rule (lambda ∝ util^exp)
    - exp_rate: growth rate for the exponential rule (lambda ∝ exp(exp_rate*util) - 1)
    - vector_prices: if True, use per-dimension prices and dot products
    """
    update_rule: str = "exponential"
    price_exponent: float = 2.0
    exp_rate: float = 3.0
    vector_prices: bool = True


@dataclass
class DLAConfig:
    """
    Dynamic Learning Algorithm controls (inspiration: Agrawal et al. 2014).
    - epsilon: base sampling fraction that sets geometric update times t_k = ε * T_onl * 2^k
    - log_prices: if True, dump per-phase prices/residuals for plotting/debugging
    - output_dir: directory where per-run DLA logs are stored
    - min_phase_len: optional lower bound on phase length to avoid tiny intervals
    - use_offline_slack: if True, respect cfg.slack settings when building residual LPs
    - lambda0_init: initial price vector before first phase update
      ("zero" | "offline_util" | "sim_lp")
    """
    epsilon: float = 0.01
    log_prices: bool = False
    output_dir: str = "outputs/generic/results/dla"
    min_phase_len: int = 25
    use_offline_slack: bool = True
    lambda0_init: str = "zero"

@dataclass
class PricingSimulationConfig:
    """
    Shared simulation settings for sampled LP pricing.
    - num_samples: number of samples used (>=1)
    - sample_online_caps: if True, sample online A_t^{cap} and feasibility
    - fallback_allowed_online_for_pricing: if True, force fallback to remain available
      in pricing LPs even when online execution disallows fallback
    """
    num_samples: int = 10
    sample_online_caps: bool = True
    fallback_allowed_online_for_pricing: bool = True

@dataclass
class RollingMILPConfig:
    """
    Rolling-horizon MILP controls.
    - rollout_mode: "single" | "batch"
    - num_rollouts: number of sampled rollouts in batch mode (>=1)
    """
    rollout_mode: str = "single"
    num_rollouts: int = 100

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
    - lambda0_init: "offline_util" | "sim_lp" | "zero"
    - offline_util_init_scale: independent scale for offline-util init.
      Must be set when lambda0_init == "offline_util".
    - sim_lp_init_scale: multiplicative scale for sim_lp warm-start prices
    """
    eta_mode: str = "sqrt"
    eta0: float = 0.05
    eta_decay: float = 0.0
    eta_min: float = 0.0
    normalize_update: bool = True
    normalize_costs: bool = False
    use_remaining_capacity_target: bool = False
    cost_scale_mode: str = "assign_mean"
    cost_scale_min: float = 1e-8
    lambda0_init: str = "offline_util"
    offline_util_init_scale: Optional[float] = None
    sim_lp_init_scale: float = 1.0

@dataclass
class SolverConfig:
    """
    Solver-specific configuration options.
    - use_warm_start: whether to automatically generate warm start solutions
    - warm_start_heuristic: which heuristic to use for warm start ("CABFD", "none")
    """
    use_warm_start: bool = True
    warm_start_heuristic: str = "CABFD" 


@dataclass
class HeuristicConfig:
    """
    Scalarization choices for vector-capacity ordering.
    - size_key: "max" | "l1" | "l2" for step ordering (FFD/BFD).
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
    seeds: Tuple[int, ...] = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    )
    track_offline_util_per_bin: bool = True # Needed to visualize residual capacities after offline phase for BGAP

@dataclass
class Config:
    problem: ProblemConfig
    generation: GenerationConfig
    cap_coeffs: CapCoeffGenerationConfig
    feasibility: FeasibilityGenerationConfig
    costs: CostConfig
    stoch: StochasticConfig
    slack: SlackConfig
    util_pricing: UtilizationPricingConfig
    dla: DLAConfig
    pricing_sim: PricingSimulationConfig
    rolling_milp: RollingMILPConfig
    primal_dual: PrimalDualConfig
    solver: SolverConfig
    heuristics: HeuristicConfig
    eval: EvalConfig


def load_config_data(data: dict) -> Config:
    """
    Build Config from a parsed YAML dictionary.
    """
    problem_data = dict(data["problem"])
    stoch_data = dict(data["stoch"])
    pricing_sim_data = dict(data.get("pricing_sim", data.get("sim_dual", {})))
    return Config(
        problem=ProblemConfig(**problem_data),
        generation=GenerationConfig(**data.get("generation", {})),
        cap_coeffs=CapCoeffGenerationConfig(**data["cap_coeffs"]),
        feasibility=FeasibilityGenerationConfig(**data["feasibility"]),
        costs=CostConfig(**data["costs"]),
        stoch=StochasticConfig(**stoch_data),
        slack=SlackConfig(**data["slack"]),
        util_pricing=UtilizationPricingConfig(**data.get("util_pricing", {})),
        dla=DLAConfig(**data.get("dla", {})),
        pricing_sim=PricingSimulationConfig(**pricing_sim_data),
        rolling_milp=RollingMILPConfig(**data.get("rolling_milp", {})),
        primal_dual=PrimalDualConfig(**data.get("primal_dual", {})),
        solver=SolverConfig(**data.get("solver", {})),
        heuristics=HeuristicConfig(**data.get("heuristics", {})),
        eval=EvalConfig(
            tuple(data["eval"]["seeds"]),
            bool(data["eval"].get("track_offline_util_per_bin", False)),
        ),
    )


def load_config(path: str | Path) -> Config:
    """
    Load YAML into strongly-typed dataclasses. Fails early if keys are missing.
    """
    data = yaml.safe_load(Path(path).read_text())
    return load_config_data(data)
