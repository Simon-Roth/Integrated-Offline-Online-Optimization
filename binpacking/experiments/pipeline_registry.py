from __future__ import annotations

import copy
import numpy as np
from typing import Callable, Dict, List

from generic.config import Config
from binpacking.experiments.pipeline_runner import PipelineSpec

from binpacking.offline.offline_solver import OfflineMILPSolver
from binpacking.offline.offline_heuristics.first_fit_decreasing import FirstFitDecreasing
from binpacking.offline.offline_heuristics.best_fit_decreasing import BestFitDecreasing
from binpacking.offline.offline_heuristics.cost_best_fit_decreasing import CostAwareBestFitDecreasing
from binpacking.offline.offline_heuristics.utilization_priced import UtilizationPricedDecreasing

from binpacking.online.online_heuristics.best_fit import BestFitOnlinePolicy
from binpacking.online.online_heuristics.next_fit import NextFitOnlinePolicy
from binpacking.online.online_heuristics.cost_best_fit import CostAwareBestFitOnlinePolicy
from binpacking.online.online_heuristics.primal_dual import PrimalDualPolicy
from binpacking.online.online_heuristics.dynamic_learning import DynamicLearningPolicy

def make_milp_solver(
    *,
    time_limit: int = 60,
    mip_gap: float = 0.01,
    threads: int = 0,
    log_to_console: bool = False,
) -> Callable[[Config], OfflineMILPSolver]:
    def factory(cfg: Config) -> OfflineMILPSolver:
        return OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )

    return factory


def make_warm_milp_solver(
    heuristic_ctor: Callable[[Config], object],
    *,
    time_limit: int = 60,
    mip_gap: float = 0.01,
    threads: int = 0,
    log_to_console: bool = False,
) -> Callable[[Config], object]:
    def factory(cfg: Config):
        solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )
        heuristic = heuristic_ctor(cfg)

        class WarmStartWrapper:
            def __init__(self) -> None:
                self._solver = solver
                self._heuristic = heuristic

            def solve(self, inst):
                warm_state, _ = self._heuristic.solve(inst)
                return self._solver.solve(inst, warm_start=warm_state.assigned_bin)

        return WarmStartWrapper()

    return factory


def _cfg_with_offline_slack(cfg: Config) -> Config:
    cfg_copy = copy.deepcopy(cfg)
    # Slack sized to expected online volume relative to total capacity.
    # E[V_online] = lo + (hi - lo) * a/(a+b) for Beta(a,b) on [0,1] scaled to [lo,hi].
    dims = max(1, int(getattr(cfg_copy.problem, "dimensions", 1)))
    beta = np.asarray(cfg_copy.volumes.online_beta, dtype=float)
    bounds = np.asarray(cfg_copy.volumes.online_bounds, dtype=float)
    if beta.shape == (2,):
        beta = np.tile(beta, (dims, 1))
    if bounds.shape == (2,):
        bounds = np.tile(bounds, (dims, 1))
    if beta.shape != (dims, 2) or bounds.shape != (dims, 2):
        raise ValueError("online_beta/online_bounds must be length-2 or shape (dims, 2).")

    mean_vol = np.zeros(dims, dtype=float)
    for d in range(dims):
        a, b = beta[d]
        lo, hi = bounds[d]
        if (a + b) > 0:
            mean_vol[d] = lo + (hi - lo) * (a / float(a + b))

    horizon = max(0, cfg_copy.stoch.horizon)
    expected_online_volume = horizon * mean_vol

    capacities = cfg_copy.problem.capacities
    if capacities:
        cap_arr = np.asarray(capacities, dtype=float)
        if cap_arr.ndim == 1 and dims > 1:
            cap_arr = np.tile(cap_arr.reshape((-1, 1)), (1, dims))
        total_capacity = np.sum(cap_arr, axis=0)
    else:
        cap_mean = np.asarray(cfg_copy.problem.capacity_mean, dtype=float)
        if cap_mean.size == 1:
            cap_mean = np.full(dims, float(cap_mean))
        total_capacity = cfg_copy.problem.N * cap_mean

    slack_fraction = 0.0
    ratios = np.zeros(dims, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(total_capacity > 0, expected_online_volume / total_capacity, 0.0)
    if np.any(ratios > 0):
        slack_fraction = max(0.0, min(0.99, float(np.max(ratios))))
    cfg_copy.slack.enforce_slack = slack_fraction > 0.0
    cfg_copy.slack.fraction = slack_fraction
    cfg_copy.slack.apply_to_online = False
    return cfg_copy



PIPELINES: List[PipelineSpec] = [
    PipelineSpec(
        name="MILP+CostAwareBestFit",
        offline_label="MILP",
        online_label="CostAwareBestFit",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="MILP",
    ),
    PipelineSpec(
        name="UtilizationPriced+CostAwareBestFit",
        offline_label="UtilizationPriced",
        online_label="CostAwareBestFit",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
    # PipelineSpec(
    #     name="UtilizationPriced(exp)+CostAwareBestFit",
    #     offline_label="UtilizationPricedExp",
    #     online_label="CostAwareBestFit",
    #     offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg, update_rule="exponential"),
    #     online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
    #     offline_cache_key="UtilizationPricedExp",
    # ),
    PipelineSpec(
        name="UtilizationPriced+PrimalDual",
        offline_label="UtilizationPriced",
        online_label="PrimalDual",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
    # PipelineSpec(
    #     name="UtilizationPriced(exp)+PrimalDual",
    #     offline_label="UtilizationPricedExp",
    #     online_label="PrimalDual",
    #     offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg, update_rule="exponential"),
    #     online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
    #     offline_cache_key="UtilizationPricedExp",
    # ),
    PipelineSpec(
        name="MILP+PrimalDual",
        offline_label="MILP",
        online_label="PrimalDual",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="MILP",
    ),
    PipelineSpec(
        name="MILP(offSlack)+PrimalDual",
        offline_label="MILP_offSlack",
        online_label="PrimalDual",
        offline_factory=lambda cfg, base_factory=make_milp_solver(): base_factory(
            _cfg_with_offline_slack(cfg)
        ),
        online_factory=lambda cfg, price_path=None: PrimalDualPolicy(cfg, price_path=price_path) if price_path else PrimalDualPolicy(cfg),
        offline_cache_key="MILP_offSlack",
    ),
    PipelineSpec(
        name="MILP(offSlack)+CostAwareBestFit",
        offline_label="MILP_offSlack",
        online_label="CostAwareBestFit",
        offline_factory=lambda cfg, base_factory=make_milp_solver(): base_factory(
            _cfg_with_offline_slack(cfg)
        ),
        online_factory=lambda cfg: CostAwareBestFitOnlinePolicy(cfg),
        offline_cache_key="MILP_offSlack",
    ),
    PipelineSpec(
        name="MILP(offSlack)+DynamicLearning",
        offline_label="MILP_offSlack",
        online_label="DynamicLearning",
        offline_factory=lambda cfg, base_factory=make_milp_solver(): base_factory(
            _cfg_with_offline_slack(cfg)
        ),
        online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
        offline_cache_key="MILP_offSlack",
    ),
    PipelineSpec(
        name="MILP+DynamicLearning",
        offline_label="MILP",
        online_label="DynamicLearning",
        offline_factory=make_milp_solver(),
        online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
        offline_cache_key="MILP",
    ),
    PipelineSpec(
        name="UtilizationPriced+DynamicLearning",
        offline_label="UtilizationPriced",
        online_label="DynamicLearning",
        offline_factory=lambda cfg: UtilizationPricedDecreasing(cfg),
        online_factory=lambda cfg, price_path=None: DynamicLearningPolicy(cfg, price_path=price_path) if price_path else DynamicLearningPolicy(cfg),
        offline_cache_key="UtilizationPriced",
    ),
    
]

PIPELINE_REGISTRY: Dict[str, PipelineSpec] = {spec.name: spec for spec in PIPELINES}


def get_pipeline(spec_name: str) -> PipelineSpec:
    try:
        return PIPELINE_REGISTRY[spec_name]
    except KeyError as exc:
        known = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline '{spec_name}'. Known pipelines: {known}") from exc
