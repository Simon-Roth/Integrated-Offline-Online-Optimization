from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

OFFLINE_MILP_GENERIC = "generic.offline.offline_solver.OfflineMILPSolver"
OFFLINE_MILP_BINPACKING = "binpacking.offline.offline_solver.OfflineMILPSolver"
OFFLINE_CABFD = "binpacking.offline.offline_heuristics.cost_best_fit_decreasing.CostAwareBestFitDecreasing"
OFFLINE_UTILIZATION = "binpacking.offline.offline_heuristics.utilization_priced.UtilizationPricedDecreasing"

ONLINE_ROLLING_MILP = "generic.online.policies.RollingHorizonMILPPolicy"
ONLINE_CABF = "binpacking.online.online_heuristics.cost_best_fit.CostAwareBestFitOnlinePolicy"
ONLINE_SIM_BASE = "binpacking.online.online_heuristics.sim_base.SimBasePolicy"
ONLINE_SIM_DUAL = "generic.online.policies.SimDualPolicy"
ONLINE_DYNAMIC_LEARNING = "binpacking.online.online_heuristics.dynamic_learning.DynamicLearningPolicy"
ONLINE_PRIMAL_DUAL = "generic.online.policies.PrimalDualPolicy"
ONLINE_POLICIES_NEED_PRICES = {ONLINE_SIM_BASE, ONLINE_SIM_DUAL}


def online_policy_needs_prices(policy_path: str) -> bool:
    return policy_path in ONLINE_POLICIES_NEED_PRICES


def online_policy_price_path(policy_path: str) -> str | None:
    if policy_path == ONLINE_SIM_BASE:
        return "binpacking/results/sim_base.json"
    if policy_path == ONLINE_SIM_DUAL:
        return "generic/results/sim_dual.json"
    return None


@dataclass(frozen=True)
class PipelineSpec:
    """
    Minimal pipeline specification: choose offline solver + online policy classes.
    """

    name: str
    offline_solver: str
    online_policy: str


class PipelineRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, PipelineSpec] = {}

    def register(self, spec: PipelineSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Pipeline '{spec.name}' is already registered.")
        self._specs[spec.name] = spec

    def list(self) -> List[PipelineSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> PipelineSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._specs))
            raise KeyError(f"Unknown pipeline '{name}'. Known pipelines: {known}") from exc


def default_registry() -> PipelineRegistry:
    """
    Populate all combinations of the registered offline/online variants.
    """
    registry = PipelineRegistry()
    offline_specs = [
        ("binpacking_milp", OFFLINE_MILP_BINPACKING),
        ("cabfd", OFFLINE_CABFD),
        ("util", OFFLINE_UTILIZATION),
    ]
    online_specs = [
        ("rolling_horizon_milp", ONLINE_ROLLING_MILP),
        ("cost_best_fit", ONLINE_CABF),
        #("sim_base", ONLINE_SIM_BASE),
        ("sim_dual", ONLINE_SIM_DUAL),
        ("dynamic_learning", ONLINE_DYNAMIC_LEARNING),
        ('primal_dual',ONLINE_PRIMAL_DUAL)
    ]
    for offline_label, offline_solver in offline_specs:
        for online_label, online_policy in online_specs:
            registry.register(
                PipelineSpec(
                    name=f"{offline_label}+{online_label}",
                    offline_solver=offline_solver,
                    online_policy=online_policy,
                )
            )
    return registry
