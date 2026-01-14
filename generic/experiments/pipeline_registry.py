from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

OFFLINE_MILP_GENERIC = "generic.offline.offline_solver.OfflineMILPSolver"
OFFLINE_MILP_BINPACKING = "binpacking.offline.offline_solver.OfflineMILPSolver"
OFFLINE_CABFD = "binpacking.offline.offline_heuristics.cost_best_fit_decreasing.CostAwareBestFitDecreasing"
OFFLINE_UTILIZATION = "binpacking.offline.offline_heuristics.utilization_priced.UtilizationPricedDecreasing"

ONLINE_CABF = "binpacking.online.online_heuristics.cost_best_fit.CostAwareBestFitOnlinePolicy"
ONLINE_PRIMAL_DUAL = "binpacking.online.online_heuristics.primal_dual.PrimalDualPolicy"
ONLINE_DYNAMIC_LEARNING = "binpacking.online.online_heuristics.dynamic_learning.DynamicLearningPolicy"

ONLINE_POLICIES_NEED_PRICES = {ONLINE_PRIMAL_DUAL}


def online_policy_needs_prices(policy_path: str) -> bool:
    return policy_path in ONLINE_POLICIES_NEED_PRICES


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
        ("cost_best_fit", ONLINE_CABF),
        ("primal_dual", ONLINE_PRIMAL_DUAL),
        ("dynamic_learning", ONLINE_DYNAMIC_LEARNING),
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
