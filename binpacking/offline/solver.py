from __future__ import annotations

from typing import Dict

from generic.core.models import Instance
from generic.offline.solver import OfflineMILPSolver as GenericOfflineMILPSolver


class OfflineMILPSolver(GenericOfflineMILPSolver):
    """Bin-packing MILP solver with warm-start heuristics."""

    def _generate_warm_start(self, inst: Instance) -> Dict[int, int]:
        heuristic_name = self.cfg.solver.warm_start_heuristic
        if heuristic_name == "FFD":
            from binpacking.offline.policies.first_fit_decreasing import FirstFitDecreasing

            heuristic = FirstFitDecreasing(self.cfg)
        elif heuristic_name == "BFD":
            from binpacking.offline.policies.best_fit_decreasing import BestFitDecreasing

            heuristic = BestFitDecreasing(self.cfg)
        elif heuristic_name == "CABFD":
            from binpacking.offline.policies.cost_best_fit_decreasing import CostAwareBestFitDecreasing

            heuristic = CostAwareBestFitDecreasing(self.cfg)
        elif heuristic_name == "PD":
            from binpacking.offline.policies.utilization_priced import (
                UtilizationPricedDecreasing,
            )

            heuristic = UtilizationPricedDecreasing(self.cfg)
        else:
            raise ValueError(f"Unknown warm start heuristic: {heuristic_name}")

        try:
            state, _ = heuristic.solve(inst)
        except ValueError as exc:
            print(
                f"Warm start heuristic {heuristic_name} failed ({exc}); "
                "continuing without warm start."
            )
            return {}
        return state.assigned_option
