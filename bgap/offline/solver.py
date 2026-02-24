from __future__ import annotations

from typing import Dict

from generic.core.models import Instance
from generic.offline.solver import OfflineMILPSolver as GenericOfflineMILPSolver


class OfflineMILPSolver(GenericOfflineMILPSolver):
    """BGAP MILP solver with warm-start heuristics."""

    def _generate_warm_start(self, inst: Instance) -> Dict[int, int]:
        heuristic_name = str(self.cfg.solver.warm_start_heuristic)
        heuristic_key = heuristic_name.upper()
        if heuristic_key == "NONE":
            return {}

        if heuristic_key == "CABFD":
            from bgap.offline.policies.cost_best_fit_decreasing import CostAwareBestFitDecreasing

            heuristic = CostAwareBestFitDecreasing(self.cfg)
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
