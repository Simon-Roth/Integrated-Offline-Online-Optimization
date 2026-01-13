from __future__ import annotations

from typing import Protocol, Tuple, Any

from generic.models import AssignmentState, Instance


class BaseOfflinePolicy(Protocol):
    """
    Interface that all offline heuristics/solvers must implement.
    """

    def solve(self, inst: Instance) -> Tuple[AssignmentState, Any]:
        """
        Solve the offline phase for the given instance.

        Returns
        -------
        AssignmentState, info:
            The assignment state and a solver-specific info object.
        """
        ...
