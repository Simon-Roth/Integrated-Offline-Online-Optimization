from __future__ import annotations

from typing import Protocol, Tuple

from generic.core.models import AssignmentState, Instance
from generic.core.models import OfflineSolutionInfo


class BaseOfflinePolicy(Protocol):
    """
    Interface that all offline heuristics/solvers must implement.
    """

    def solve(self, inst: Instance) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """
        Solve the offline phase for the given instance.

        Returns
        -------
        AssignmentState, info:
            The assignment state and a solver-specific info object.
        """
        ...
