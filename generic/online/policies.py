from __future__ import annotations

from typing import Protocol, Optional

import numpy as np

from generic.models import AssignmentState, Instance, OnlineItem, Decision


class BaseOnlinePolicy(Protocol):
    """
    Interface that all online heuristics/solvers must implement.
    """

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        """
        Decide how to place 'item' given the current assignment state.

        Parameters
        ----------
        item:
            The arriving online item.
        state:
            Current assignment state (will be mutated by the solver after the decision).
        instance:
            The full problem instance (offline + online data).
        feasible_row:
            Row of the feasibility matrix for this item, if available (length N or N+1).

        Returns
        -------
        Decision:
            Encodes bin placement, evictions, reassignments, and incremental cost.
        """
        ...


class PolicyInfeasibleError(Exception):
    """Raised when a policy cannot produce a feasible placement for the arriving item."""

    pass
