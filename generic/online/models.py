from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from generic.models import Decision


@dataclass
class OnlineSolutionInfo:
    """
    Lightweight summary of the online phase.

    Attributes
    ----------
    status:
        Run outcome string (e.g. 'COMPLETED', 'NO_ITEMS', 'ERROR').
    runtime:
        Wall-clock duration of the online decision loop.
    total_cost:
        Sum of incremental costs reported by the policy.
    fallback_items:
        Number of items assigned to the fallback bin after the online phase (0 if disabled).
    evicted_offline:
        Count of offline eviction events during the online phase (penalty-bearing moves).
    decisions:
        Optional log of per-item decisions for analysis/debugging.
    """

    status: str
    runtime: float
    total_cost: float
    fallback_items: int
    evicted_offline: int
    decisions: List[Decision] = field(default_factory=list)
