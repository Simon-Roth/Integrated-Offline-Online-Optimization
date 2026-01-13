from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HeuristicSolutionInfo:
    """Information about heuristic solution"""
    algorithm: str
    runtime: float
    obj_value: float
    feasible: bool
    items_in_fallback: int
    utilization: float