from dataclasses import dataclass
from gurobipy import GRB

@dataclass
class OfflineSolutionInfo:
    """Lightweight summary of the offline solve for logging/eval."""
    algorithm: str
    status: str
    obj_value: float
    runtime: float
    feasible: bool


def _status_name(code: int) -> str:
    """
    Robust, versionsunabhängiger Mapper von Gurobi-Statuscodes auf Namen.
    (Dadurch kein Zugriff auf private Attribute wie _intToStatus. -> versionsabhängig)
    """
    return {
        GRB.OPTIMAL:         "OPTIMAL",
        GRB.INFEASIBLE:      "INFEASIBLE",
        GRB.INF_OR_UNBD:     "INF_OR_UNBD",
        GRB.UNBOUNDED:       "UNBOUNDED",
        GRB.TIME_LIMIT:      "TIME_LIMIT",
        GRB.INTERRUPTED:     "INTERRUPTED",
        GRB.SUBOPTIMAL:      "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT:  "USER_OBJ_LIMIT",
    }.get(code, f"STATUS_{code}")
    
