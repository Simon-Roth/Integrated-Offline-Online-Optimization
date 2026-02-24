# generic/core/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

OptionId = int
StepId = int

@dataclass(frozen=True)
class StepSpec:
    """
    Step specification.
    - step_id: unique integer id (offline: 0..T_off-1; online steps start at T_off)
    - cap_matrix: A_t^{cap} for this step (shape m x n)
    - feas_matrix: A_t^{feas} for this step (shape p_t x n')
    - feas_rhs: b_t for this step (shape p_t,) (p_t made up of structural (one-hot in our case) + compatibility (uniform/exp.) parts)
    """
    step_id: StepId
    cap_matrix: np.ndarray
    feas_matrix: np.ndarray
    feas_rhs: np.ndarray

@dataclass
class Costs:
    """
    Assignment and eviction costs.
    - assignment_costs: (T_total x n) or (T_total x (n+1)); last column is fallback when enabled
    - reassignment_penalty: base penalty used when evicting an OFFLINE step
    - penalty_mode: 'per_item' or 'per_usage' (per-step penalty, I kept "item" name for compatibility, so "item" = "step")
    - per_usage_scale: factor for per-usage penalty (||A_t^{cap}||_1)
    - huge_fallback: large constant cost for the fallback option (rejection)
    """
    assignment_costs: np.ndarray
    reassignment_penalty: float
    penalty_mode: str = "per_item"     # 'per_item' or 'per_usage'
    per_usage_scale: float = 10.0
    huge_fallback: float = 1e6

@dataclass
class Instance:
    """
    Full offline and optional online instance at t=0.
    - n: number of regular options
    - m: number of capacity constraints (length of b)
    - b: capacity vector b (length m)
    - offline_steps: list of StepSpec (length T_off)
    - costs: Costs
    - fallback_option_index: index == n (after regular options), or -1 if disabled
    - online_steps: list of StepSpec (length T_onl)
    """
    n: int
    m: int
    b: np.ndarray
    offline_steps: List[StepSpec]
    costs: Costs
    fallback_option_index: int
    online_steps: List[StepSpec] = field(default_factory=list)

# ---------------------------
# Mutable assignment state
# ---------------------------

@dataclass
class AssignmentState:
    """
    Current assignment state during the process/simulation.
    - load: current resource usage (shape (m,))
    - assigned_option: mapping step -> option index
    - offline_evicted_steps: set of offline step ids that were evicted at least once during online phase
    """
    load: np.ndarray
    assigned_option: Dict[StepId, OptionId]
    offline_evicted_steps: set[StepId] = field(default_factory=set)

@dataclass
class Decision:
    """
    Bookkeeping for a single online step.
    - placed_step: (step_id, option_id_selected) # (assumes one-hot -> expressed with ID)
    - evicted_offline_steps: list of (step_id, from_option) that were evicted
    - reassigned_offline_steps: list of (offline_step_id, fallback_option) when fallback is enabled
    - incremental_cost: cost accrued by this decision
    """
    placed_step: Tuple[StepId, OptionId]
    evicted_offline_steps: List[Tuple[StepId, OptionId]] # if allow_reassignment is false -> empty (only for BGAP logic)
    reassigned_offline_steps: List[Tuple[StepId, OptionId]] # if allow_reassignment is false -> empty (only for BGAP logic)
    incremental_cost: float


@dataclass
class OfflineSolutionInfo:
    """Lightweight summary of the offline solve for logging/eval."""
    algorithm: str
    status: str
    obj_value: float
    runtime: float
    feasible: bool


def _status_name(code: int) -> str: # legacy
    """
    Robust, versionsunabhängiger Mapper von Gurobi-Statuscodes auf Namen.
    (Dadurch kein Zugriff auf private Attribute wie _intToStatus. -> versionsabhängig)
    """
    try:
        from gurobipy import GRB
    except Exception:
        return f"STATUS_{code}"
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
    total_objective:
        Sum of incremental objective contributions reported by the policy.
    fallback_steps:
        Number of steps assigned to the fallback option after the online phase (0 if disabled).
    evicted_offline_steps:
        Count of offline eviction events during the online phase (penalty-bearing moves).
    decisions:
        Optional log of per-step decisions for analysis/debugging.
    """

    status: str
    runtime: float
    total_objective: float
    fallback_steps: int
    evicted_offline_steps: int
    decisions: List[Decision] = field(default_factory=list)
