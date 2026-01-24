# generic/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

ActionId = int
ItemId = int

@dataclass(frozen=True)
class ItemSpec:
    """
    Item specification.
    - id: unique integer id (offline: 0..M_off-1; online arrivals start at M_off)
    - cap_matrix: A_t^{cap} for this item (shape m x n)
    - feas_matrix: A_t^{feas} for this item (shape p_t x n')
    - feas_rhs: b_t for this item (shape p_t,)
    """
    id: ItemId
    cap_matrix: np.ndarray
    feas_matrix: np.ndarray
    feas_rhs: np.ndarray

@dataclass
class Costs:
    """
    Assignment and eviction costs.
    - assignment_costs: (M_total x n) or (M_total x (n+1)); last column is fallback when enabled
    - reassignment_penalty: base penalty used when evicting an OFFLINE item
    - penalty_mode: 'per_item' or 'per_usage'
    - per_usage_scale: factor for per-usage penalty (||A_t^{cap}||_1)
    - huge_fallback: large constant cost for the fallback action (rejection)
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
    - n: number of regular actions
    - m: number of capacity constraints (length of b)
    - b: capacity vector b (length m)
    - offline_items: list of ItemSpec (length M_off)
    - costs: Costs
    - fallback_action_index: index == n (after regular actions), or -1 if disabled
    - online_items: List of OnlineItem (length M_on)
    """
    n: int
    m: int
    b: np.ndarray
    offline_items: List[ItemSpec]
    costs: Costs
    fallback_action_index: int
    online_items: List[OnlineItem] = field(default_factory=list)


# ---------------------------
# Online arrivals
# ---------------------------

@dataclass(frozen=True)
class OnlineItem:
    """
    Online item arrival.
    - id: unique id (>= M_off)
    - cap_matrix: A_t^{cap} for this item (shape m x n)
    - feas_matrix: A_t^{feas} for this item (shape p_t x n')
    - feas_rhs: b_t for this item (shape p_t,)
    """
    id: ItemId
    cap_matrix: np.ndarray
    feas_matrix: np.ndarray
    feas_rhs: np.ndarray

# ---------------------------
# Mutable assignment state
# ---------------------------

@dataclass
class AssignmentState:
    """
    Current assignment state during the process/simulation.
    - load: current resource usage (shape (m,))
    - assigned_action: mapping item -> action index
    - offline_evicted: set of offline ids that were evicted at least once during online phase
    """
    load: np.ndarray
    assigned_action: Dict[ItemId, ActionId]
    offline_evicted: set[ItemId] = field(default_factory=set)

@dataclass
class Decision:
    """
    Bookkeeping for a single online arrival 
    - placed_item: (item_id, action_id_selected)
    - evicted_offline: list of (item_id, from_action) that were evicted
    - reassignments: list of (offline_item_id, fallback_action) when fallback is enabled
    - incremental_cost: cost accrued by this decision
    """
    placed_item: Tuple[ItemId, ActionId]
    evicted_offline: List[Tuple[ItemId, ActionId]]
    reassignments: List[Tuple[ItemId, ActionId]]
    incremental_cost: float
