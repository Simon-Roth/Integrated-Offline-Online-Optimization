# generic/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

ActionId = int
ItemId = int

@dataclass(frozen=True)
class ItemSpec:
    """
    Item specification.
    - id: unique integer id (offline: 0..M_off-1; online arrivals start at M_off)
    - cap_matrix: A_t^{cap} for this item (shape m x n)
    """
    id: ItemId
    cap_matrix: np.ndarray

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
class FeasibleGraph:
    """
    Feasibility mask for item-action edges.
    - feasible[j, i] = 1 if action i is allowed for item j; 0 otherwise.
      For ONLINE items, the fallback column is controlled by the online fallback flag.
    """
    feasible: np.ndarray  # shape (M_phase, n) or (M_phase, n+1)

@dataclass
class Instance:
    """
    Full offline and optional online instance at t=0.
    - n: number of regular actions
    - m: number of capacity constraints (length of b)
    - b: capacity vector b (length m)
    - offline_items: list of ItemSpec (length M_off)
    - costs: Costs
    - offline_feasible: FeasibleGraph for OFFLINE items (G_off)
    - fallback_action_index: index == n (after regular actions), or -1 if disabled
    - online_items: List of OnlineItem (length M_on)
    - online_feasible: Optional feasible graph for online items
    """
    n: int
    m: int
    b: np.ndarray
    offline_items: List[ItemSpec]
    costs: Costs
    offline_feasible: FeasibleGraph
    fallback_action_index: int
    online_items: List[OnlineItem] = field(default_factory=list) 
    online_feasible: Optional[FeasibleGraph] = None


# ---------------------------
# Online arrivals
# ---------------------------

@dataclass(frozen=True)
class OnlineItem:
    """
    Online item arrival.
    - id: unique id (>= M_off)
    - cap_matrix: A_t^{cap} for this item (shape m x n)
    - feasible_actions: regular actions only (no fallback), given at arrival time
    """
    id: ItemId
    cap_matrix: np.ndarray
    feasible_actions: List[ActionId]

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
