# generic/models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

BinId = int
ItemId = int

# ---------------------------
# Static problem definitions
# ---------------------------

@dataclass(frozen=True)
class BinSpec:
    """
    Regular bin specification.
    - id: 0..N-1
    - capacity: capacity vector (length d)
    """
    id: BinId
    capacity: np.ndarray

@dataclass(frozen=True)
class ItemSpec:
    """
    Item specification.
    - id: unique integer id (offline: 0..M_off-1; online arrivals start at M_off)
    - volume: volume vector (length d)
    """
    id: ItemId
    volume: np.ndarray

@dataclass
class Costs:
    """
    Assignment and eviction costs.
    - assign: (M_total x N) or (M_total x (N+1)); last column is fallback when enabled
    - reassignment_penalty: base penalty used when evicting an OFFLINE item
    - penalty_mode: 'per_item' or 'per_volume'
    - per_volume_scale: factor for per-volume penalty
    - huge_fallback: large constant cost for fallback bin to guarantee feasibility
    """
    assign: np.ndarray
    reassignment_penalty: float
    penalty_mode: str = "per_item"     # 'per_item' or 'per_volume'
    per_volume_scale: float = 10.0
    huge_fallback: float = 1e6

@dataclass
class FeasibleGraph:
    """
    Feasibility mask for item-bin edges.
    - feasible[j, i] = 1 if item j is allowed in bin i; 0 otherwise.
      For ONLINE items, the fallback column must be 0 when fallback is enabled.
    """
    feasible: np.ndarray  # shape (M_total, N) or (M_total, N+1)

@dataclass
class Instance:
    """
    Full offline and optional online instance at t=0.
    - bins: list of BinSpec (length N)
    - offline_items: list of ItemSpec (length M_off)
    - costs: Costs
    - feasible: FeasibleGraph for OFFLINE items (G_off)
    - fallback_bin_index: index == N (i.e., after N regular bins), or -1 if disabled
    - online_items: List of OnlineItem (length M_on)
    - online_feasible: Optional feasible graph for online items
    """
    bins: List[BinSpec]
    offline_items: List[ItemSpec]
    costs: Costs
    feasible: FeasibleGraph
    fallback_bin_index: int
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
    - volume: size vector (length d)
    - feasible_bins: regular bins only (no fallback), given at arrival time
    """
    id: ItemId
    volume: np.ndarray
    feasible_bins: List[BinId]

# ---------------------------
# Mutable assignment state
# ---------------------------

@dataclass
class AssignmentState:
    """
    Current assignment state during the process/simulation.
    - load: current load per bin (shape (N, d) or (N+1, d))
    - assigned_bin: mapping item -> bin index
    - offline_evicted: set of offline ids that were evicted at least once during online phase
    """
    load: np.ndarray
    assigned_bin: Dict[ItemId, BinId]
    offline_evicted: set[ItemId] = field(default_factory=set)

@dataclass
class Decision:
    """
    Bookkeeping for a single online arrival 
    - placed_item: (item_id, bin_id_selected)
    - evicted_offline: list of (item_id, from_bin) that were evicted
    - reassignments: list of (offline_item_id, fallback_bin) when fallback is enabled
    - incremental_cost: cost accrued by this decision
    """
    placed_item: Tuple[ItemId, BinId]
    evicted_offline: List[Tuple[ItemId, BinId]]
    reassignments: List[Tuple[ItemId, BinId]]
    incremental_cost: float
