from __future__ import annotations

import sys
import copy
from dataclasses import dataclass
from typing import Callable, Tuple
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from generic.config import Config
from generic.models import Instance, ItemSpec, FeasibleGraph, Costs, AssignmentState
from generic.offline.models import OfflineSolutionInfo

OfflineSolverFactory = Callable[[Config], object]


def build_full_horizon_instance(instance: Instance) -> Instance:
    """
    Construct an Instance that contains both offline and online items so that
    solving the offline MILP on it yields the full-horizon optimum.
    Online items are converted into ItemSpec and their feasibility matrix rows
    appended with fallback column set to zero.
    """
        
    offline_specs = [ItemSpec(id=item.id, volume=item.volume) for item in instance.offline_items]
    online_specs = [ItemSpec(id=item.id, volume=item.volume) for item in instance.online_items]
    all_items = offline_specs + online_specs

    offline_feas = instance.feasible.feasible
    if instance.online_feasible is not None:
        online_feas = instance.online_feasible.feasible
        feas_full = np.vstack([offline_feas, online_feas])
    else:
        feas_full = offline_feas.copy()

    # Ensure the fallback column is feasible for all items when fallback is enabled,
    # so warm-start heuristics (e.g., CBFD) do not fail during the optimal solve.
    fallback_idx = instance.fallback_bin_index
    if 0 <= fallback_idx < feas_full.shape[1]:
        feas_full[len(offline_specs) :, fallback_idx] = 1

    costs = Costs(
        assign=instance.costs.assign.copy(),
        reassignment_penalty=instance.costs.reassignment_penalty,
        penalty_mode=instance.costs.penalty_mode,
        per_volume_scale=instance.costs.per_volume_scale,
        huge_fallback=instance.costs.huge_fallback,
    )

    return Instance(
        bins=instance.bins,
        offline_items=all_items,
        costs=costs,
        feasible=FeasibleGraph(feasible=feas_full),
        fallback_bin_index=instance.fallback_bin_index,
        online_items=[],
        online_feasible=None,
    )

 
def solve_full_horizon_optimum(
    cfg: Config,
    base_instance: Instance,
    offline_solver_factory: OfflineSolverFactory,
) -> Tuple[AssignmentState, OfflineSolutionInfo]:
    """
    Solve the full-horizon (offline + online) MILP optimally by converting the
    combined instance into a single offline MILP.
    """
    full_instance = build_full_horizon_instance(base_instance)
    cfg_no_slack = copy.deepcopy(cfg)
    cfg_no_slack.slack.enforce_slack = False
    cfg_no_slack.slack.fraction = 0.0
    solver = offline_solver_factory(cfg_no_slack)
    return solver.solve(full_instance)
