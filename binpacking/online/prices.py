from __future__ import annotations
from pathlib import Path
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from generic.config import Config
from generic.general_utils import effective_capacity
from generic.models import AssignmentState, Instance

def compute_prices(
    cfg: Config,
    instance: Instance,
    offline_state: AssignmentState,
    out_path: Path,
    *,
    log_to_console: bool = False,
) -> dict[int, float | list[float]]:
    """
    1) Take the offline state and compute the residual problem
    2) Build a fractional LP for ONLINE items only (regular bins 0..N-1).
    3) Minimize assignment cost; read duals Pi of cap constraints as λ_i.
    4) Save to JSON and return {bin_i: lambda_i}.
    """
    inst = instance
    N = len(inst.bins)

    # Effective capacities (respect slack if online should honor it)
    use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
    slack_fraction = cfg.slack.fraction if use_slack else 0.0
    caps_eff = [
        effective_capacity(b.capacity, use_slack, slack_fraction)
        for b in inst.bins
    ]
    residual = [caps_eff[i] - offline_state.load[i] for i in range(N)]
    residual = [np.maximum(0.0, r) for r in residual]

    # -- Step 2: fractional LP for ONLINE items only
    m = gp.Model("online_fractional_pricing")
    m.Params.OutputFlag = 1 if log_to_console else 0

    allow_fallback = cfg.problem.fallback_is_enabled and cfg.problem.fallback_allowed_online
    # Variables x[j,i] for feasible regular bins and y[j] for fallback usage
    x = {}
    y_fallback = {}
    online_volumes = {item.id: item.volume for item in inst.online_items}
    for item in inst.online_items:
        for i in item.feasible_bins:   # regular bins only
            x[(item.id, i)] = m.addVar(lb=0.0, ub=1.0, name=f"x_{item.id}_{i}")
        if allow_fallback:
            y_fallback[item.id] = m.addVar(lb=0.0, ub=1.0, name=f"y_fallback_{item.id}")
    m.update()
    
    # Bin capacities (residual)
    dims = inst.bins[0].capacity.shape[0] if inst.bins else 1
    cap_constr: dict[tuple[int, int], gp.Constr] = {}
    for i in range(N):
        for d in range(dims):
            expr = gp.quicksum(
                online_volumes[j][d] * var for (j, ii), var in x.items() if ii == i
            )
            cap_constr[(i, d)] = m.addConstr(expr <= float(residual[i][d]), name=f"cap_{i}_{d}")

    # Assignment equality per online item: either use regular capacity or fallback slack.
    for item in inst.online_items:
        item_vars = [var for (j, i), var in x.items() if j == item.id]
        expr = gp.quicksum(item_vars)
        if allow_fallback:
            expr += y_fallback[item.id]
        m.addConstr(expr == 1.0, name=f"assign_{item.id}")

    # Objective: minimize ONLINE assignment cost
    obj = gp.quicksum(inst.costs.assignment_costs[j, i] * var for (j, i), var in x.items())
    if allow_fallback:
        fallback_cost = cfg.costs.huge_fallback
        obj += gp.quicksum(fallback_cost * y for y in y_fallback.values())
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP not optimal, status={m.Status}")

    prices: dict[int, float | list[float]] = {}
    for i in range(N):
        if dims == 1:
            prices[i] = abs(float(cap_constr[(i, 0)].Pi))
        else:
            prices[i] = [abs(float(cap_constr[(i, d)].Pi)) for d in range(dims)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prices": prices}, f, indent=2)
    return prices
