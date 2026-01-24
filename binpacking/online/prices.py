from __future__ import annotations
from pathlib import Path
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from generic.config import Config
from generic.data.instance_generators import resample_online_items
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.general_utils import effective_capacity
from generic.models import AssignmentState, Instance
from binpacking.block_utils import block_dim, extract_volume, split_capacities

def _pricing_duals_saa(
    cfg: Config,
    samples: list[Instance],
    offline_state: AssignmentState,
    base_instance: Instance,
    *,
    sample_online_costs: bool,
    log_to_console: bool = False,
) -> np.ndarray:
    if not samples:
        return np.zeros((0, 0), dtype=float)
    inst0 = samples[0]
    n = inst0.n
    d = block_dim(n, inst0.m)
    if n == 0:
        return np.zeros((0, d), dtype=float)

    sample_count = max(1, len(samples))
    fallback_idx = inst0.fallback_action_index
    M_off = len(inst0.offline_items)

    # Effective capacities (respect slack if online should honor it)
    use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
    slack_fraction = cfg.slack.fraction if use_slack else 0.0
    caps_eff = np.asarray(
        effective_capacity(split_capacities(inst0.b, n), use_slack, slack_fraction),
        dtype=float,
    )
    load = np.asarray(offline_state.load, dtype=float)
    if load.ndim == 1:
        load = load.reshape((n, d))
    residual = np.maximum(0.0, caps_eff - load)

    cap_list: list[np.ndarray] = []
    costs_list: list[np.ndarray] = []
    feas_matrices: list[np.ndarray] = []
    feas_rhs: list[np.ndarray] = []
    for inst in samples:
        M_onl = len(inst.online_items)
        if M_onl <= 0:
            continue
        caps = np.asarray([item.cap_matrix for item in inst.online_items], dtype=float)
        cap_list.append(caps)
        cost_source = inst if sample_online_costs else base_instance
        costs = np.asarray(
            cost_source.costs.assignment_costs[M_off : M_off + M_onl, :],
            dtype=float,
        )
        costs_list.append(costs)
        for item in inst.online_items:
            feas_matrices.append(np.asarray(item.feas_matrix, dtype=float))
            feas_rhs.append(np.asarray(item.feas_rhs, dtype=float))

    if not cap_list:
        return np.zeros((n, d), dtype=float)

    scale = 1.0 / float(sample_count)
    cap_matrices = np.concatenate(cap_list, axis=0) * scale
    costs = np.vstack(costs_list) * scale
    data = build_offline_milp_data_from_arrays(
        cap_matrices=cap_matrices,
        costs=costs,
        feas_matrices=feas_matrices,
        feas_rhs=feas_rhs,
        b=residual.reshape(-1),
        fallback_idx=fallback_idx,
        slack_enforce=False,
        slack_fraction=0.0,
    )

    m = gp.Model("online_fractional_pricing_saa")
    m.Params.OutputFlag = 1 if log_to_console else 0
    num_vars = int(data.c.size)
    x = m.addMVar(shape=num_vars, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    constr = None
    if data.A.size:
        constr = m.addMConstr(data.A, x, "<", data.b, name="Axb")
    m.setObjective(data.c @ x, GRB.MINIMIZE)

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP not optimal, status={m.Status}")

    prices = np.zeros((n, d), dtype=float)
    if constr is None:
        return prices
    pi = np.asarray(constr.Pi, dtype=float)
    for i in range(n):
        for d_idx in range(d):
            row_idx = i * d + d_idx
            if row_idx < pi.size:
                prices[i, d_idx] = abs(float(pi[row_idx]))
    return prices


def compute_prices(
    cfg: Config,
    instance: Instance,
    offline_state: AssignmentState,
    out_path: Path,
    *,
    log_to_console: bool = False,
    sample_online_caps: bool = True,
    sample_online_costs: bool = True,
    sample_seed: int | None = None,
    num_samples: int = 1,
) -> dict[int, float | list[float]]:
    """
    1) Take the offline state and compute the residual problem.
    2) Build a fractional LP for sampled ONLINE sets (regular actions 0..n-1).
       If sample_online_caps is True, online items/feasibility are resampled using sample_seed.
       If sample_online_costs is True, online costs are resampled per sample.
       For num_samples >= 1, solve a single SAA LP with shared capacities.
    3) Minimize assignment cost; read duals Pi of cap constraints as λ_i.
    4) Save to JSON and return {action_i: lambda_i} (used by sim_base/sim_dual).
    """
    sample_count = max(1, int(num_samples))
    if not sample_online_caps:
        sample_count = 1

    samples: list[Instance] = []
    if sample_online_caps:
        for idx in range(sample_count):
            seed = None if sample_seed is None else int(sample_seed) + idx
            samples.append(
                resample_online_items(
                    cfg,
                    instance,
                    seed=seed,
                    M_onl=len(instance.online_items),
                )
            )
    else:
        samples.append(instance)
    avg = _pricing_duals_saa(
        cfg,
        samples,
        offline_state,
        instance,
        sample_online_costs=sample_online_costs,
        log_to_console=log_to_console,
    )

    n = instance.n
    d = block_dim(n, instance.m)
    prices_out: dict[int, float | list[float]] = {}
    for i in range(n):
        if d == 1:
            prices_out[i] = float(avg[i, 0])
        else:
            prices_out[i] = [float(x) for x in avg[i]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prices": prices_out}, f, indent=2)
    return prices_out
