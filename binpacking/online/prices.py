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

def _pricing_duals_saa(
    cfg: Config,
    samples: list[Instance],
    offline_state: AssignmentState,
    *,
    log_to_console: bool = False,
) -> np.ndarray:
    if not samples:
        return np.zeros((0, 0), dtype=float)
    inst0 = samples[0]
    N = len(inst0.bins)
    dims = inst0.bins[0].capacity.shape[0] if inst0.bins else 1
    if N == 0:
        return np.zeros((0, dims), dtype=float)

    sample_count = max(1, len(samples))
    fallback_idx = inst0.fallback_bin_index
    cols = N + (1 if fallback_idx >= 0 else 0)
    M_off = len(inst0.offline_items)
    allow_fallback = bool(cfg.problem.fallback_is_enabled and cfg.problem.fallback_allowed_online)

    # Effective capacities (respect slack if online should honor it)
    use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
    slack_fraction = cfg.slack.fraction if use_slack else 0.0
    caps_eff = np.asarray(
        [effective_capacity(b.capacity, use_slack, slack_fraction) for b in inst0.bins],
        dtype=float,
    )
    if caps_eff.ndim == 1:
        caps_eff = caps_eff.reshape((N, -1))
    load = np.asarray(offline_state.load, dtype=float)
    if load.ndim == 1:
        load = load.reshape((load.shape[0], -1))
    load = load[:N]
    residual = np.maximum(0.0, caps_eff - load)

    volumes_list: list[np.ndarray] = []
    costs_list: list[np.ndarray] = []
    feas_list: list[np.ndarray] = []
    for inst in samples:
        M_onl = len(inst.online_items)
        if M_onl <= 0:
            continue
        vols = np.asarray([item.volume for item in inst.online_items], dtype=float)
        if vols.ndim == 1:
            vols = vols.reshape((-1, 1))
        volumes_list.append(vols)
        costs = np.asarray(
            inst.costs.assignment_costs[M_off : M_off + M_onl, :],
            dtype=float,
        )
        costs_list.append(costs)
        if inst.online_feasible is not None and inst.online_feasible.feasible.size:
            feas = np.asarray(inst.online_feasible.feasible, dtype=int)
        else:
            feas = np.ones((M_onl, cols), dtype=int)
        if fallback_idx >= 0:
            if feas.shape[1] == N:
                fallback_val = 1 if allow_fallback else 0
                fallback_col = np.full((M_onl, 1), fallback_val, dtype=int)
                feas = np.hstack([feas, fallback_col])
            else:
                feas[:, fallback_idx] = 1 if allow_fallback else 0
        feas_list.append(feas)

    if not volumes_list:
        return np.zeros((N, dims), dtype=float)

    scale = 1.0 / float(sample_count)
    volumes = np.vstack(volumes_list) * scale
    costs = np.vstack(costs_list) * scale
    feasible = np.vstack(feas_list)

    data = build_offline_milp_data_from_arrays(
        volumes=volumes,
        costs=costs,
        feasible=feasible,
        capacities=residual,
        fallback_idx=fallback_idx,
        fallback_capacity=cfg.problem.fallback_capacity_online,
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

    prices = np.zeros((N, dims), dtype=float)
    if constr is None:
        return prices
    pi = np.asarray(constr.Pi, dtype=float)
    for i in range(N):
        for d in range(dims):
            row_idx = i * dims + d
            if row_idx < pi.size:
                prices[i, d] = abs(float(pi[row_idx]))
    return prices


def compute_prices(
    cfg: Config,
    instance: Instance,
    offline_state: AssignmentState,
    out_path: Path,
    *,
    log_to_console: bool = False,
    sample_online: bool = True,
    sample_seed: int | None = None,
    num_samples: int = 1,
) -> dict[int, float | list[float]]:
    """
    1) Take the offline state and compute the residual problem.
    2) Build a fractional LP for sampled ONLINE sets (regular bins 0..N-1).
       If sample_online is True, online items are resampled using sample_seed.
       For num_samples >= 1, solve a single SAA LP with shared capacities.
    3) Minimize assignment cost; read duals Pi of cap constraints as λ_i.
    4) Save to JSON and return {bin_i: lambda_i} (used by sim_base/sim_dual).
    """
    sample_count = max(1, int(num_samples))
    if not sample_online:
        sample_count = 1

    samples: list[Instance] = []
    if sample_online:
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
        log_to_console=log_to_console,
    )

    N = len(instance.bins)
    dims = instance.bins[0].capacity.shape[0] if instance.bins else 1
    prices_out: dict[int, float | list[float]] = {}
    for i in range(N):
        if dims == 1:
            prices_out[i] = float(avg[i, 0])
        else:
            prices_out[i] = [float(x) for x in avg[i]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prices": prices_out}, f, indent=2)
    return prices_out
