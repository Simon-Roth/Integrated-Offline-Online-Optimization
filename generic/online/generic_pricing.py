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
    base_instance: Instance,
    *,
    sample_online_costs: bool,
    log_to_console: bool = True,
) -> np.ndarray:
    if not samples:
        return np.zeros((0,), dtype=float)
    inst0 = samples[0]
    n = inst0.n
    m = inst0.m
    if m == 0:
        return np.zeros((0,), dtype=float)

    fallback_idx = inst0.fallback_action_index
    M_off = len(inst0.offline_items)

    use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
    slack_fraction = cfg.slack.fraction if use_slack else 0.0
    b_eff = effective_capacity(np.asarray(inst0.b, dtype=float), use_slack, slack_fraction)
    load = np.asarray(offline_state.load, dtype=float).reshape(-1)
    residual = np.maximum(0.0, b_eff - load)

    def _solve_sample(inst: Instance) -> np.ndarray | None:
        M_onl = len(inst.online_items)
        if M_onl <= 0:
            return np.zeros((m,), dtype=float)
        caps = np.asarray([item.cap_matrix for item in inst.online_items], dtype=float)
        cost_source = inst if sample_online_costs else base_instance
        costs = np.asarray(
            cost_source.costs.assignment_costs[M_off : M_off + M_onl, :],
            dtype=float,
        )
        feas_matrices = [np.asarray(item.feas_matrix, dtype=float) for item in inst.online_items]
        feas_rhs = [np.asarray(item.feas_rhs, dtype=float) for item in inst.online_items]
        data = build_offline_milp_data_from_arrays(
            cap_matrices=caps,
            costs=costs,
            feas_matrices=feas_matrices,
            feas_rhs=feas_rhs,
            b=residual,
            fallback_idx=fallback_idx,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        model = gp.Model("generic_fractional_pricing_sample")
        model.Params.OutputFlag = 1 if log_to_console else 0
        num_vars = int(data.c.size)
        x = model.addMVar(shape=num_vars, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        constr = None
        if data.A.size:
            constr = model.addMConstr(data.A, x, "<", data.b, name="Axb")
        model.setObjective(data.c @ x, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return None
        if constr is None:
            return np.zeros((m,), dtype=float)
        pi = np.asarray(constr.Pi, dtype=float)
        prices = np.zeros((m,), dtype=float)
        take = min(m, pi.size)
        prices[:take] = np.abs(pi[:take])
        return prices

    prices_list: list[np.ndarray] = []
    skipped = 0
    for inst in samples:
        sample_prices = _solve_sample(inst)
        if sample_prices is None:
            skipped += 1
            continue
        prices_list.append(sample_prices)

    if not prices_list:
        if log_to_console:
            print("Pricing: all samples infeasible, returning zero prices.")
        return np.zeros((m,), dtype=float)
    if skipped and log_to_console:
        print(f"Pricing: skipped {skipped} infeasible samples.")
    return np.mean(prices_list, axis=0)


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
) -> dict[int, list[float]]:
    """
    Compute generic dual prices (capacity constraints only) via an SAA LP.
    Prices are returned per action, each containing the same lambda vector (length m).
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

    prices_out: dict[int, list[float]] = {}
    for i in range(instance.n):
        prices_out[i] = [float(x) for x in avg]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prices": prices_out}, f, indent=2)
    return prices_out
