from __future__ import annotations

from pathlib import Path
import copy
import json

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from generic.core.config import Config
from generic.data.instance_generators import BaseInstanceGenerator
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.core.utils import effective_capacity
from generic.core.models import AssignmentState, Instance, StepSpec


def _build_sampling_lp(
    cfg: Config,
    samples: list[Instance],
    base_instance: Instance,
    *,
    sample_online_costs: bool,
    residual: np.ndarray,
) -> tuple[gp.Model, gp.MConstr | None]:
    if not samples:
        return gp.Model(), None
    inst0 = samples[0]
    fallback_idx = inst0.fallback_option_index
    T_off = len(inst0.offline_steps)

    cap_list: list[np.ndarray] = []
    costs_list: list[np.ndarray] = []
    feas_matrices: list[np.ndarray] = []
    feas_rhs: list[np.ndarray] = []
    for inst in samples:
        T_onl = len(inst.online_steps)
        if T_onl <= 0:
            continue
        caps = np.asarray([step.cap_matrix for step in inst.online_steps], dtype=float)
        cap_list.append(caps)
        cost_source = inst if sample_online_costs else base_instance
        costs = np.asarray(
            cost_source.costs.assignment_costs[T_off : T_off + T_onl, :],
            dtype=float,
        )
        costs_list.append(costs)
        for step in inst.online_steps:
            feas_matrices.append(np.asarray(step.feas_matrix, dtype=float))
            feas_rhs.append(np.asarray(step.feas_rhs, dtype=float))

    if not cap_list:
        return gp.Model(), None

    scale = 1.0 / float(max(1, len(samples)))
    cap_matrices = np.concatenate(cap_list, axis=0) * scale
    costs = np.vstack(costs_list) * scale
    data = build_offline_milp_data_from_arrays(
        cap_matrices=cap_matrices,
        costs=costs,
        feas_matrices=feas_matrices,
        feas_rhs=feas_rhs,
        b=residual,
        fallback_idx=fallback_idx,
        slack_enforce=False,
        slack_fraction=0.0,
    )

    model = gp.Model("pricing_fractional_sampling")
    x = model.addMVar(shape=int(data.c.size), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    constr = None
    if data.A.size:
        constr = model.addMConstr(data.A, x, "<", data.b, name="Axb")
    model.setObjective(data.c @ x, GRB.MINIMIZE)
    return model, constr


def _allow_fallback_online(instance: Instance) -> Instance:
    """
    Return a shallow copy of `instance` whose online feasibility constraints
    allow fallback (if fallback exists). Capacity matrices and costs are preserved.
    """
    fallback_idx = int(instance.fallback_option_index)
    if fallback_idx < 0 or not instance.online_steps:
        return instance

    online_steps: list[StepSpec] = []
    for step in instance.online_steps:
        A = np.asarray(step.feas_matrix, dtype=float)
        b = np.asarray(step.feas_rhs, dtype=float).reshape(-1)
        if A.ndim != 2 or b.size != A.shape[0]:
            online_steps.append(step)
            continue
        if A.shape[1] <= fallback_idx:
            online_steps.append(step)
            continue

        row_sum = np.sum(np.abs(A), axis=1)
        is_fallback_row = (b == 0.0) & (A[:, fallback_idx] == 1.0) & (row_sum == 1.0)
        if np.any(is_fallback_row):
            A = A[~is_fallback_row]
            b = b[~is_fallback_row]

        online_steps.append(
            StepSpec(
                step_id=step.step_id,
                cap_matrix=step.cap_matrix,
                feas_matrix=A,
                feas_rhs=b,
            )
        )

    return Instance(
        n=instance.n,
        m=instance.m,
        b=instance.b,
        offline_steps=instance.offline_steps,
        costs=instance.costs,
        fallback_option_index=instance.fallback_option_index,
        online_steps=online_steps,
    )


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
    Compute dual prices via a sampled fractional LP.
    - If sample_online_caps is True, online steps/feasibility are resampled using sample_seed.
    - If sample_online_costs is True, online costs are resampled per sample.
    - For num_samples >= 1, solve a single averaged LP with shared capacities.
    """
    sample_count = max(1, int(num_samples))
    if not sample_online_caps:
        sample_count = 1
    pricing_cfg = cfg
    # This is for purely online scenarios, so simdual is not just like cabf with 0 prices. but can be removed. 
    if not cfg.problem.fallback_allowed_online:
        pricing_cfg = copy.deepcopy(cfg)
        pricing_cfg.problem.fallback_allowed_online = True
    pricing_instance = instance
    if len(instance.online_steps) == 0:
        if log_to_console:
            print("Pricing: skipped (no online steps).")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"prices": {}}, f, indent=2)
        return {}

    generator = BaseInstanceGenerator.from_config(pricing_cfg)
    samples: list[Instance] = []
    if sample_online_caps:
        for idx in range(sample_count):
            seed = None if sample_seed is None else int(sample_seed) + idx
            samples.append(
                generator.resample_online_phase(
                    pricing_cfg,
                    instance,
                    seed=seed,
                    T_onl=len(instance.online_steps),
                )
            )
    else:
        if not cfg.problem.fallback_allowed_online:
            pricing_instance = _allow_fallback_online(instance)
        samples.append(pricing_instance)

    mode = getattr(getattr(cfg, "generation", None), "generator", "generic")
    inst0 = samples[0] if samples else instance
    n = inst0.n
    m = inst0.m
    if mode == "binpacking":
        from binpacking.core.block_utils import block_dim, split_capacities

        d = block_dim(n, m)
        use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
        slack_fraction = cfg.slack.fraction if use_slack else 0.0
        caps_eff = np.asarray(
            effective_capacity(split_capacities(inst0.b, n), use_slack, slack_fraction),
            dtype=float,
        )
        load = np.asarray(offline_state.load, dtype=float)
        if load.ndim == 1:
            load = load.reshape((n, d))
        residual = np.maximum(0.0, caps_eff - load).reshape(-1)
    else:
        use_slack = cfg.slack.enforce_slack and getattr(cfg.slack, "apply_to_online", True)
        slack_fraction = cfg.slack.fraction if use_slack else 0.0
        b_eff = effective_capacity(np.asarray(inst0.b, dtype=float), use_slack, slack_fraction)
        load = np.asarray(offline_state.load, dtype=float).reshape(-1)
        residual = np.maximum(0.0, b_eff - load)

    model, constr = _build_sampling_lp(
        pricing_cfg,
        samples,
        instance,
        sample_online_costs=sample_online_costs,
        residual=residual,
    )
    model.Params.OutputFlag = 1 if log_to_console else 0
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP not optimal, status={model.Status}")

    pi = np.asarray(constr.Pi, dtype=float) if constr is not None else np.zeros((0,), dtype=float)
    prices_out: dict[int, float | list[float]] = {}
    if mode == "binpacking":
        from binpacking.core.block_utils import block_dim

        d = block_dim(n, m)
        for i in range(n):
            if d == 1:
                prices_out[i] = float(abs(pi[i])) if i < pi.size else 0.0
            else:
                start = i * d
                end = start + d
                vals = np.abs(pi[start:end]).tolist()
                if len(vals) < d:
                    vals += [0.0] * (d - len(vals))
                prices_out[i] = [float(x) for x in vals]
    else:
        prices = np.zeros((m,), dtype=float)
        take = min(m, pi.size)
        prices[:take] = np.abs(pi[:take])
        prices_out = {i: [float(x) for x in prices] for i in range(n)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prices": prices_out}, f, indent=2)
    return prices_out
