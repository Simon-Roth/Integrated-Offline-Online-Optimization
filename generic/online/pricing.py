from __future__ import annotations

import copy

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from generic.core.config import Config
from generic.data.instance_generators import BaseInstanceGenerator
from generic.core.utils import effective_capacity, feasible_option_indices
from generic.core.models import AssignmentState, Instance, StepSpec


def _build_sampling_lp_sparse(
    cfg: Config,
    samples: list[Instance],
    base_instance: Instance,
    *,
    residual: np.ndarray,
) -> tuple[gp.Model, list[gp.Constr]]:
    if not samples:
        return gp.Model(), []

    inst0 = samples[0]
    n = int(inst0.n)
    T_off = len(inst0.offline_steps)
    m = int(np.asarray(residual, dtype=float).reshape(-1).size)

    scale = 1.0 / float(max(1, len(samples)))
    allow_future_online_costs = bool(cfg.costs.observe_future_online_costs)

    model = gp.Model("pricing_fractional_sampling_sparse")
    cap_expr = [gp.LinExpr() for _ in range(m)]

    for sample_idx, inst in enumerate(samples):
        T_onl = len(inst.online_steps)
        if T_onl <= 0:
            continue
        cost_source = base_instance if allow_future_online_costs else inst
        costs = np.asarray(
            cost_source.costs.assignment_costs[T_off : T_off + T_onl, :],
            dtype=float,
        )
        if costs.shape[0] != T_onl:
            raise ValueError(
                f"Pricing costs row mismatch for sample {sample_idx}: "
                f"expected {T_onl}, got {costs.shape[0]}."
            )

        for step_pos, step in enumerate(inst.online_steps):
            step_cap = np.asarray(step.cap_matrix, dtype=float)
            feasible_opts = feasible_option_indices(step.feas_matrix, step.feas_rhs)
            if not feasible_opts:
                raise RuntimeError(
                    f"No feasible options in pricing sample for step {step.step_id}."
                )

            step_vars: list[gp.Var] = []
            for option_id in feasible_opts:
                if option_id >= costs.shape[1]:
                    raise ValueError(
                        f"Pricing cost column mismatch for option {option_id} "
                        f"(available cols={costs.shape[1]})."
                    )
                var = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    obj=float(costs[step_pos, option_id]) * scale,
                    name=f"x_s{sample_idx}_t{step.step_id}_o{option_id}",
                )
                step_vars.append(var)
                if 0 <= option_id < n:
                    for r in range(m):
                        cap_expr[r].addTerms(float(step_cap[r, option_id]) * scale, var)

            model.addConstr(
                gp.quicksum(step_vars) == 1.0,
                name=f"assign_s{sample_idx}_t{step.step_id}",
            )

    cap_constr = [
        model.addConstr(cap_expr[r] <= float(residual[r]), name=f"cap_{r}")
        for r in range(m)
    ]
    model.ModelSense = GRB.MINIMIZE
    return model, cap_constr


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


def compute_resource_prices(
    cfg: Config,
    instance: Instance,
    offline_state: AssignmentState,
    *,
    log_to_console: bool = False,
    sample_online_caps: bool = True,
    sample_seed: int | None = None,
    num_samples: int = 1,
) -> np.ndarray:
    """
    Compute dual prices via a sampled fractional LP.
    - If sample_online_caps is True, online steps/feasibility are resampled using sample_seed.
    - If cfg.costs.observe_future_online_costs is False, pricing uses sampled online costs.
      If True, pricing uses realized online costs from the base instance.
    - For num_samples >= 1, solve a single averaged LP with shared capacities.
    """
    sample_count = max(1, int(num_samples))
    observe_future_online_costs = bool(cfg.costs.observe_future_online_costs)
    pricing_keep_fallback = bool(
        getattr(cfg.pricing_sim, "fallback_allowed_online_for_pricing", True)
    )

    # If future realized costs are hidden, pricing must use sampled scenarios.
    use_sampled_online_caps = bool(sample_online_caps) or (not observe_future_online_costs)
    if not use_sampled_online_caps:
        sample_count = 1
    pricing_cfg = cfg
    # Ensure fallback remains available in the pricing LP, even if disabled for online execution 
    # We do this so if a small subset of samples is infeasible, the whole pricing is not invalid. 
    # Another approach would be to average duals only over feasible lps instead of this SAA approach
    if pricing_keep_fallback and not cfg.problem.fallback_allowed_online:
        pricing_cfg = copy.deepcopy(cfg)
        pricing_cfg.problem.fallback_allowed_online = True
    pricing_instance = instance
    if len(instance.online_steps) == 0:
        if log_to_console:
            print("Pricing: skipped (no online steps).")
        return np.zeros((instance.m,), dtype=float)

    generator = BaseInstanceGenerator.from_config(pricing_cfg)
    samples: list[Instance] = []
    if use_sampled_online_caps:
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
        if pricing_keep_fallback and not cfg.problem.fallback_allowed_online:
            pricing_instance = _allow_fallback_online(instance)
        samples.append(pricing_instance)

    mode = getattr(getattr(cfg, "generation", None), "generator", "generic")
    inst0 = samples[0] if samples else instance
    n = inst0.n
    m = inst0.m
    if mode == "bgap":
        from bgap.core.block_utils import block_dim, split_capacities

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

    model, cap_constr = _build_sampling_lp_sparse(
        pricing_cfg,
        samples,
        instance,
        residual=residual,
    )
    model.Params.OutputFlag = 1 if log_to_console else 0
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"LP not optimal, status={model.Status}")

    pi = np.asarray([float(con.Pi) for con in cap_constr], dtype=float)
    prices = np.zeros((m,), dtype=float)
    take = min(m, pi.size)
    prices[:take] = np.maximum(0.0, -pi[:take])
    return prices
