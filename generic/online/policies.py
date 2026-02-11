from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence
import json
import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from generic.core.config import Config
from generic.data.instance_generators import BaseInstanceGenerator
from generic.data.generator_utils import (
    _ensure_row_feasible,
    _build_feas_constraints,
    _sample_feas_mask_by_option,
)
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.core.utils import (
    option_is_feasible,
    effective_capacity,
    feasible_option_indices,
    make_rng,
    scalarize_vector,
)
from generic.core.models import AssignmentState, Instance, StepSpec, Decision
from generic.online.policy_utils import (
    current_cost_row,
    lookup_assignment_cost,
    PolicyInfeasibleError,
    remaining_capacities,
)


class BaseOnlinePolicy(Protocol):
    """
    Interface that all online heuristics/solvers must implement.
    """

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        """
        Decide how to act on the arriving step given the current assignment state.

        Parameters
        ----------
        step:
            The arriving online step (with its A_t^{cap}, A_t^{feas}, b_t).
        state:
            Current assignment state (will be mutated by the solver after the decision).
        instance:
            The full problem instance (offline + online data).

        Returns
        -------
        Decision:
            Encodes option selection, evictions, reassigned_offline_steps, and incremental cost.
        """
        ...


__all__ = [
    "BaseOnlinePolicy",
    "PolicyInfeasibleError",
    "GenericDynamicLearningPolicy",
    "PrimalDualPolicy",
    "SimDualPolicy",
    "RollingHorizonMILPPolicy",
]


class PrimalDualPolicy(BaseOnlinePolicy):
    """
    Online primal-dual MILP policy (price-aware MILP per arrival).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        time_limit: int = 60,
        mip_gap: float = 0.01,
        threads: int = 0,
        log_to_console: bool = False,
    ) -> None:
        self.cfg = cfg
        from generic.offline.solver import OfflineMILPSolver

        self._solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )
        self._instance_id: Optional[int] = None
        self._lambda: Optional[np.ndarray] = None
        self._cap_per_step: Optional[np.ndarray] = None
        self._effective_caps: Optional[np.ndarray] = None
        self._cost_scale: float = 1.0
        self._eta_scale: float = 1.0
        self._total_steps: int = 0
        self._t: int = 0

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        # Step 1-2: initialize duals and remaining capacities once per instance.
        self._reset_if_needed(instance, state)
        assert self._lambda is not None
        assert self._cap_per_step is not None
        assert self._effective_caps is not None

        # Step 3: observe A_t^{cap}, feasibility, and c_t (assignment costs).
        n = instance.n
        fallback_idx = instance.fallback_option_index
        cols = n + (1 if fallback_idx >= 0 else 0)
        cap_matrix = np.asarray(step.cap_matrix, dtype=float)
        feas_matrix = np.asarray(step.feas_matrix, dtype=float)
        feas_rhs = np.asarray(step.feas_rhs, dtype=float)
        costs = current_cost_row(self.cfg, instance, step.step_id, cols).reshape(1, -1)

        # Step 4: solve the price-aware MILP for this single arrival.
        capacities = remaining_capacities(
            self.cfg,
            state,
            instance,
            self._effective_caps,
        )
        target = self._current_target(capacities)
        costs = self._price_aware_costs(costs, cap_matrix, n, target)
        assigned_option = self._solve_price_aware_milp(
            cap_matrix,
            costs,
            feas_matrix,
            feas_rhs,
            capacities,
            fallback_idx,
        )

        # Step 5-7: apply x_t (the OnlineSolver updates state/load), then update lambda.
        self._update_duals(assigned_option, cap_matrix, n, capacities)

        incremental_cost = lookup_assignment_cost(self.cfg, instance, step.step_id, assigned_option)
        return Decision(
            placed_step=(step.step_id, int(assigned_option)),
            evicted_offline_steps=[],
            reassigned_offline_steps=[],
            incremental_cost=incremental_cost,
        )

    # Initialize per-instance state (lambda, caps, scaling).
    def _reset_if_needed(self, instance: Instance, state: AssignmentState) -> None:
        # Reset per-instance state so prices/counters don't leak across runs.
        if self._instance_id == id(instance) and self._lambda is not None:
            return
        self._instance_id = id(instance)
        use_slack = bool(self.cfg.slack.enforce_slack and self.cfg.slack.apply_to_online)
        slack_fraction = self.cfg.slack.fraction if use_slack else 0.0
        caps_eff = np.asarray(
            effective_capacity(instance.b, use_slack, slack_fraction),
            dtype=float,
        ).reshape(-1)
        T = max(1, len(instance.online_steps))
        self._effective_caps = caps_eff
        self._cap_per_step = caps_eff / float(T)
        self._total_steps = T
        self._cost_scale = self._compute_cost_scale(instance, instance.n)
        self._eta_scale = 1.0 / self._cost_scale if self.cfg.primal_dual.normalize_costs else 1.0
        self._t = 0

        load = np.asarray(state.load, dtype=float).reshape(-1)
        denom = np.where(caps_eff > 0.0, caps_eff, 1.0)
        utilization = np.maximum(0.0, load / denom)
        eta0 = self._eta_t() * self._eta_scale
        self._lambda = eta0 * utilization

    # Apply optional cost scaling and add the price term to costs.
    def _price_aware_costs(
        self,
        costs: np.ndarray,
        cap_matrix: np.ndarray,
        n: int,
        target: np.ndarray,
    ) -> np.ndarray:
        if self.cfg.primal_dual.normalize_costs:
            costs = costs / self._cost_scale
        if n:
            price_terms = self._price_terms(cap_matrix, target)
            costs[0, :n] = costs[0, :n] + price_terms
        return costs

    # Compute lambda^T A_t^cap term (optionally normalized).
    def _price_terms(self, cap_matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
        cap = np.asarray(cap_matrix, dtype=float)
        if self.cfg.primal_dual.normalize_update:
            denom = np.where(target > 1.0, target, 1.0)
            scaled = cap / denom.reshape(-1, 1)
            return scaled.T @ self._lambda
        return cap.T @ self._lambda

    # Solve the one-step price-aware MILP and return the chosen option.
    def _solve_price_aware_milp(
        self,
        cap_matrix: np.ndarray,
        costs: np.ndarray,
        feas_matrix: np.ndarray,
        feas_rhs: np.ndarray,
        capacities: np.ndarray,
        fallback_idx: int,
    ) -> int:
        data = build_offline_milp_data_from_arrays(
            cap_matrices=cap_matrix.reshape(1, *cap_matrix.shape),
            costs=costs,
            feas_matrices=[feas_matrix],
            feas_rhs=[feas_rhs],
            b=capacities,
            fallback_idx=fallback_idx,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._solver.solve_from_data(data)
        if not rh_info.feasible:
            raise PolicyInfeasibleError("Primal-dual MILP has no feasible solution.")
        assigned_option = rh_state.assigned_option.get(0)
        if assigned_option is None:
            raise PolicyInfeasibleError("Primal-dual MILP did not assign the current step.")
        return int(assigned_option)

    # Update dual prices after choosing x_t.
    def _update_duals(
        self,
        assigned_option: int,
        cap_matrix: np.ndarray,
        n: int,
        remaining_caps: np.ndarray,
    ) -> None:
        eta_t = self._eta_t() * self._eta_scale
        usage = np.zeros_like(self._lambda)
        if 0 <= assigned_option < n:
            usage = cap_matrix[:, assigned_option]
        target = self._current_target(remaining_caps)
        if self.cfg.primal_dual.normalize_update:
            denom = np.where(target > 1.0, target, 1.0)
            delta = (usage - target) / denom
        else:
            delta = usage - target
        self._lambda = np.maximum(self._lambda + eta_t * delta, 0.0)
        self._t += 1

    def _current_target(self, remaining_caps: np.ndarray) -> np.ndarray:
        if self.cfg.primal_dual.use_remaining_capacity_target:
            steps_left = max(1, self._total_steps - self._t)
            return remaining_caps.reshape(-1) / float(steps_left)
        return self._cap_per_step

    # Step-size schedule for the dual update.
    def _eta_t(self) -> float:
        mode = self.cfg.primal_dual.eta_mode.lower()
        eta0 = float(self.cfg.primal_dual.eta0)
        decay = float(self.cfg.primal_dual.eta_decay)
        eta_min = float(self.cfg.primal_dual.eta_min)
        t = self._t + 1
        if mode == "constant":
            return eta0
        if mode == "sqrt":
            return eta0 / float(np.sqrt(t))
        if mode == "linear":
            return max(eta_min, eta0 - decay * float(t - 1))
        if mode == "exponential":
            return max(eta_min, eta0 * float(np.exp(-decay * float(t - 1))))
        raise ValueError(f"Unknown eta_mode: {self.cfg.primal_dual.eta_mode}")

    # Compute the scale used to normalize costs.
    def _compute_cost_scale(self, instance: Instance, n: int) -> float:
        min_scale = float(self.cfg.primal_dual.cost_scale_min)
        mode = self.cfg.primal_dual.cost_scale_mode.lower()
        if mode == "assign_bounds":
            lo, hi = self.cfg.costs.assign_bounds
            return max(min_scale, 0.5 * (float(lo) + float(hi)))
        if mode == "assign_mean":
            costs = instance.costs.assignment_costs
            if costs is None or not costs.size or n <= 0:
                return max(min_scale, 1.0)
        online_step_ids = [step_spec.step_id for step_spec in instance.online_steps]
        if not online_step_ids:
            return max(min_scale, 1.0)
        values = costs[np.asarray(online_step_ids, dtype=int), :n].reshape(-1)
        if values.size == 0:
            return max(min_scale, 1.0)
        return max(min_scale, float(np.mean(values)))
        raise ValueError(f"Unknown cost_scale_mode: {self.cfg.primal_dual.cost_scale_mode}")


class SimDualPolicy(BaseOnlinePolicy):
    """
    Price-aware MILP policy with fixed dual prices (no online updates).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        price_path: Path = Path("outputs/generic/results/sim_dual.json"),
        time_limit: int = 60,
        mip_gap: float = 0.01,
        threads: int = 0,
        log_to_console: bool = False,
    ) -> None:
        self.cfg = cfg
        self._price_path = price_path
        self._lambda = self._load_prices(price_path)
        from generic.offline.solver import OfflineMILPSolver

        self._solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        # Step 1: observe A_t^{cap}, feasibility, and c_t (assignment costs)
        n = instance.n
        fallback_idx = instance.fallback_option_index
        cols = n + (1 if fallback_idx >= 0 else 0)
        cap_matrix = np.asarray(step.cap_matrix, dtype=float)
        feas_matrix = np.asarray(step.feas_matrix, dtype=float)
        feas_rhs = np.asarray(step.feas_rhs, dtype=float)
        costs = current_cost_row(self.cfg, instance, step.step_id, cols).reshape(1, -1)

        # Step 2: add fixed dual prices to the objective and solve the one-step MILP.
        capacities = remaining_capacities(self.cfg, state, instance)
        costs = self._price_aware_costs(costs, cap_matrix, n)
        assigned_option = self._solve_price_aware_milp(
            cap_matrix,
            costs,
            feas_matrix,
            feas_rhs,
            capacities,
            fallback_idx,
        )

        incremental_cost = lookup_assignment_cost(self.cfg, instance, step.step_id, assigned_option)
        return Decision(
            placed_step=(step.step_id, int(assigned_option)),
            evicted_offline_steps=[],
            reassigned_offline_steps=[],
            incremental_cost=incremental_cost,
        )

    # Load per-option dual prices from JSON.
    def _load_prices(self, price_path: Path) -> dict[int, np.ndarray]:
        with open(price_path) as handle:
            data = json.load(handle)
        prices = {}
        for key, value in data.get("prices", {}).items():
            if isinstance(value, list):
                prices[int(key)] = np.asarray(value, dtype=float)
            else:
                prices[int(key)] = np.asarray([float(value)], dtype=float)
        return prices

    # Add lambda^T A_t^cap to assignment costs (regular options only).
    def _price_aware_costs(self, costs: np.ndarray, cap_matrix: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return costs
        cap = np.asarray(cap_matrix, dtype=float)
        price_terms = np.zeros((n,), dtype=float)
        for option_id in range(n):
            lam = self._lambda.get(option_id)
            if lam is None:
                continue
            lam_vec = lam.reshape(-1)
            col = cap[:, option_id].reshape(-1)
            if lam_vec.size == 1 and col.size > 1:
                lam_vec = np.full((col.size,), float(lam_vec[0]))
            if lam_vec.size != col.size:
                raise ValueError(
                    f"Price vector dimension mismatch for option {option_id}: "
                    f"{lam_vec.size} vs {col.size}"
                )
            price_terms[option_id] = float(np.dot(lam_vec, col))
        costs[0, :n] = costs[0, :n] + price_terms
        return costs

    # Solve the one-step price-aware MILP and return the chosen option.
    def _solve_price_aware_milp(
        self,
        cap_matrix: np.ndarray,
        costs: np.ndarray,
        feas_matrix: np.ndarray,
        feas_rhs: np.ndarray,
        capacities: np.ndarray,
        fallback_idx: int,
    ) -> int:
        data = build_offline_milp_data_from_arrays(
            cap_matrices=cap_matrix.reshape(1, *cap_matrix.shape),
            costs=costs,
            feas_matrices=[feas_matrix],
            feas_rhs=[feas_rhs],
            b=capacities,
            fallback_idx=fallback_idx,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._solver.solve_from_data(data)
        if not rh_info.feasible:
            raise PolicyInfeasibleError("SimDual MILP has no feasible solution.")
        assigned_option = rh_state.assigned_option.get(0)
        if assigned_option is None:
            raise PolicyInfeasibleError("SimDual MILP did not assign the current step.")
        return int(assigned_option)


class RollingHorizonMILPPolicy(BaseOnlinePolicy):
    """
    Rolling horizon MILP policy for the online phase.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        time_limit: int = 60,
        mip_gap: float = 0.01,
        threads: int = 0,
        log_to_console: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self._rng = make_rng(seed)
        self._generator = BaseInstanceGenerator.from_config(cfg)
        from generic.offline.solver import OfflineMILPSolver

        self._solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        # Step 1: observe A_t and remaining capacity (from state).
        idx = self._online_index(step, instance)
        remaining = max(0, len(instance.online_steps) - idx - 1)

        n = instance.n
        fallback_idx = instance.fallback_option_index
        cols = n + (1 if fallback_idx >= 0 else 0)

        # Step 2: sample future A_tau for tau > t and build the remaining MILP.
        cap_matrices = self._build_cap_matrices(step, instance, remaining)
        costs = self._build_costs(instance, step.step_id, remaining, n, cols, online_index=idx)
        feas_matrices, feas_rhs = self._build_feasible(
            step,
            remaining,
            n,
            cols,
            fallback_idx,
        )
        capacities = remaining_capacities(self.cfg, state, instance)

        # Step 3: solve the remaining-horizon MILP.
        data = build_offline_milp_data_from_arrays(
            cap_matrices=cap_matrices,
            costs=costs,
            feas_matrices=feas_matrices,
            feas_rhs=feas_rhs,
            b=capacities,
            fallback_idx=fallback_idx,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._solver.solve_from_data(data)
        if not rh_info.feasible:
            raise PolicyInfeasibleError("Rolling horizon MILP has no feasible solution.")

        assigned_option = rh_state.assigned_option.get(0)
        if assigned_option is None:
            raise PolicyInfeasibleError("Rolling horizon MILP did not assign the current step.")
        # Step 4-5: fix x_t (current step), OnlineSolver updates remaining capacity after applying.
        incremental_cost = lookup_assignment_cost(self.cfg, instance, step.step_id, assigned_option)
        return Decision(
            placed_step=(step.step_id, int(assigned_option)),
            evicted_offline_steps=[],
            reassigned_offline_steps=[],
            incremental_cost=incremental_cost,
        )

    # Map online step id to its index in the arrival sequence.
    def _online_index(self, step: StepSpec, instance: Instance) -> int:
        base = len(instance.offline_steps)
        idx = step.step_id - base
        if idx < 0 or idx >= len(instance.online_steps):
            raise KeyError(f"Online step id {step.step_id} is out of expected range.")
        if instance.online_steps[idx].step_id != step.step_id:
            raise KeyError(
                f"Online step id {step.step_id} does not match instance ordering."
            )
        return idx

    # Stack current step A_t^{cap} with sampled future A_tau^{cap}.
    def _build_cap_matrices(self, step: StepSpec, instance: Instance, remaining: int) -> np.ndarray:
        current = np.asarray(step.cap_matrix, dtype=float).reshape(1, *step.cap_matrix.shape)
        if remaining <= 0:
            return current
        future = self._generator.sample_cap_matrices(
            self.cfg,
            self._rng,
            remaining,
            instance.n,
            instance.m,
            phase="online",
        )
        return np.vstack([current, future])

    # Build assignment costs for current + sampled steps.
    def _build_costs(
        self,
        instance: Instance,
        step_id: int,
        remaining: int,
        n: int,
        cols: int,
        *,
        online_index: int,
    ) -> np.ndarray:
        costs = np.zeros((1 + remaining, cols), dtype=float)
        costs[0] = current_cost_row(self.cfg, instance, step_id, cols)
        if remaining <= 0:
            return costs
        future_steps = instance.online_steps[online_index + 1 : online_index + 1 + remaining]
        for offset, future_step in enumerate(future_steps, start=1):
            costs[offset] = current_cost_row(self.cfg, instance, future_step.step_id, cols)
        return costs

    # Build feasibility rows for current + sampled steps.
    def _build_feasible(
        self,
        step: StepSpec,
        remaining: int,
        n: int,
        cols: int,
        fallback_idx: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        current_mats = [np.asarray(step.feas_matrix, dtype=float)]
        current_rhs = [np.asarray(step.feas_rhs, dtype=float)]
        if remaining <= 0:
            return current_mats, current_rhs
        base_mask = _sample_feas_mask_by_option(self.cfg, self._rng, remaining, n, phase="online")
        _ensure_row_feasible(base_mask, self._rng)
        future_mats: list[np.ndarray] = []
        future_rhs: list[np.ndarray] = []
        for idx in range(remaining):
            feas_matrix, feas_rhs = _build_feas_constraints(
                base_mask[idx],
                fallback_allowed=self.cfg.problem.fallback_allowed_online,
                fallback_idx=fallback_idx,
                cols=cols,
            )
            future_mats.append(feas_matrix)
            future_rhs.append(feas_rhs)
        return current_mats + future_mats, current_rhs + future_rhs


class GenericDynamicLearningPolicy(BaseOnlinePolicy):
    """
    Generic dynamic price-updating policy (no evictions).
    """

    def __init__(self, cfg: Config, price_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.price_path = Path(price_path) if price_path else None

        # Runtime state
        self._phase_idx: int = -1
        self._schedule: List[int] = []
        self._current_prices: Optional[np.ndarray] = None  # length m
        self._processed: int = 0  # number of online steps already placed
        self._seen_steps: List[StepSpec] = []
        self._log_records: List[Dict[str, object]] = []
        from generic.offline.solver import OfflineMILPSolver

        self._milp_solver = OfflineMILPSolver(cfg)

    def select_action(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Decision:
        # Lazily build phase schedule on the first call.
        if not self._schedule:
            self._schedule = self._build_schedule(len(instance.online_steps))

        # Update prices if a phase boundary was reached after the last step.
        self._maybe_update_prices(state, instance)

        candidate_options = self._candidate_options(step, instance)
        if not candidate_options:
            raise PolicyInfeasibleError(
                f"No feasible regular option for online step {step.step_id}"
            )

        # First, try to place without evictions via a one-step MILP (regular options only).
        milp_decision = self._milp_no_eviction(step, state, instance)
        if milp_decision is not None:
            self._mark_processed(step)
            return milp_decision

        # No feasible regular option (no evictions in generic policy).
        raise PolicyInfeasibleError(
            f"GenericDynamicLearningPolicy could not place step {step.step_id}"
        )

    # ------------------------------------------------------------------
    # Pricing + scheduling
    # ------------------------------------------------------------------
    def _build_schedule(self, horizon: int) -> List[int]:
        """Geometric schedule t_k = ε * T_onl * 2^k (rounded to ints)."""
        eps = max(self.cfg.dla.epsilon, 1e-9)
        min_len = max(1, int(self.cfg.dla.min_phase_len))
        schedule: List[int] = []
        base = math.ceil(eps * horizon)
        t_k = max(min_len, base)
        while t_k < horizon:
            if schedule and t_k <= schedule[-1]:
                t_k = schedule[-1] + min_len
            schedule.append(min(t_k, horizon))
            t_k = math.ceil(t_k * 2)
        if schedule and schedule[-1] > horizon:
            schedule[-1] = horizon
        if not schedule or schedule[-1] < horizon:
            schedule.append(horizon)
        return schedule

    def _maybe_update_prices(self, state: AssignmentState, instance: Instance) -> None:
        """Update prices when we've completed a phase (after placing t_k steps)."""
        if self._processed == 0:
            return
        if self._phase_idx + 1 >= len(self._schedule):
            return
        next_boundary = self._schedule[self._phase_idx + 1]
        if self._processed < next_boundary:
            return

        # Compute new prices using observed steps up to t_k = next_boundary.
        t_k = next_boundary
        horizon = max(1, len(instance.online_steps))
        h_k = self.cfg.dla.epsilon * math.sqrt(horizon / float(t_k))
        self._current_prices = self._compute_prices(instance, state, t_k, h_k)
        self._phase_idx += 1
        self._log_phase(t_k, h_k)

    def _compute_prices(
        self,
        instance: Instance,
        state: AssignmentState,
        t_k: int,
        h_k: float,
    ) -> np.ndarray:
        """
        Solve fractional LP on observed steps to get dual prices for capacities.
        Returns a global lambda vector (length m).
        """
        observed: Sequence[StepSpec] = self._seen_steps[:t_k]
        n = instance.n
        m = instance.m
        if m <= 0:
            return np.zeros((0,), dtype=float)

        model = gp.Model("dla_fractional_pricing")
        model.Params.OutputFlag = 1 if self.cfg.dla.log_prices else 0

        # Capacity scaling per phase, optionally respecting offline slack.
        use_slack = self.cfg.dla.use_offline_slack and self.cfg.slack.enforce_slack
        slack_fraction = self.cfg.slack.fraction if use_slack else 0.0
        eff_total = effective_capacity(np.asarray(instance.b, dtype=float), use_slack, slack_fraction)

        # Subtract fixed OFFLINE load (online load is modeled by LP vars).
        offline_load = np.zeros((m,), dtype=float)
        fallback_idx = instance.fallback_option_index
        for step_id, option_id in state.assigned_option.items():
            if step_id >= len(instance.offline_steps):
                continue
            if option_id < 0 or option_id == fallback_idx:
                continue
            cap_mat = np.asarray(instance.offline_steps[step_id].cap_matrix, dtype=float)
            offline_load += cap_mat[:, option_id]

        base_residual = np.maximum(0.0, eff_total - offline_load)
        horizon = float(len(instance.online_steps))
        scaled = (1.0 - h_k) * (t_k / horizon) * base_residual
        residual = np.maximum(0.0, scaled)

        x: Dict[tuple[int, int], gp.Var] = {}
        y_fallback: Dict[int, gp.Var] = {}
        cap_by_id = {
            step_spec.step_id: np.asarray(step_spec.cap_matrix, dtype=float)
            for step_spec in observed
        }
        for step_spec in observed:
            feasible_options = feasible_option_indices(
                step_spec.feas_matrix,
                step_spec.feas_rhs,
                option_ids=range(n),
            )
            for i in feasible_options:
                x[(step_spec.step_id, i)] = model.addVar(
                    lb=0.0, ub=1.0, name=f"x_{step_spec.step_id}_{i}"
                )
            if option_is_feasible(step_spec.feas_matrix, step_spec.feas_rhs, fallback_idx):
                y_fallback[step_spec.step_id] = model.addVar(
                    lb=0.0, ub=1.0, name=f"y_fallback_{step_spec.step_id}"
                )
        model.update()

        cap_constr: Dict[int, gp.Constr] = {}
        for r in range(m):
            expr = gp.quicksum(
                cap_by_id[j][r, i] * var
                for (j, i), var in x.items()
            )
            cap_constr[r] = model.addConstr(expr <= float(residual[r]), name=f"cap_{r}")

        for step_spec in observed:
            step_vars = [var for (j, _), var in x.items() if j == step_spec.step_id]
            expr = gp.quicksum(step_vars)
            if step_spec.step_id in y_fallback:
                expr += y_fallback[step_spec.step_id]
            model.addConstr(expr == 1.0, name=f"assign_{step_spec.step_id}")

        obj = gp.quicksum(instance.costs.assignment_costs[j, i] * var for (j, i), var in x.items())
        if y_fallback:
            fallback_cost = self.cfg.costs.huge_fallback
            obj += gp.quicksum(fallback_cost * y for y in y_fallback.values())
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()
        if model.Status != GRB.OPTIMAL:
            if self._current_prices is not None:
                return self._current_prices
            return np.zeros((m,), dtype=float)

        lam = np.zeros((m,), dtype=float)
        for r in range(m):
            lam[r] = abs(float(cap_constr[r].Pi))
        return lam

    def _log_phase(self, t_k: int, h_k: float) -> None:
        if not self.cfg.dla.log_prices or self.price_path is None:
            return
        prices = (
            [float(x) for x in self._current_prices.tolist()]
            if self._current_prices is not None
            else []
        )
        record = {
            "phase": int(self._phase_idx),
            "t_k": int(t_k),
            "h_k": float(h_k),
            "prices": prices,
        }
        self._log_records.append(record)
        self.price_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.price_path, "w") as f:
            json.dump(
                {
                    "epsilon": float(self.cfg.dla.epsilon),
                    "schedule": self._schedule,
                    "records": self._log_records,
                },
                f,
                indent=2,
            )

    def _mark_processed(self, step: StepSpec) -> None:
        self._seen_steps.append(step)
        self._processed += 1

    # ------------------------------------------------------------------
    # Placement helpers (reuse primal-dual structure)
    # ------------------------------------------------------------------
    def _milp_no_eviction(
        self,
        step: StepSpec,
        state: AssignmentState,
        instance: Instance,
    ) -> Optional[Decision]:
        regular_options = instance.n
        if regular_options <= 0:
            return None
        if not feasible_option_indices(
            step.feas_matrix, step.feas_rhs, option_ids=range(regular_options)
        ):
            return None

        costs = np.array(
            [
                [
                    self._score(option_id, step, instance)
                    for option_id in range(regular_options)
                ]
            ],
            dtype=float,
        )
        cap_matrix = np.asarray(step.cap_matrix, dtype=float)
        capacities = remaining_capacities(self.cfg, state, instance)
        feas_matrix = np.asarray(step.feas_matrix, dtype=float)[:, :regular_options]
        feas_rhs = np.asarray(step.feas_rhs, dtype=float)
        data = build_offline_milp_data_from_arrays(
            cap_matrices=cap_matrix.reshape(1, *cap_matrix.shape),
            costs=costs,
            feas_matrices=[feas_matrix],
            feas_rhs=[feas_rhs],
            b=capacities,
            fallback_idx=-1,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._milp_solver.solve_from_data(data)
        if not rh_info.feasible:
            return None
        assigned_option = rh_state.assigned_option.get(0)
        if assigned_option is None:
            return None
        incremental_cost = lookup_assignment_cost(self.cfg, instance, step.step_id, assigned_option)
        return Decision(
            placed_step=(step.step_id, int(assigned_option)),
            evicted_offline_steps=[],
            reassigned_offline_steps=[],
            incremental_cost=incremental_cost,
        )

    def _candidate_options(
        self,
        step: StepSpec,
        instance: Instance,
    ) -> List[int]:
        regular_options = instance.n
        return feasible_option_indices(
            step.feas_matrix, step.feas_rhs, option_ids=range(regular_options)
        )

    def _score(self, option_id: int, step: StepSpec, instance: Instance) -> float:
        c_ji = lookup_assignment_cost(self.cfg, instance, step.step_id, option_id)
        cap_matrix = np.asarray(step.cap_matrix, dtype=float)
        usage = cap_matrix[:, option_id]
        lam = self._current_prices
        if lam is None or lam.size == 0:
            lam = np.zeros_like(usage)
        if self.cfg.util_pricing.vector_prices:
            return c_ji + float(np.dot(lam, usage))
        lam_scalar = scalarize_vector(lam, "max")
        return c_ji + lam_scalar * scalarize_vector(usage, self.cfg.heuristics.size_key)
