from __future__ import annotations

from pathlib import Path
from typing import Protocol, Optional
import json

import numpy as np

from generic.config import Config
from generic.data.instance_generators import (
    _ensure_row_feasible,
    _sample_assignment_costs,
    _sample_online_volumes,
)
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.general_utils import effective_capacity, make_rng
from generic.models import AssignmentState, Instance, OnlineItem, Decision
from generic.online.policy_utils import (
    current_cost_row,
    current_feasible_row,
    lookup_assignment_cost,
    PolicyInfeasibleError,
    remaining_capacities,
)


class BaseOnlinePolicy(Protocol):
    """
    Interface that all online heuristics/solvers must implement.
    """

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        """
        Decide how to place 'item' given the current assignment state.

        Parameters
        ----------
        item:
            The arriving online item.
        state:
            Current assignment state (will be mutated by the solver after the decision).
        instance:
            The full problem instance (offline + online data).
        feasible_row:
            Row of the feasibility matrix for this item, if available (length N or N+1).

        Returns
        -------
        Decision:
            Encodes bin placement, evictions, reassignments, and incremental cost.
        """
        ...


__all__ = [
    "BaseOnlinePolicy",
    "PolicyInfeasibleError",
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
        from generic.offline.offline_solver import OfflineMILPSolver

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
        self._t: int = 0

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        # Step 1-2: initialize duals and remaining capacities once per instance.
        self._reset_if_needed(instance, state)
        assert self._lambda is not None
        assert self._cap_per_step is not None
        assert self._effective_caps is not None

        # Step 3: observe A_t (volume + feasibility) and c_t (assignment costs).
        N = len(instance.bins)
        fallback_idx = instance.fallback_bin_index
        cols = N + (1 if fallback_idx >= 0 else 0)
        volume = np.asarray(item.volume, dtype=float).reshape(1, -1)
        feasible = current_feasible_row(
            self.cfg,
            item,
            feasible_row,
            N,
            cols,
            fallback_idx,
        )
        costs = current_cost_row(self.cfg, instance, item.id, cols).reshape(1, -1)

        # Step 4: solve the price-aware MILP for this single arrival.
        capacities, fallback_capacity = remaining_capacities(
            self.cfg,
            state,
            instance,
            self._effective_caps,
        )
        costs = self._price_aware_costs(costs, volume, N)
        assigned_bin = self._solve_price_aware_milp(
            volume,
            costs,
            feasible,
            capacities,
            fallback_idx,
            fallback_capacity,
        )

        # Step 5-7: apply x_t (the OnlineSolver updates state/load), then update lambda.
        self._update_duals(assigned_bin, volume, N)

        incremental_cost = lookup_assignment_cost(self.cfg, instance, item.id, assigned_bin)
        return Decision(
            placed_item=(item.id, int(assigned_bin)),
            evicted_offline=[],
            reassignments=[],
            incremental_cost=incremental_cost,
        )

    # Initialize per-instance state (lambda, caps, scaling).
    def _reset_if_needed(self, instance: Instance, state: AssignmentState) -> None:
        # Reset per-instance state so prices/counters don't leak across runs.
        if self._instance_id == id(instance) and self._lambda is not None:
            return
        self._instance_id = id(instance)
        N = len(instance.bins)
        use_slack = bool(self.cfg.slack.enforce_slack and self.cfg.slack.apply_to_online)
        slack_fraction = self.cfg.slack.fraction if use_slack else 0.0
        caps = [
            effective_capacity(b.capacity, use_slack, slack_fraction)
            for b in instance.bins
        ]
        caps_eff = np.asarray(caps, dtype=float)
        if caps_eff.ndim == 1:
            caps_eff = caps_eff.reshape((N, -1))
        T = max(1, len(instance.online_items))
        self._effective_caps = caps_eff
        self._cap_per_step = caps_eff / float(T)
        self._cost_scale = self._compute_cost_scale(instance, N)
        self._eta_scale = 1.0 / self._cost_scale if self.cfg.primal_dual.normalize_costs else 1.0
        self._t = 0

        load = np.asarray(state.load, dtype=float)
        if load.ndim == 1:
            load = load.reshape((load.shape[0], -1))
        load = load[:N]
        denom = np.where(caps_eff > 0.0, caps_eff, 1.0)
        utilization = np.maximum(0.0, load / denom)
        eta0 = self._eta_t() * self._eta_scale
        self._lambda = eta0 * utilization

    # Apply optional cost scaling and add the price term to costs.
    def _price_aware_costs(self, costs: np.ndarray, volume: np.ndarray, N: int) -> np.ndarray:
        if self.cfg.primal_dual.normalize_costs:
            costs = costs / self._cost_scale
        if N:
            price_terms = self._price_terms(volume)
            costs[0, :N] = costs[0, :N] + price_terms
        return costs

    # Compute lambda^T A_t^cap term (optionally normalized).
    def _price_terms(self, volume: np.ndarray) -> np.ndarray:
        volume_vec = volume.reshape(-1)
        if self.cfg.primal_dual.normalize_update:
            denom = np.where(self._cap_per_step > 0.0, self._cap_per_step, 1.0)
            scaled = volume_vec / denom
            return (self._lambda * scaled).sum(axis=1)
        return self._lambda @ volume_vec

    # Solve the one-step price-aware MILP and return the chosen bin.
    def _solve_price_aware_milp(
        self,
        volume: np.ndarray,
        costs: np.ndarray,
        feasible: np.ndarray,
        capacities: np.ndarray,
        fallback_idx: int,
        fallback_capacity: float | np.ndarray,
    ) -> int:
        data = build_offline_milp_data_from_arrays(
            volumes=volume,
            costs=costs,
            feasible=feasible,
            capacities=capacities,
            fallback_idx=fallback_idx,
            fallback_capacity=fallback_capacity,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._solver.solve_from_data(data)
        if not rh_info.feasible:
            raise PolicyInfeasibleError("Primal-dual MILP has no feasible solution.")
        assigned_bin = rh_state.assigned_bin.get(0)
        if assigned_bin is None:
            raise PolicyInfeasibleError("Primal-dual MILP did not assign the current item.")
        return int(assigned_bin)

    # Update dual prices after choosing x_t.
    def _update_duals(self, assigned_bin: int, volume: np.ndarray, N: int) -> None:
        eta_t = self._eta_t() * self._eta_scale
        usage = np.zeros_like(self._lambda)
        if 0 <= assigned_bin < N:
            usage[assigned_bin] = volume.reshape(-1)
        if self.cfg.primal_dual.normalize_update:
            denom = np.where(self._cap_per_step > 0.0, self._cap_per_step, 1.0)
            delta = (usage - self._cap_per_step) / denom
        else:
            delta = usage - self._cap_per_step
        self._lambda = np.maximum(self._lambda + eta_t * delta, 0.0)
        self._t += 1

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
    def _compute_cost_scale(self, instance: Instance, N: int) -> float:
        min_scale = float(self.cfg.primal_dual.cost_scale_min)
        mode = self.cfg.primal_dual.cost_scale_mode.lower()
        if mode == "assign_bounds":
            lo, hi = self.cfg.costs.assign_bounds
            return max(min_scale, 0.5 * (float(lo) + float(hi)))
        if mode == "assign_mean":
            costs = instance.costs.assignment_costs
            if costs is None or not costs.size or N <= 0:
                return max(min_scale, 1.0)
            online_ids = [item.id for item in instance.online_items]
            if not online_ids:
                return max(min_scale, 1.0)
            values = costs[np.asarray(online_ids, dtype=int), :N].reshape(-1)
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
        price_path: Path = Path("binpacking/results/sim_dual.json"),
        time_limit: int = 60,
        mip_gap: float = 0.01,
        threads: int = 0,
        log_to_console: bool = False,
    ) -> None:
        self.cfg = cfg
        self._price_path = price_path
        self._lambda = self._load_prices(price_path)
        from generic.offline.offline_solver import OfflineMILPSolver

        self._solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        # Step 1: observe A_t (volume + feasibility) and c_t (assignment costs)
        N = len(instance.bins)
        fallback_idx = instance.fallback_bin_index
        cols = N + (1 if fallback_idx >= 0 else 0)
        volume = np.asarray(item.volume, dtype=float).reshape(1, -1)
        feasible = current_feasible_row(
            self.cfg,
            item,
            feasible_row,
            N,
            cols,
            fallback_idx,
        )
        costs = current_cost_row(self.cfg, instance, item.id, cols).reshape(1, -1)

        # Step 2: add fixed dual prices to the objective and solve the one-step MILP.
        capacities, fallback_capacity = remaining_capacities(self.cfg, state, instance)
        costs = self._price_aware_costs(costs, volume, N)
        assigned_bin = self._solve_price_aware_milp(
            volume,
            costs,
            feasible,
            capacities,
            fallback_idx,
            fallback_capacity,
        )

        incremental_cost = lookup_assignment_cost(self.cfg, instance, item.id, assigned_bin)
        return Decision(
            placed_item=(item.id, int(assigned_bin)),
            evicted_offline=[],
            reassignments=[],
            incremental_cost=incremental_cost,
        )

    # Load per-bin dual prices from JSON.
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

    # Add lambda^T A_t^cap to assignment costs (regular bins only).
    def _price_aware_costs(self, costs: np.ndarray, volume: np.ndarray, N: int) -> np.ndarray:
        if N <= 0:
            return costs
        vol = volume.reshape(-1)
        price_terms = np.zeros((N,), dtype=float)
        for bin_id in range(N):
            lam = self._lambda.get(bin_id)
            if lam is None:
                continue
            lam_vec = lam.reshape(-1)
            if lam_vec.size == 1 and vol.size > 1:
                lam_vec = np.full((vol.size,), float(lam_vec[0]))
            if lam_vec.size != vol.size:
                raise ValueError(
                    f"Price vector dimension mismatch for bin {bin_id}: "
                    f"{lam_vec.size} vs {vol.size}"
                )
            price_terms[bin_id] = float(np.dot(lam_vec, vol))
        costs[0, :N] = costs[0, :N] + price_terms
        return costs

    # Solve the one-step price-aware MILP and return the chosen bin.
    def _solve_price_aware_milp(
        self,
        volume: np.ndarray,
        costs: np.ndarray,
        feasible: np.ndarray,
        capacities: np.ndarray,
        fallback_idx: int,
        fallback_capacity: float | np.ndarray,
    ) -> int:
        data = build_offline_milp_data_from_arrays(
            volumes=volume,
            costs=costs,
            feasible=feasible,
            capacities=capacities,
            fallback_idx=fallback_idx,
            fallback_capacity=fallback_capacity,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._solver.solve_from_data(data)
        if not rh_info.feasible:
            raise PolicyInfeasibleError("SimDual MILP has no feasible solution.")
        assigned_bin = rh_state.assigned_bin.get(0)
        if assigned_bin is None:
            raise PolicyInfeasibleError("SimDual MILP did not assign the current item.")
        return int(assigned_bin)


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
        from generic.offline.offline_solver import OfflineMILPSolver

        self._solver = OfflineMILPSolver(
            cfg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )

    def select_bin(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        # Step 1: observe A_t and remaining capacity (from state).
        idx = self._online_index(item, instance)
        remaining = max(0, len(instance.online_items) - idx - 1)

        N = len(instance.bins)
        fallback_idx = instance.fallback_bin_index
        cols = N + (1 if fallback_idx >= 0 else 0)

        # Step 2: sample future A_tau for tau > t and build the remaining MILP.
        volumes = self._build_volumes(item, remaining)
        costs = self._build_costs(instance, item.id, remaining, N, cols)
        feasible = self._build_feasible(
            item,
            feasible_row,
            remaining,
            N,
            cols,
            fallback_idx,
        )
        capacities, fallback_capacity = remaining_capacities(self.cfg, state, instance)

        # Step 3: solve the remaining-horizon MILP.
        data = build_offline_milp_data_from_arrays(
            volumes=volumes,
            costs=costs,
            feasible=feasible,
            capacities=capacities,
            fallback_idx=fallback_idx,
            fallback_capacity=fallback_capacity,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._solver.solve_from_data(data)
        if not rh_info.feasible:
            raise PolicyInfeasibleError("Rolling horizon MILP has no feasible solution.")

        assigned_bin = rh_state.assigned_bin.get(0)
        if assigned_bin is None:
            raise PolicyInfeasibleError("Rolling horizon MILP did not assign the current item.")
        # Step 4-5: fix x_t (current item), OnlineSolver updates remaining capacity after applying.
        incremental_cost = lookup_assignment_cost(self.cfg, instance, item.id, assigned_bin)
        return Decision(
            placed_item=(item.id, int(assigned_bin)),
            evicted_offline=[],
            reassignments=[],
            incremental_cost=incremental_cost,
        )

    # Map online item id to its index in the arrival sequence.
    def _online_index(self, item: OnlineItem, instance: Instance) -> int:
        base = len(instance.offline_items)
        idx = item.id - base
        if idx < 0 or idx >= len(instance.online_items):
            raise KeyError(f"Online item id {item.id} is out of expected range.")
        if instance.online_items[idx].id != item.id:
            raise KeyError(
                f"Online item id {item.id} does not match instance ordering."
            )
        return idx

    # Stack current item volume with sampled future volumes.
    def _build_volumes(self, item: OnlineItem, remaining: int) -> np.ndarray:
        current_volume = np.asarray(item.volume, dtype=float).reshape(1, -1)
        if remaining <= 0:
            return current_volume
        future_volumes = _sample_online_volumes(self.cfg, self._rng, remaining)
        return np.vstack([current_volume, future_volumes])

    # Build assignment costs for current + sampled items.
    def _build_costs(
        self,
        instance: Instance,
        item_id: int,
        remaining: int,
        N: int,
        cols: int,
    ) -> np.ndarray:
        costs = np.zeros((1 + remaining, cols), dtype=float)
        costs[0] = current_cost_row(self.cfg, instance, item_id, cols)
        if remaining <= 0:
            return costs
        base_costs = _sample_assignment_costs(self.cfg, self._rng, remaining, N)
        if cols > N:
            fallback_costs = np.full((remaining, 1), self.cfg.costs.huge_fallback, dtype=float)
            costs[1:] = np.hstack([base_costs, fallback_costs])
        else:
            costs[1:] = base_costs
        return costs

    # Build feasibility rows for current + sampled items.
    def _build_feasible(
        self,
        item: OnlineItem,
        feasible_row: Optional[np.ndarray],
        remaining: int,
        N: int,
        cols: int,
        fallback_idx: int,
    ) -> np.ndarray:
        current = current_feasible_row(
            self.cfg,
            item,
            feasible_row,
            N,
            cols,
            fallback_idx,
        )
        if remaining <= 0:
            return current
        base_mask = (self._rng.uniform(size=(remaining, N)) < self.cfg.graphs.p_onl).astype(int)
        _ensure_row_feasible(base_mask, self._rng)
        if cols > N:
            fallback_val = 1 if self.cfg.problem.fallback_allowed_online else 0
            fallback_col = np.full((remaining, 1), fallback_val, dtype=int)
            future = np.hstack([base_mask, fallback_col])
        else:
            future = base_mask
        return np.vstack([current, future])
