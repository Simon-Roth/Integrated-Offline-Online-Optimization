from __future__ import annotations

"""
Dynamic Learning Algorithm 

Key elements implemented here:
 - Geometric phase schedule t_k = epsilon * M_onl * 2^k (with integer rounding).
 - Capacity scaling per phase: b_i^(k) = (1 - h_k) * (t_k / M_onl) * C_i where
   h_k = epsilon * sqrt(M_onl / t_k).
 - Dual-based price computation on the observed items so far (fractional LP).
 - Online placement using price-aware milp
 - If no placement is possible, we try evicting offline items (except if disabled in config)
 - If still no placement possible, generic/online_solver.py will try the fallback (if enabled) as last resort

The policy optionally logs per-phase prices/residuals to a JSON file if a path is
provided and cfg.dla.log_prices is True.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from generic.config import Config
from generic.data.offline_milp_assembly import build_offline_milp_data_from_arrays
from generic.general_utils import effective_capacity, scalarize_vector, vector_fits, residual_vector
from generic.models import AssignmentState, Decision, Instance, OnlineItem
from generic.offline.offline_solver import OfflineMILPSolver
from generic.online.policies import BaseOnlinePolicy, PolicyInfeasibleError
from generic.online.policy_utils import current_feasible_row, lookup_assignment_cost, remaining_capacities
from binpacking.online.state_utils import (
    PlacementContext,
    build_context,
    execute_placement,
    offline_volumes,
    TOLERANCE,
)
from binpacking.block_utils import block_dim, extract_volume, split_capacities


class DynamicLearningPolicy(BaseOnlinePolicy):
    """
    Dynamic price-updating policy
    """

    def __init__(self, cfg: Config, price_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.price_path = Path(price_path) if price_path else None

        # Runtime state
        self._phase_idx: int = -1
        self._schedule: List[int] = []
        self._current_prices: Dict[int, np.ndarray] = {}
        self._processed: int = 0  # number of arrivals already placed
        self._seen_items: List[OnlineItem] = []
        self._log_records: List[Dict[str, object]] = []
        self._milp_solver = OfflineMILPSolver(cfg)

   
    def select_action(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Decision:
        # Lazily build phase schedule on the first call.
        if not self._schedule:
            self._schedule = self._build_schedule(len(instance.online_items))

        # Update prices if a phase boundary was reached after the last item.
        self._maybe_update_prices(state, instance)

        candidate_bins = self._candidate_bins(item, instance, feasible_row)
        if not candidate_bins:
            raise PolicyInfeasibleError(f"No feasible regular bin for online item {item.id}")

        # First, try to place without evictions via a one-item MILP (regular bins only).
        milp_decision = self._milp_no_eviction(item, state, instance, feasible_row)
        if milp_decision is not None:
            self._mark_processed(item)
            return milp_decision

        if not self.cfg.problem.allow_reassignment:
            raise PolicyInfeasibleError(
                f"DynamicLearningPolicy could not place item {item.id}"
            )

        # Allow evictions if no feasible bin remained 
        best_decision: Optional[Decision] = None
        best_score = float("inf")
        for bin_id in candidate_bins:
            ctx: PlacementContext = build_context(self.cfg, instance, state)
            score = self._score(bin_id, item, instance)
            if score >= best_score - 1e-12:
                continue
            decision = execute_placement(
                bin_id,
                item,
                ctx,
                eviction_order_fn=self._eviction_order_desc,
                destination_fn=self._select_reassignment_bin,
                allow_eviction=True,
            )
            if decision is not None:
                best_score = score
                best_decision = decision

        if best_decision is not None:
            self._mark_processed(item)
            return best_decision

        # No feasible regular bin (even with evictions) -> signal infeasibility.
        raise PolicyInfeasibleError(f"DynamicLearningPolicy could not place item {item.id}")

    # ------------------------------------------------------------------
    # Pricing + scheduling
    # ------------------------------------------------------------------
    def _build_schedule(self, horizon: int) -> List[int]:
        """Geometric schedule t_k = ε * M_onl * 2^k (rounded to ints)."""
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
        """Update prices when we've completed a phase (after placing t_k items)."""
        if self._processed == 0:
            return
        if self._phase_idx + 1 >= len(self._schedule):
            return
        next_boundary = self._schedule[self._phase_idx + 1]
        if self._processed < next_boundary:
            return

        # Compute new prices using observed items up to t_k = next_boundary.
        t_k = next_boundary
        horizon = max(1, len(instance.online_items))
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
    ) -> Dict[int, np.ndarray]:
        """
        Solve fractional LP on observed items to get dual prices for capacities.
        """
        model = gp.Model("dla_fractional_pricing")
        model.Params.OutputFlag = 1 if self.cfg.dla.log_prices else 0

        observed: Sequence[OnlineItem] = self._seen_items[:t_k]
        n = instance.n
        m = instance.m
        d = block_dim(n, m)
        online_volumes = {
            item.id: extract_volume(item.cap_matrix, n, m) for item in observed
        }

        # Capacity scaling per phase, optionally respecting offline slack.
        caps_scaled: List[np.ndarray] = []
        use_slack = self.cfg.dla.use_offline_slack and self.cfg.slack.enforce_slack
        slack_fraction = self.cfg.slack.fraction if use_slack else 0.0
        offline_load = np.zeros((n, d), dtype=float)
        off_vols = offline_volumes(instance)
        fallback_idx = instance.fallback_action_index
        # Count fixed offline load currently occupying regular bins.
        for item_id, bin_id in state.assigned_action.items():
            if item_id >= len(instance.offline_items):
                continue
            if bin_id < 0 or bin_id >= n or bin_id == fallback_idx:
                continue
            offline_load[bin_id] += off_vols.get(item_id, np.zeros(d))

        horizon = float(len(instance.online_items))
        caps = split_capacities(instance.b, n)
        for idx in range(n):
            phys = caps[idx]
            eff_total = effective_capacity(phys, use_slack, slack_fraction)
            base_residual = np.maximum(0.0, eff_total - offline_load[idx])
            scaled = (1.0 - h_k) * (t_k / horizon) * base_residual
            caps_scaled.append(np.maximum(0.0, scaled))

        # Residual after offline load only (online load is modeled by LP vars).
        residual = caps_scaled

        allow_fallback = self.cfg.problem.fallback_is_enabled
        x = {}
        y_fallback = {}
        for item in observed:
            for i in item.feasible_actions:
                x[(item.id, i)] = model.addVar(
                    lb=0.0, ub=1.0, name=f"x_{item.id}_{i}"
                )
            if allow_fallback:
                y_fallback[item.id] = model.addVar(
                    lb=0.0, ub=1.0, name=f"y_fallback_{item.id}"
                )
        model.update()

        cap_constr: Dict[tuple[int, int], gp.Constr] = {}
        for i in range(n):
            for d_idx in range(d):
                expr = gp.quicksum(
                    online_volumes[j][d_idx] * var
                    for (j, ii), var in x.items()
                    if ii == i
                )
                cap_constr[(i, d_idx)] = model.addConstr(
                    expr <= float(residual[i][d_idx]), name=f"cap_{i}_{d_idx}"
                )

        for item in observed:
            item_vars = [var for (j, i), var in x.items() if j == item.id]
            expr = gp.quicksum(item_vars)
            if allow_fallback:
                expr += y_fallback[item.id]
            model.addConstr(expr == 1.0, name=f"assign_{item.id}")

        obj = gp.quicksum(instance.costs.assignment_costs[j, i] * var for (j, i), var in x.items())
        if allow_fallback:
            fallback_cost = self.cfg.costs.huge_fallback
            obj += gp.quicksum(fallback_cost * y for y in y_fallback.values())
        model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"DLA pricing LP not optimal, status={model.Status}")

        prices: Dict[int, np.ndarray] = {}
        for i in range(n):
            lam = np.zeros(d, dtype=float)
            for d_idx in range(d):
                lam[d_idx] = abs(float(cap_constr[(i, d_idx)].Pi))
            prices[i] = lam
        return prices

    def _log_phase(self, t_k: int, h_k: float) -> None:
        if not self.cfg.dla.log_prices or self.price_path is None:
            return
        record = {
            "phase": int(self._phase_idx),
            "t_k": int(t_k),
            "h_k": float(h_k),
            "prices": {int(k): [float(x) for x in v.tolist()] for k, v in self._current_prices.items()},
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

    def _mark_processed(self, item: OnlineItem) -> None:
        self._seen_items.append(item)
        self._processed += 1

    # ------------------------------------------------------------------
    # Placement helpers (reuse primal-dual structure)
    # ------------------------------------------------------------------
    def _milp_no_eviction(
        self,
        item: OnlineItem,
        state: AssignmentState,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> Optional[Decision]:
        regular_bins = instance.n
        if regular_bins <= 0:
            return None
        fallback_idx = instance.fallback_action_index
        cols = regular_bins + (1 if fallback_idx >= 0 else 0)
        try:
            feasible_full = current_feasible_row(
                self.cfg,
                item,
                feasible_row,
                regular_bins,
                cols,
                fallback_idx,
            )
        except PolicyInfeasibleError:
            return None

        feasible = feasible_full.reshape(-1)[:regular_bins]
        if feasible.sum() == 0:
            return None

        costs = np.array(
            [[self._score(bin_id, item, instance) for bin_id in range(regular_bins)]],
            dtype=float,
        )
        cap_matrix = np.asarray(item.cap_matrix, dtype=float)
        capacities = remaining_capacities(self.cfg, state, instance)
        data = build_offline_milp_data_from_arrays(
            cap_matrices=cap_matrix.reshape(1, *cap_matrix.shape),
            costs=costs,
            feasible=feasible.reshape(1, -1),
            b=capacities,
            fallback_idx=-1,
            slack_enforce=False,
            slack_fraction=0.0,
        )
        rh_state, rh_info = self._milp_solver.solve_from_data(data)
        if not rh_info.feasible:
            return None
        assigned_action = rh_state.assigned_action.get(0)
        if assigned_action is None:
            return None
        incremental_cost = lookup_assignment_cost(self.cfg, instance, item.id, assigned_action)
        return Decision(
            placed_item=(item.id, int(assigned_action)),
            evicted_offline=[],
            reassignments=[],
            incremental_cost=incremental_cost,
        )

    def _candidate_bins(
        self,
        item: OnlineItem,
        instance: Instance,
        feasible_row: Optional[np.ndarray],
    ) -> List[int]:
        regular_bins = instance.n
        fallback_idx = instance.fallback_action_index
        cols = regular_bins + (1 if fallback_idx >= 0 else 0)
        try:
            feasible = current_feasible_row(
                self.cfg,
                item,
                feasible_row,
                regular_bins,
                cols,
                fallback_idx,
            )
        except PolicyInfeasibleError:
            return []
        row = feasible.reshape(-1)
        return [idx for idx in range(regular_bins) if row[idx] == 1]

    def _score(self, bin_id: int, item: OnlineItem, instance: Instance) -> float:
        c_ji = lookup_assignment_cost(self.cfg, instance, item.id, bin_id)
        volume = extract_volume(item.cap_matrix, instance.n, instance.m)
        lam_i = self._current_prices.get(bin_id, np.zeros_like(volume))
        if self.cfg.util_pricing.vector_prices:
            return c_ji + float(np.dot(lam_i, volume))
        lam_scalar = scalarize_vector(lam_i, "max")
        return c_ji + lam_scalar * scalarize_vector(volume, self.cfg.heuristics.size_key)

    def _eviction_order_desc(
        self,
        bin_id: int,
        ctx: PlacementContext,
    ) -> List[int]:
        offline_ids = [
            itm_id
            for itm_id, assigned_bin in ctx.assignments.items()
            if assigned_bin == bin_id and itm_id < len(ctx.instance.offline_items)
        ]
        size_key = self.cfg.heuristics.size_key
        zero_vec = np.zeros_like(ctx.effective_caps[0])
        offline_ids.sort(
            key=lambda oid: scalarize_vector(ctx.offline_volumes.get(oid, zero_vec), size_key),
            reverse=True,
        )
        return offline_ids

    def _select_reassignment_bin(
        self,
        offline_id: int,
        origin_bin: int,
        ctx: PlacementContext,
    ) -> Optional[int]:
        zero_vec = np.zeros_like(ctx.effective_caps[0])
        volume = ctx.offline_volumes.get(offline_id, zero_vec)
        instance = ctx.instance
        feasible_row = instance.offline_feasible.feasible[offline_id]
        regular_bins = instance.n
        fallback_idx = instance.fallback_action_index

        best_candidate: Optional[int] = None
        best_cost = float("inf")
        best_residual = float("inf")

        for candidate in range(regular_bins):
            if candidate == origin_bin or feasible_row[candidate] != 1:
                continue
            residual_vec = residual_vector(ctx.loads[candidate], volume, ctx.effective_caps[candidate])
            if not vector_fits(ctx.loads[candidate], volume, ctx.effective_caps[candidate], TOLERANCE):
                continue
            cost = instance.costs.assignment_costs[offline_id, candidate]
            residual_score = scalarize_vector(residual_vec, self.cfg.heuristics.residual_scalarization)
            if cost < best_cost - 1e-9 or (
                abs(cost - best_cost) <= 1e-9 and residual_score < best_residual
            ):
                best_cost = cost
                best_residual = residual_score
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate

        if (
            self.cfg.problem.fallback_is_enabled and self.cfg.problem.fallback_allowed_offline
            and fallback_idx < feasible_row.shape[0]
            and feasible_row[fallback_idx] == 1
        ):
            return fallback_idx
        return None
