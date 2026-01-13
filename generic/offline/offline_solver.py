from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from generic.config import Config
from generic.models import Instance, AssignmentState
from generic.general_utils import effective_capacity as _effective_capacity

from generic.offline.models import OfflineSolutionInfo, _status_name

class OfflineMILPSolver:
    """
    Offline MILP für die initiale Zuordnung der OFFLINE-Items.

    Eckpunkte:
    - Regular bins: Indizes 0..N-1 (harte Kapazitäten)
    - Fallback bin: Index N (keine Kapazitätsgrenze), nur für OFFLINE-Items
    - Online-Items werden später behandelt (online-Phase)
    - Feasibility-Graph wird respektiert (Variablen nur auf zulässigen Kanten)
    - Slack global konfigurierbar (Default: aus)
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
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.log_to_console = log_to_console

        # werden in _build_model gesetzt:
        self.model: Optional[gp.Model] = None
        self.x: Dict[Tuple[int, int], gp.Var] = {}  # (j, i) -> Var
        self.N: int = 0
        self.M: int = 0
        self.fallback_idx: int = -1
        self.vol: Optional[np.ndarray] = None
        self.cost: Optional[np.ndarray] = None
        self.feas: Optional[np.ndarray] = None

    # ---------- Public API ----------

    def solve(
        self,
        inst: Instance,
        warm_start: Optional[Dict[int, int]] = None,  # {offline_item_id -> bin_id}
    ) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """
        Modell bauen, lösen, Lösung extrahieren.
        """
        self._build_model(inst)
        
        # Generate warm start if not provided but config requests it
        if warm_start is None and self.cfg.solver.use_warm_start:
            warm_start = self._generate_warm_start(inst)
        
        if warm_start:
            self._apply_warm_start(warm_start)

        m = self.model
        assert m is not None
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.mip_gap
        if self.threads:
            m.Params.Threads = self.threads
        m.Params.OutputFlag = 1 if self.log_to_console else 0

        m.optimize()
        state, info = self._extract_solution()
        return state, info

    # ---------- Model construction ----------

    def _build_model(self, inst: Instance) -> None:
        self.N = len(inst.bins)                 # reguläre Bins
        self.M = len(inst.offline_items)        # Anzahl OFFLINE-Items
        self.fallback_idx = inst.fallback_bin_index  # sollte == N sein (0-basiert)
        assert self.fallback_idx == self.N, "Expected fallback index to be N (0-based)."

        # Volumina und Kosten
        self.vol = np.array([it.volume for it in inst.offline_items], dtype=float)
        self.cost = inst.costs.assign[:self.M, :self.N+1]#.astype(float, copy=True)
        self.feas = inst.feasible.feasible[:self.M, :self.N+1]#.astype(int, copy=True)

        # Gurobi-Modell
        self.model = gp.Model("offline_initial_allocation")
        m = self.model
        self.x.clear()

        # Variablen nur für zulässige Kanten (weil spart Speicher + Constraints)
        for j in range(self.M):
            for i in range(self.N + 1):  # inkl. Fallback
                if self.feas[j, i] == 1:
                    self.x[(j, i)] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{i}")
        m.update()

        # Jede OFFLINE-Item-Zuweisung genau einmal
        for j in range(self.M):
            vars_j = [self.x[(j, i)] for i in range(self.N + 1) if (j, i) in self.x]
            if not vars_j:
                # sollte nicht passieren, da Fallback für OFFLINE existiert
                raise ValueError(f"Offline item {j} has no feasible bin (including fallback).")
            m.addConstr(gp.quicksum(vars_j) == 1, name=f"assign_{j}")

        # Kapazitätsrestriktionen nur auf regulären Bins (0..N-1)
        cap_used: Dict[int, np.ndarray] = {}
        load_exprs: Dict[Tuple[int, int], gp.LinExpr] = {}
        for i in range(self.N):
            cap_i = _effective_capacity(
                inst.bins[i].capacity,
                self.cfg.slack.enforce_slack,
                self.cfg.slack.fraction,
            )
            cap_used[i] = cap_i
            for d in range(self.vol.shape[1]):
                vars_i = []
                coeffs = []
                for j in range(self.M):
                    if (j, i) in self.x:
                        vars_i.append(self.x[(j, i)])
                        coeffs.append(self.vol[j, d])
                if vars_i:
                    load_expr = gp.LinExpr(coeffs, vars_i)
                    load_exprs[(i, d)] = load_expr
                    m.addConstr(load_expr <= float(cap_i[d]), name=f"cap_{i}_{d}")

        # Zielfunktion: minimale Zuweisungskosten
        obj_terms = []
        for j in range(self.M):
            for i in range(self.N + 1):
                if (j, i) in self.x:
                    obj_terms.append(self.cost[j, i] * self.x[(j, i)])

        m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
        m.update()

    # ---------- Optional warm start ----------

    @staticmethod
    def state_to_warm_start(state: AssignmentState) -> Dict[int, int]:
        """Convert AssignmentState to warm start format for MILP."""
        return state.assigned_bin.copy()

    def _generate_warm_start(self, inst: Instance) -> Dict[int, int]:
        """Generate warm start solution (override in problem-specific solvers)."""
        return {}

    def _apply_warm_start(self, warm_start: Dict[int, int]) -> None:
        """
        Warm-Start (Start-Lösung) setzen, sofern Kante zulässig ist.
        """
        if self.model is None:
            return
        for j, i in warm_start.items():
            var = self.x.get((j, i))
            if var is not None:
                var.Start = 1.0

    # ---------- Extraction ----------

    def _extract_solution(self) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """
        Lösung robust extrahieren (zugriffssicher, auch ohne incumbent).
        """
        m = self.model
        assert m is not None

        status_code = m.Status
        status_name = _status_name(status_code)

        # Nur wenn Gurobi eine Lösung kennt (SolCount > 0), dürfen wir var.X/ObjVal lesen.
        has_solution = (getattr(m, "SolCount", 0) is not None) and (m.SolCount > 0)

        # Dense 0/1-Matrix der Größe M x (N+1) bauen (auch wenn keine Lösung existiert)
        x_sol = np.zeros((self.M, self.N + 1), dtype=int)
        if has_solution:
            for (j, i), var in self.x.items():
                x_sol[j, i] = int(round(var.X))

        # Loads & Zuordnungen nur berechnen, wenn eine Lösung existiert
        load = np.zeros((self.N + 1, self.vol.shape[1]), dtype=float)
        assigned_bin: Dict[int, int] = {}
        if has_solution:
            for j in range(self.M):
                i = int(np.argmax(x_sol[j, :]))
                assigned_bin[j] = i
                if i < self.N:
                    load[i] += self.vol[j]
                else: 
                    load[self.fallback_idx] += self.vol[j]
        state = AssignmentState(
            load=load,
            assigned_bin=assigned_bin,
            offline_evicted=set(),
        )

        info = OfflineSolutionInfo(
            status=status_name,
            obj_value=float(m.ObjVal) if has_solution else float("inf"),
            mip_gap=float(m.MIPGap) if has_solution else float("inf"),
            runtime=float(m.Runtime),
            assignments=x_sol,
        )
        return state, info
