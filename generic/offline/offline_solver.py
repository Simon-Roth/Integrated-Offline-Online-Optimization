from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from generic.config import Config
from generic.models import Instance, AssignmentState
from generic.data.offline_milp_assembly import OfflineMILPData, build_offline_milp_data

from generic.offline.models import OfflineSolutionInfo, _status_name

class OfflineMILPSolver:
    """
    Offline MILP für die initiale Zuordnung der OFFLINE-Items.

    Eckpunkte:
    - Regular bins: Indizes 0..N-1 (harte Kapazitäten)
    - Fallback bin: Index N (optional; falls aktiviert)
    - Online-Items werden später behandelt (online-Phase)
    - Feasibility-Graph wird respektiert (über Ax <= b)
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
        self.x: Optional[gp.MVar] = None
        self.N: int = 0
        self.bins_total: int = 0
        self.M: int = 0
        self.var_shape: Tuple[int, int] = (0, 0)
        self.fallback_idx: int = -1
        self.dimensions: int = 1
        self.vol: Optional[np.ndarray] = None
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
        data = build_offline_milp_data(inst, self.cfg)

        # Generate warm start if not provided but config requests it.
        if warm_start is None and self.cfg.solver.use_warm_start:
            warm_start = self._generate_warm_start(inst)

        return self.solve_from_data(data, warm_start=warm_start)

    def solve_from_data(
        self,
        data: OfflineMILPData,
        *,
        warm_start: Optional[Dict[int, int]] = None,
    ) -> Tuple[AssignmentState, OfflineSolutionInfo]:
        """
        Solve the MILP directly from A, b, c data (no instance generation needed).
        """
        self._build_model_from_data(data)

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

    def _build_model_from_data(self, data: OfflineMILPData) -> None:
        self.var_shape = data.var_shape
        self.M, bins_total = self.var_shape
        self.bins_total = bins_total
        self.N = bins_total - 1 if data.fallback_idx >= 0 else bins_total
        self.fallback_idx = data.fallback_idx
        self.dimensions = data.dimensions
        self.vol = data.volumes
        self.feas = data.feasible
        if self.fallback_idx >= 0:
            assert self.fallback_idx == self.N, "Expected fallback index to be N (0-based)."

        # Gurobi-Modell
        self.model = gp.Model("offline_initial_allocation")
        m = self.model
        num_vars = int(data.c.size)

        if num_vars:
            self.x = m.addMVar(shape=num_vars, vtype=GRB.BINARY, name="x")
            if data.A.size:
                m.addMConstr(data.A, self.x, "<", data.b, name="Axb")
            m.setObjective(data.c @ self.x, GRB.MINIMIZE)
        else:
            self.x = m.addMVar(shape=0, vtype=GRB.BINARY, name="x")
            m.setObjective(0.0, GRB.MINIMIZE)
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
        if self.model is None or self.x is None:
            return
        M, Np1 = self.var_shape
        for j, i in warm_start.items():
            if not (0 <= j < M and 0 <= i < Np1):
                continue
            if self.feas is not None and self.feas[j, i] == 0:
                continue
            idx = j * Np1 + i
            self.x[idx].Start = 1.0

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
        x_sol = np.zeros(self.var_shape, dtype=int)
        if has_solution and self.x is not None:
            x_vec = np.asarray(self.x.X, dtype=float)
            if x_vec.size:
                x_sol = np.rint(x_vec.reshape(self.var_shape)).astype(int)

        # Loads & Zuordnungen nur berechnen, wenn eine Lösung existiert
        load = np.zeros((self.bins_total, self.dimensions), dtype=float)
        assigned_bin: Dict[int, int] = {}
        if has_solution:
            for j in range(self.M):
                i = int(np.argmax(x_sol[j, :]))
                assigned_bin[j] = i
                if i < self.N:
                    load[i] += self.vol[j]
                elif self.fallback_idx >= 0 and i == self.fallback_idx:
                    load[self.fallback_idx] += self.vol[j]
        state = AssignmentState(
            load=load,
            assigned_bin=assigned_bin,
            offline_evicted=set(),
        )

        info = OfflineSolutionInfo(
            algorithm=self.__class__.__name__,
            status=status_name,
            obj_value=float(m.ObjVal) if has_solution else float("inf"),
            runtime=float(m.Runtime),
            feasible=bool(has_solution),
        )
        return state, info
