"""
LAFS -- MILP Flow Placement Solver
====================================
COMP 6910 -- Group 7

Solves the multi-objective flow placement MILP:

    Minimise:
        z  +  lambda * Sigma_{f in mice} Sigma_p x[f,p] * hops[p]

    Subject to:
        x[f,p] in {0, 1}                         (binary assignment)
        Sigma_p x[f,p] = 1       for all f        (each flow gets one path)
        z >= u_l + Sigma_{f,p: l in path_p(f)} x[f,p] * b_f / C_l
                                 for all links l   (max utilisation tracking)

    where:
        z    -- auxiliary continuous variable = max link utilisation
        u_l  -- predicted utilisation on link l (from NetworkLoadForecast)
        b_f  -- bandwidth demand of flow f = size_bytes * 8 / window_s  (bps)
        C_l  -- capacity of link l (bps)

Solver backend is selected by the LAFS_SOLVER environment variable:
    LAFS_SOLVER=pulp    (default) -- PuLP + CBC open-source solver
    LAFS_SOLVER=gurobi             -- Gurobi (requires valid license)

If the primary solver raises any exception, the module falls back
automatically to a greedy least-loaded-path heuristic.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.workload.flow import Flow

_log = logging.getLogger("lafs.optimizer.milp")

_SOLVER_ENV = "LAFS_SOLVER"
_DEFAULT_SOLVER = "pulp"


# =============================================================================
# MILPConfig
# =============================================================================

@dataclass
class MILPConfig:
    """Tunable parameters for the MILP solver.

    Attributes
    ----------
    solver : str
        Backend to use. Overridden by the LAFS_SOLVER environment variable.
        Values: ``"pulp"`` (default, open-source CBC) or ``"gurobi"``.
    time_limit_s : float
        Maximum wall-clock time given to the solver per scheduling window.
        Default 5 s is adequate for k=8 fat-tree (1 000 flows, 16 paths each).
    mip_gap : float
        Acceptable relative optimality gap.  1 % (0.01) is fine for network
        scheduling -- tiny improvements in objective are not worth extra solve
        time.
    mice_hop_weight : float
        Small coefficient on the hop-count penalty for mice flows.  Keeps the
        primary objective (max utilisation) dominant while preferring shorter
        paths for latency-sensitive small flows as a tie-breaker.
    verbose : bool
        When True, the underlying solver prints its progress to stdout.
    """

    solver: str = field(
        default_factory=lambda: os.environ.get(_SOLVER_ENV, _DEFAULT_SOLVER)
    )
    time_limit_s: float = 5.0
    mip_gap: float = 0.01
    mice_hop_weight: float = 1e-3
    verbose: bool = False


# =============================================================================
# MILPResult
# =============================================================================

@dataclass
class MILPResult:
    """Output of one MILP solve.

    Attributes
    ----------
    assignments : dict[str, list[str]]
        Maps each ``flow_id`` to its assigned path (list of node names).
        Flows with no candidate paths are absent from this dict.
    max_utilisation : float
        Achieved maximum link utilisation across all links (0.0 -- inf).
    solve_time_s : float
        Wall-clock seconds from solver entry to result extraction.
    status : str
        Human-readable solver status: "Optimal", "Feasible", "Infeasible",
        "Fallback", or "Error".
    n_flows : int
        Total number of flows passed to the solver.
    n_vars : int
        Total number of binary decision variables in the model.
    n_constraints : int
        Total number of constraints in the model.
    n_links : int
        Number of directed links with at least one path crossing them.
    solver_used : str
        Which backend actually ran: "PuLP/CBC", "Gurobi", or "Greedy".
    """

    assignments: Dict[str, List[str]]
    max_utilisation: float
    solve_time_s: float
    status: str
    n_flows: int
    n_vars: int
    n_constraints: int
    n_links: int
    solver_used: str

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"[{self.solver_used}] status={self.status} "
            f"flows={self.n_flows} vars={self.n_vars} "
            f"max_util={self.max_utilisation:.3f} "
            f"time={self.solve_time_s*1e3:.1f}ms"
        )


# =============================================================================
# LAFSMILPSolver
# =============================================================================

class LAFSMILPSolver:
    """
    MILP solver for LAFS load-aware flow placement.

    Parameters
    ----------
    config : MILPConfig or None
        Solver configuration.  If None, default MILPConfig() is used, which
        reads the LAFS_SOLVER environment variable.

    Usage
    -----
    ::

        solver = LAFSMILPSolver()
        result = solver.solve(
            flows=flow_list,
            candidate_paths={fid: [[...], [...], ...], ...},
            predicted_utils={(u, v): 0.42, ...},
            link_capacities={(u, v): 1e9, ...},
        )
        for fid, path in result.assignments.items():
            print(fid, "->", path)
    """

    def __init__(self, config: Optional[MILPConfig] = None) -> None:
        self.config = config or MILPConfig()
        self._backend = self.config.solver.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        flows: List[Flow],
        candidate_paths: Dict[str, List[List[str]]],
        predicted_utils: Dict[Tuple[str, str], float],
        link_capacities: Dict[Tuple[str, str], float],
        window_s: float = 0.1,
    ) -> MILPResult:
        """
        Place *flows* onto candidate paths minimising maximum link utilisation.

        Parameters
        ----------
        flows : List[Flow]
            Flows awaiting path assignment.
        candidate_paths : dict[flow_id -> list of paths]
            ``candidate_paths[flow_id]`` is a list of candidate paths.
            Each path is an ordered list of node names.
        predicted_utils : dict[(u, v) -> float]
            Forecast utilisation per directed link, from ``NetworkLoadForecast``.
            Values should be in [0, 1].  Missing links are treated as 0.
        link_capacities : dict[(u, v) -> float]
            Capacity in bps for every directed link present in candidate paths.
        window_s : float
            Length of the current scheduling window in seconds.
            Used to convert flow.size_bytes to a bandwidth demand in bps.

        Returns
        -------
        MILPResult
        """
        if not flows:
            return MILPResult(
                assignments={}, max_utilisation=0.0, solve_time_s=0.0,
                status="Optimal", n_flows=0, n_vars=0, n_constraints=0,
                n_links=0, solver_used=self._backend.upper(),
            )

        t0 = time.perf_counter()

        # Per-flow bandwidth demand (bps) for capacity constraint
        demands: Dict[str, float] = {
            f.flow_id: f.size_bytes * 8.0 / window_s for f in flows
        }

        try:
            if self._backend == "gurobi":
                result = self._solve_gurobi(
                    flows, candidate_paths, demands,
                    predicted_utils, link_capacities, t0,
                )
            else:
                result = self._solve_pulp(
                    flows, candidate_paths, demands,
                    predicted_utils, link_capacities, t0,
                )
        except Exception as exc:
            _log.warning(
                "MILP solver (%s) failed -- falling back to greedy: %s",
                self._backend, exc,
            )
            result = self._fallback_greedy(flows, candidate_paths, demands, link_capacities, t0)

        return result

    # ------------------------------------------------------------------
    # PuLP / CBC backend
    # ------------------------------------------------------------------

    def _solve_pulp(
        self,
        flows: List[Flow],
        candidate_paths: Dict[str, List[List[str]]],
        demands: Dict[str, float],
        predicted_utils: Dict[Tuple[str, str], float],
        link_capacities: Dict[Tuple[str, str], float],
        t0: float,
    ) -> MILPResult:
        import pulp  # imported lazily; always present (listed in requirements)

        prob = pulp.LpProblem("LAFS_Flow_Placement", pulp.LpMinimize)

        # --- Decision variables: x[fid][path_idx] in {0, 1} ---
        x: Dict[str, List] = {}
        for flow in flows:
            fid = flow.flow_id
            n_p = len(candidate_paths.get(fid, []))
            x[fid] = [
                pulp.LpVariable(f"x_{fid}_{i}", cat="Binary")
                for i in range(n_p)
            ]

        # --- Auxiliary: z = max link utilisation ---
        z = pulp.LpVariable("z", lowBound=0.0)

        # --- Objective: min z + mice hop-count penalty ---
        obj_terms = [z]
        for flow in flows:
            if flow.is_mice:
                fid = flow.flow_id
                for i, path in enumerate(candidate_paths.get(fid, [])):
                    if i < len(x[fid]):
                        obj_terms.append(
                            self.config.mice_hop_weight * (len(path) - 1) * x[fid][i]
                        )
        prob += pulp.lpSum(obj_terms), "Objective"

        # --- Constraint 1: each flow assigned to exactly one path ---
        n_assign = 0
        for flow in flows:
            fid = flow.flow_id
            if x[fid]:
                prob += pulp.lpSum(x[fid]) == 1, f"assign_{fid}"
                n_assign += 1

        # --- Build link incidence: link -> list of (fid, path_idx) ---
        link_incidence: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
        for flow in flows:
            fid = flow.flow_id
            for pi, path in enumerate(candidate_paths.get(fid, [])):
                for hop in range(len(path) - 1):
                    link = (path[hop], path[hop + 1])
                    if link not in link_incidence:
                        link_incidence[link] = []
                    link_incidence[link].append((fid, pi))

        # --- Constraint 2: z >= predicted_util[l] + new_load[l] ---
        n_util = 0
        for link, incidence in link_incidence.items():
            cap = link_capacities.get(link, 1e9)
            if cap <= 0:
                continue
            pred = float(predicted_utils.get(link, 0.0))
            new_load_terms = [
                x[fid][pi] * (demands.get(fid, 0.0) / cap)
                for fid, pi in incidence
                if fid in x and pi < len(x[fid])
            ]
            # Sanitise link name for PuLP constraint name
            lname = f"{link[0]}__{link[1]}".replace("-", "_")
            prob += z >= pred + pulp.lpSum(new_load_terms), f"util_{lname}"
            n_util += 1

        # --- Solve ---
        solver = pulp.PULP_CBC_CMD(
            timeLimit=self.config.time_limit_s,
            gapRel=self.config.mip_gap,
            msg=1 if self.config.verbose else 0,
        )
        prob.solve(solver)
        solve_time = time.perf_counter() - t0
        status_str = pulp.LpStatus.get(prob.status, "Unknown")

        # --- Extract assignments ---
        assignments: Dict[str, List[str]] = {}
        for flow in flows:
            fid = flow.flow_id
            paths = candidate_paths.get(fid, [])
            if not paths:
                continue
            # Pick path with highest x value (should be 0/1 for MIP)
            best_idx = 0
            best_val = -1.0
            for i, var in enumerate(x[fid]):
                val = pulp.value(var)
                if val is not None and val > best_val:
                    best_val = val
                    best_idx = i
            assignments[fid] = paths[best_idx]

        max_util = float(pulp.value(z) or 0.0)
        n_vars = sum(len(v) for v in x.values()) + 1  # +1 for z

        return MILPResult(
            assignments=assignments,
            max_utilisation=max_util,
            solve_time_s=solve_time,
            status=status_str,
            n_flows=len(flows),
            n_vars=n_vars,
            n_constraints=n_assign + n_util,
            n_links=len(link_incidence),
            solver_used="PuLP/CBC",
        )

    # ------------------------------------------------------------------
    # Gurobi backend
    # ------------------------------------------------------------------

    def _solve_gurobi(
        self,
        flows: List[Flow],
        candidate_paths: Dict[str, List[List[str]]],
        demands: Dict[str, float],
        predicted_utils: Dict[Tuple[str, str], float],
        link_capacities: Dict[Tuple[str, str], float],
        t0: float,
    ) -> MILPResult:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("LAFS")
        if not self.config.verbose:
            m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", self.config.time_limit_s)
        m.setParam("MIPGap", self.config.mip_gap)

        # --- Decision variables ---
        x: Dict[str, List] = {}
        for flow in flows:
            fid = flow.flow_id
            n_p = len(candidate_paths.get(fid, []))
            x[fid] = [
                m.addVar(vtype=GRB.BINARY, name=f"x_{fid}_{i}")
                for i in range(n_p)
            ]

        z = m.addVar(lb=0.0, name="z")

        # --- Objective ---
        obj = z
        for flow in flows:
            if flow.is_mice:
                fid = flow.flow_id
                for i, path in enumerate(candidate_paths.get(fid, [])):
                    if i < len(x[fid]):
                        obj = obj + self.config.mice_hop_weight * (len(path) - 1) * x[fid][i]
        m.setObjective(obj, GRB.MINIMIZE)

        # --- Assignment constraints ---
        n_assign = 0
        for flow in flows:
            fid = flow.flow_id
            if x[fid]:
                m.addConstr(gp.quicksum(x[fid]) == 1, name=f"assign_{fid}")
                n_assign += 1

        # --- Link incidence ---
        link_incidence: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
        for flow in flows:
            fid = flow.flow_id
            for pi, path in enumerate(candidate_paths.get(fid, [])):
                for hop in range(len(path) - 1):
                    link = (path[hop], path[hop + 1])
                    if link not in link_incidence:
                        link_incidence[link] = []
                    link_incidence[link].append((fid, pi))

        # --- Utilisation constraints ---
        n_util = 0
        for link, incidence in link_incidence.items():
            cap = link_capacities.get(link, 1e9)
            if cap <= 0:
                continue
            pred = float(predicted_utils.get(link, 0.0))
            new_load = gp.quicksum(
                x[fid][pi] * (demands.get(fid, 0.0) / cap)
                for fid, pi in incidence
                if fid in x and pi < len(x[fid])
            )
            lname = f"util_{link[0]}__{link[1]}".replace("-", "_")
            m.addConstr(z >= pred + new_load, name=lname)
            n_util += 1

        m.optimize()
        solve_time = time.perf_counter() - t0

        _STATUS = {
            GRB.OPTIMAL: "Optimal",
            GRB.SUBOPTIMAL: "Feasible",
            GRB.INFEASIBLE: "Infeasible",
            GRB.TIME_LIMIT: "Feasible",
        }
        status_str = _STATUS.get(m.status, f"Code{m.status}")

        assignments: Dict[str, List[str]] = {}
        for flow in flows:
            fid = flow.flow_id
            paths = candidate_paths.get(fid, [])
            if not paths:
                continue
            best_idx = max(
                range(len(x[fid])),
                key=lambda i: (x[fid][i].X if x[fid][i].X is not None else -1.0),
            )
            assignments[fid] = paths[best_idx]

        try:
            max_util = float(z.X)
        except Exception:
            max_util = 0.0

        n_vars = sum(len(v) for v in x.values()) + 1
        return MILPResult(
            assignments=assignments,
            max_utilisation=max_util,
            solve_time_s=solve_time,
            status=status_str,
            n_flows=len(flows),
            n_vars=n_vars,
            n_constraints=n_assign + n_util,
            n_links=len(link_incidence),
            solver_used="Gurobi",
        )

    # ------------------------------------------------------------------
    # Greedy fallback
    # ------------------------------------------------------------------

    def _fallback_greedy(
        self,
        flows: List[Flow],
        candidate_paths: Dict[str, List[List[str]]],
        demands: Dict[str, float],
        link_capacities: Dict[Tuple[str, str], float],
        t0: float,
    ) -> MILPResult:
        """Least-loaded-path greedy assignment (O(F * P * L) per window)."""
        link_loads: Dict[Tuple[str, str], float] = {}
        assignments: Dict[str, List[str]] = {}

        for flow in flows:
            fid = flow.flow_id
            paths = candidate_paths.get(fid, [])
            if not paths:
                continue

            best_path = paths[0]
            best_load = float("inf")

            for path in paths:
                path_max = max(
                    (link_loads.get((path[h], path[h + 1]), 0.0)
                     for h in range(len(path) - 1)),
                    default=0.0,
                )
                if path_max < best_load:
                    best_load = path_max
                    best_path = path

            assignments[fid] = best_path

            # Update link load accumulators
            demand_bps = demands.get(fid, 0.0)
            for h in range(len(best_path) - 1):
                link = (best_path[h], best_path[h + 1])
                cap = link_capacities.get(link, 1e9)
                link_loads[link] = link_loads.get(link, 0.0) + demand_bps / max(cap, 1.0)

        max_util = max(link_loads.values(), default=0.0)

        return MILPResult(
            assignments=assignments,
            max_utilisation=max_util,
            solve_time_s=time.perf_counter() - t0,
            status="Fallback",
            n_flows=len(flows),
            n_vars=0,
            n_constraints=0,
            n_links=len(link_loads),
            solver_used="Greedy",
        )
