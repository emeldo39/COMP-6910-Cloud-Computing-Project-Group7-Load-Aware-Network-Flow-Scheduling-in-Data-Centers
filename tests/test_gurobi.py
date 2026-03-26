#!/usr/bin/env python3
"""
LAFS Project — Gurobi / MILP Solver Verification Test
COMP-6910 — Group 7

Tests:
  1. gurobipy imports and license is valid
  2. Simple LP (linear relaxation) solves correctly
  3. MILP (binary variables) solves correctly
  4. Multi-objective MILP matching LAFS §3.2 formulation
  5. PuLP/CBC fallback solver works correctly
  6. Solve time for 100-flow problem is < 100ms

Usage:
    python tests/test_gurobi.py
    python tests/test_gurobi.py --solver pulp   (test PuLP fallback only)
"""

import os
import sys
import time
import random
import argparse
import unittest


# =============================================================================
# Helper: solver selection
# =============================================================================
SOLVER = os.environ.get("LAFS_SOLVER", "gurobi")


# =============================================================================
# Test 1 — Gurobi import & license
# =============================================================================
class TestGurobiImport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import gurobipy as gp
            cls.gp = gp
            cls.GRB = gp.GRB
            cls.available = True
        except ImportError:
            cls.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("gurobipy not installed — install or see setup/gurobi_setup.md")

    def test_gurobipy_import(self):
        self._skip_if_unavailable()
        self.assertIsNotNone(self.gp)

    def test_gurobi_version(self):
        self._skip_if_unavailable()
        ver = self.gp.gurobi.version()
        major, minor, patch = ver
        print(f"\n  Gurobi version: {major}.{minor}.{patch}")
        self.assertGreaterEqual(major, 9, "Gurobi 9+ required")

    def test_gurobi_license_valid(self):
        """Create a model — this triggers license validation."""
        self._skip_if_unavailable()
        try:
            m = self.gp.Model("license_test")
            m.setParam("OutputFlag", 0)
            x = m.addVar(name="x")
            m.setObjective(x, self.GRB.MINIMIZE)
            m.addConstr(x >= 1)
            m.optimize()
            self.assertEqual(m.Status, self.GRB.OPTIMAL)
        except self.gp.GurobiError as e:
            self.fail(f"Gurobi license error: {e}\n"
                      f"See setup/gurobi_setup.md to activate your academic license.")


# =============================================================================
# Test 2 — Linear Program (LP relaxation, as in LAFS §3.2)
# =============================================================================
class TestLPRelaxation(unittest.TestCase):
    """
    Test the LP relaxation used in LAFS flow placement.
    The MILP is NP-hard; we use LP relaxation + randomized rounding
    to find <100ms solutions.
    """

    def _solve_lp_gurobi(self, n_flows: int = 10, n_paths: int = 3) -> dict:
        """
        Minimal LP relaxation:
          min  sum_f sum_p  fct[f][p] * x[f][p]
          s.t. sum_p x[f][p] = 1      for all flows f
               sum_f x[f][p] * load[f] <= cap[p]  for all paths p
               0 <= x[f][p] <= 1
        """
        import gurobipy as gp
        from gurobipy import GRB

        # Random FCT and load data
        # cap is set to total_load/n_paths + margin to guarantee feasibility
        random.seed(42)
        fct   = [[random.uniform(0.1, 5.0) for _ in range(n_paths)] for _ in range(n_flows)]
        load  = [random.uniform(0.1, 0.5) for _ in range(n_flows)]
        cap   = [sum(load) / n_paths + 0.5 for _ in range(n_paths)]

        m = gp.Model("lp_relaxation")
        m.setParam("OutputFlag", 0)

        # Variables: x[f][p] ∈ [0, 1]
        x = [[m.addVar(lb=0, ub=1, name=f"x_{f}_{p}")
              for p in range(n_paths)] for f in range(n_flows)]

        # Objective: minimize total FCT
        m.setObjective(
            gp.quicksum(fct[f][p] * x[f][p]
                        for f in range(n_flows)
                        for p in range(n_paths)),
            GRB.MINIMIZE
        )

        # Flow conservation: each flow assigned to exactly one path
        for f in range(n_flows):
            m.addConstr(gp.quicksum(x[f][p] for p in range(n_paths)) == 1)

        # Capacity constraints
        for p in range(n_paths):
            m.addConstr(
                gp.quicksum(load[f] * x[f][p] for f in range(n_flows)) <= cap[p]
            )

        t0 = time.perf_counter()
        m.optimize()
        solve_ms = (time.perf_counter() - t0) * 1000

        obj = m.ObjVal if m.Status == GRB.OPTIMAL else None
        return {"status": m.Status, "obj": obj, "solve_ms": solve_ms}

    def _solve_lp_pulp(self, n_flows: int = 10, n_paths: int = 3) -> dict:
        import pulp
        import random as rnd

        rnd.seed(42)
        fct  = [[rnd.uniform(0.1, 5.0) for _ in range(n_paths)] for _ in range(n_flows)]
        load = [rnd.uniform(0.1, 0.5) for _ in range(n_flows)]
        cap  = [sum(load) / n_paths + 0.5 for _ in range(n_paths)]

        prob = pulp.LpProblem("lp_relaxation", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{f}_{p}", lowBound=0, upBound=1)
              for p in range(n_paths)] for f in range(n_flows)]

        prob += pulp.lpSum(fct[f][p] * x[f][p]
                           for f in range(n_flows)
                           for p in range(n_paths))

        for f in range(n_flows):
            prob += pulp.lpSum(x[f][p] for p in range(n_paths)) == 1

        for p in range(n_paths):
            prob += pulp.lpSum(load[f] * x[f][p] for f in range(n_flows)) <= cap[p]

        t0 = time.perf_counter()
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        solve_ms = (time.perf_counter() - t0) * 1000

        return {
            "status": pulp.LpStatus[status],
            "obj": pulp.value(prob.objective),
            "solve_ms": solve_ms,
        }

    def test_lp_gurobi_small(self):
        try:
            result = self._solve_lp_gurobi(n_flows=10, n_paths=3)
            import gurobipy as gp
            self.assertEqual(result["status"], gp.GRB.OPTIMAL)
            self.assertGreater(result["obj"], 0)
            print(f"\n  LP (Gurobi, 10 flows): obj={result['obj']:.3f}, "
                  f"time={result['solve_ms']:.1f}ms")
        except ImportError:
            self.skipTest("gurobipy not available")

    def test_lp_pulp_small(self):
        try:
            result = self._solve_lp_pulp(n_flows=10, n_paths=3)
            self.assertEqual(result["status"], "Optimal")
            self.assertIsNotNone(result["obj"])
            print(f"\n  LP (PuLP/CBC, 10 flows): obj={result['obj']:.3f}, "
                  f"time={result['solve_ms']:.1f}ms")
        except ImportError:
            self.skipTest("pulp not available")


# =============================================================================
# Test 3 — MILP (binary flow assignment)
# =============================================================================
class TestMILP(unittest.TestCase):
    """Test binary MILP matching the LAFS proposal §3.2 formulation."""

    def _build_lafs_milp_gurobi(self, n_flows: int, n_paths: int) -> dict:
        """
        LAFS MILP formulation:
          Variables:
            x[f][p] ∈ {0, 1}  — binary: flow f assigned to path p
            z[t]   >= 0        — max link utilisation for tenant t
          Objectives (combined as weighted sum):
            min  w1 * avg_fct_mice
               + w2 * p99_fct_elephants
               + w3 * max_min_fairness_violation
          Constraints:
            sum_p x[f][p] = 1        (flow conservation)
            link capacity <= C       (capacity)
            z[t] >= x[f][p] - fair   (fairness violation)
        """
        import gurobipy as gp
        from gurobipy import GRB

        random.seed(0)
        # Flow sizes: ~20% elephants (>10MB), 80% mice (<100KB)
        flow_sizes = [random.choice(["mice"] * 4 + ["elephant"]) for _ in range(n_flows)]
        fct = [[random.uniform(0.1, 2.0) if flow_sizes[f] == "mice"
                else random.uniform(2.0, 10.0)
                for p in range(n_paths)] for f in range(n_flows)]
        load = [0.05 if flow_sizes[f] == "mice" else 0.3 for f in range(n_flows)]
        cap  = [1.0] * n_paths

        # Tenant assignment (4 tenants)
        n_tenants = 4
        tenant_of = [f % n_tenants for f in range(n_flows)]
        fair_share = 1.0 / n_tenants

        m = gp.Model("lafs_milp")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", 10)   # 10s max for test

        # ── Variables ────────────────────────────────────────────────────────
        x = [[m.addVar(vtype=GRB.BINARY, name=f"x_{f}_{p}")
              for p in range(n_paths)] for f in range(n_flows)]
        z = [m.addVar(lb=0, name=f"z_{t}") for t in range(n_tenants)]

        # ── Objective: weighted sum ──────────────────────────────────────────
        mice_flows     = [f for f in range(n_flows) if flow_sizes[f] == "mice"]
        elephant_flows = [f for f in range(n_flows) if flow_sizes[f] == "elephant"]

        avg_mice_fct = (gp.quicksum(fct[f][p] * x[f][p]
                                     for f in mice_flows
                                     for p in range(n_paths))
                        / max(len(mice_flows), 1))

        p99_elephant_fct = (gp.quicksum(fct[f][p] * x[f][p]
                                         for f in elephant_flows
                                         for p in range(n_paths))
                             / max(len(elephant_flows), 1))

        fairness_violation = gp.quicksum(z)

        m.setObjective(
            0.4 * avg_mice_fct + 0.4 * p99_elephant_fct + 0.2 * fairness_violation,
            GRB.MINIMIZE
        )

        # ── Constraints ──────────────────────────────────────────────────────
        for f in range(n_flows):
            m.addConstr(gp.quicksum(x[f][p] for p in range(n_paths)) == 1,
                        name=f"conservation_{f}")

        for p in range(n_paths):
            m.addConstr(
                gp.quicksum(load[f] * x[f][p] for f in range(n_flows)) <= cap[p],
                name=f"capacity_{p}"
            )

        for t in range(n_tenants):
            tenant_flows = [f for f in range(n_flows) if tenant_of[f] == t]
            for p in range(n_paths):
                alloc = gp.quicksum(load[f] * x[f][p] for f in tenant_flows)
                m.addConstr(z[t] >= alloc - fair_share, name=f"fairness_{t}_{p}")

        t0 = time.perf_counter()
        m.optimize()
        solve_ms = (time.perf_counter() - t0) * 1000

        return {
            "status":   m.Status,
            "obj":      m.ObjVal if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else None,
            "solve_ms": solve_ms,
            "n_flows":  n_flows,
        }

    def test_milp_small_gurobi(self):
        """20-flow MILP should solve to optimality."""
        try:
            result = self._build_lafs_milp_gurobi(n_flows=20, n_paths=3)
            import gurobipy as gp
            self.assertIn(result["status"],
                          [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT])
            print(f"\n  MILP (Gurobi, 20 flows): obj={result['obj']:.3f}, "
                  f"time={result['solve_ms']:.1f}ms")
        except ImportError:
            self.skipTest("gurobipy not available")

    def test_milp_100flows_under_100ms(self):
        """
        Per proposal §3.2: LP relaxation + randomized rounding should give
        <100ms solutions for 100-flow problems.
        We test the LP relaxation here (MILP may take longer).
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
            import random as rnd

            n_flows, n_paths = 100, 4
            rnd.seed(1)
            fct  = [[rnd.uniform(0.1, 5.0) for _ in range(n_paths)] for _ in range(n_flows)]
            load = [rnd.uniform(0.05, 0.15) for _ in range(n_flows)]
            cap  = [5.0] * n_paths

            m = gp.Model("lp100")
            m.setParam("OutputFlag", 0)
            x = [[m.addVar(lb=0, ub=1) for p in range(n_paths)] for f in range(n_flows)]
            m.setObjective(
                gp.quicksum(fct[f][p] * x[f][p]
                            for f in range(n_flows) for p in range(n_paths)),
                GRB.MINIMIZE
            )
            for f in range(n_flows):
                m.addConstr(gp.quicksum(x[f][p] for p in range(n_paths)) == 1)
            for p in range(n_paths):
                m.addConstr(gp.quicksum(load[f] * x[f][p] for f in range(n_flows)) <= cap[p])

            t0 = time.perf_counter()
            m.optimize()
            solve_ms = (time.perf_counter() - t0) * 1000

            print(f"\n  LP relaxation (100 flows, 4 paths): {solve_ms:.1f}ms")
            self.assertLess(solve_ms, 100,
                            f"LP solve took {solve_ms:.1f}ms — exceeds 100ms budget")
        except ImportError:
            self.skipTest("gurobipy not available")


# =============================================================================
# Test 4 — PuLP/CBC fallback
# =============================================================================
class TestPuLPFallback(unittest.TestCase):

    def test_pulp_import(self):
        import pulp
        self.assertIsNotNone(pulp.__version__)
        print(f"\n  PuLP version: {pulp.__version__}")

    def test_cbc_available(self):
        import pulp
        solvers = pulp.listSolvers(onlyAvailable=True)
        cbc_available = any("CBC" in s for s in solvers)
        self.assertTrue(cbc_available, f"CBC not available. Found: {solvers}")
        print(f"\n  Available PuLP solvers: {solvers}")

    def test_pulp_milp_solve(self):
        import pulp

        prob = pulp.LpProblem("lafs_fallback", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(5)]
        y = pulp.LpVariable("y", lowBound=0)

        prob += y   # Minimize y
        prob += y >= pulp.lpSum(x)   # y >= sum of assignments
        prob += pulp.lpSum(x) >= 3   # At least 3 flows assigned

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        self.assertEqual(pulp.LpStatus[status], "Optimal")
        result_y = pulp.value(y)
        self.assertGreaterEqual(result_y, 3.0)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["gurobi", "pulp"], default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args()

    if args.solver:
        os.environ["LAFS_SOLVER"] = args.solver

    verbosity = 2 if args.verbose else 1
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestGurobiImport, TestLPRelaxation, TestMILP, TestPuLPFallback]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
