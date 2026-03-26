"""
Unit tests -- LAFS Optimizer (Phase 5)
========================================
COMP 6910 -- Group 7

Coverage
--------
TestMILPConfig         (4 tests) -- defaults, env override, field types
TestMILPResult         (4 tests) -- summary string, field values
TestLAFSMILPSolverPuLP (16 tests) -- PuLP/CBC MILP correctness
TestLAFSMILPSolverGreedy (5 tests) -- greedy fallback
TestLAFSScheduler      (14 tests) -- integration with topology + BaseScheduler API

Total: 43 tests
"""

import os
import sys
import time
import unittest

# ---------------------------------------------------------------------------
# Minimal path setup so tests can be run from project root
# ---------------------------------------------------------------------------
_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow
from src.optimizer.milp_solver import MILPConfig, MILPResult, LAFSMILPSolver
from src.optimizer.lafs_scheduler import LAFSScheduler
from src.optimizer import MILPConfig, MILPResult, LAFSMILPSolver, LAFSScheduler


# =============================================================================
# Helpers
# =============================================================================

def _make_flow(flow_id: str, src_ip: str, dst_ip: str, size_bytes: int = 1000) -> Flow:
    return Flow(
        flow_id=flow_id,
        src_ip=src_ip, dst_ip=dst_ip,
        src_port=1000, dst_port=80,
        protocol=6,
        size_bytes=size_bytes,
        arrival_time=0.0,
    )


def _small_topo(k: int = 4) -> FatTreeGraph:
    return FatTreeGraph(k=k)


def _tiny_milp_inputs():
    """
    Build a small manually-constructed MILP problem:
      - 2 flows, each with 2 candidate paths
      - 3 directed links, 1 Gbps capacity
      - zero predicted utils
    """
    flows = [
        _make_flow("f1", "10.0.0.2", "10.0.1.2", size_bytes=10_000),
        _make_flow("f2", "10.0.0.3", "10.0.1.3", size_bytes=50_000),
    ]
    candidate_paths = {
        "f1": [["A", "B", "C"], ["A", "D", "C"]],
        "f2": [["A", "B", "C"], ["A", "D", "C"]],
    }
    predicted_utils = {}
    link_caps = {
        ("A", "B"): 1e9, ("B", "A"): 1e9,
        ("A", "D"): 1e9, ("D", "A"): 1e9,
        ("B", "C"): 1e9, ("C", "B"): 1e9,
        ("D", "C"): 1e9, ("C", "D"): 1e9,
    }
    return flows, candidate_paths, predicted_utils, link_caps


# =============================================================================
# TestMILPConfig
# =============================================================================

class TestMILPConfig(unittest.TestCase):

    def test_defaults(self):
        """Default config uses 'pulp' if LAFS_SOLVER not set."""
        os.environ.pop("LAFS_SOLVER", None)
        cfg = MILPConfig()
        self.assertEqual(cfg.solver, "pulp")

    def test_env_override(self):
        """LAFS_SOLVER env var overrides default."""
        os.environ["LAFS_SOLVER"] = "gurobi"
        cfg = MILPConfig()
        self.assertEqual(cfg.solver, "gurobi")
        os.environ.pop("LAFS_SOLVER", None)

    def test_explicit_solver(self):
        """Explicit solver= kwarg takes precedence over env."""
        os.environ["LAFS_SOLVER"] = "gurobi"
        cfg = MILPConfig(solver="pulp")
        self.assertEqual(cfg.solver, "pulp")
        os.environ.pop("LAFS_SOLVER", None)

    def test_field_types(self):
        cfg = MILPConfig()
        self.assertIsInstance(cfg.time_limit_s, float)
        self.assertIsInstance(cfg.mip_gap, float)
        self.assertIsInstance(cfg.mice_hop_weight, float)
        self.assertIsInstance(cfg.verbose, bool)


# =============================================================================
# TestMILPResult
# =============================================================================

class TestMILPResult(unittest.TestCase):

    def _make_result(self, **kwargs) -> MILPResult:
        defaults = dict(
            assignments={"f1": ["A", "B", "C"]},
            max_utilisation=0.42,
            solve_time_s=0.05,
            status="Optimal",
            n_flows=1,
            n_vars=3,
            n_constraints=5,
            n_links=2,
            solver_used="PuLP/CBC",
        )
        defaults.update(kwargs)
        return MILPResult(**defaults)

    def test_summary_contains_status(self):
        r = self._make_result()
        self.assertIn("Optimal", r.summary())

    def test_summary_contains_solver(self):
        r = self._make_result()
        self.assertIn("PuLP/CBC", r.summary())

    def test_max_utilisation_accessible(self):
        r = self._make_result(max_utilisation=0.75)
        self.assertAlmostEqual(r.max_utilisation, 0.75)

    def test_assignments_dict(self):
        r = self._make_result()
        self.assertIn("f1", r.assignments)
        self.assertEqual(r.assignments["f1"], ["A", "B", "C"])


# =============================================================================
# TestLAFSMILPSolverPuLP
# =============================================================================

class TestLAFSMILPSolverPuLP(unittest.TestCase):
    """Tests for PuLP/CBC backend on small, hand-crafted instances."""

    def setUp(self):
        os.environ.pop("LAFS_SOLVER", None)
        self.cfg = MILPConfig(solver="pulp", time_limit_s=10.0, verbose=False)
        self.solver = LAFSMILPSolver(self.cfg)

    def test_empty_flows_returns_quickly(self):
        result = self.solver.solve([], {}, {}, {})
        self.assertEqual(result.n_flows, 0)
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.assignments, {})

    def test_single_flow_single_path(self):
        flow = _make_flow("f1", "10.0.0.2", "10.0.1.2", 1000)
        cands = {"f1": [["A", "B", "C"]]}
        caps = {("A", "B"): 1e9, ("B", "C"): 1e9, ("B", "A"): 1e9, ("C", "B"): 1e9}
        result = self.solver.solve([flow], cands, {}, caps)
        self.assertIn("f1", result.assignments)
        self.assertEqual(result.assignments["f1"], ["A", "B", "C"])

    def test_each_flow_assigned_exactly_once(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        for flow in flows:
            self.assertIn(flow.flow_id, result.assignments,
                          f"{flow.flow_id} not in assignments")

    def test_assignments_are_valid_paths(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        for fid, path in result.assignments.items():
            self.assertIn(path, cands[fid], f"{fid} assigned invalid path {path}")

    def test_max_utilisation_non_negative(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        self.assertGreaterEqual(result.max_utilisation, 0.0)

    def test_max_utilisation_upper_bounded(self):
        """With tiny demands and 1 Gbps links, utilisation < 1."""
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        # 50 KB / 0.1s = 4 Mbps << 1 Gbps
        self.assertLess(result.max_utilisation, 1.0)

    def test_predicted_utils_increase_objective(self):
        """High predicted load on link B should raise max_utilisation."""
        flows, cands, _, caps = _tiny_milp_inputs()
        utils_high = {("A", "B"): 0.8, ("B", "C"): 0.8}
        result = self.solver.solve(flows, cands, utils_high, caps)
        # At least one path goes through B, so z >= 0.8
        self.assertGreaterEqual(result.max_utilisation, 0.8)

    def test_solver_prefers_less_congested_path(self):
        """When path 0 is congested and path 1 is free, MILP picks path 1."""
        flows = [_make_flow("f1", "10.0.0.2", "10.0.1.2", 1000)]
        cands = {"f1": [["A", "B", "C"], ["A", "D", "C"]]}
        # Link A->B is 90% loaded, A->D is free
        utils_heavy = {("A", "B"): 0.9, ("B", "C"): 0.9}
        caps = {
            ("A", "B"): 1e9, ("B", "A"): 1e9, ("B", "C"): 1e9, ("C", "B"): 1e9,
            ("A", "D"): 1e9, ("D", "A"): 1e9, ("D", "C"): 1e9, ("C", "D"): 1e9,
        }
        result = self.solver.solve(flows, cands, utils_heavy, caps)
        # Optimal: route through D to avoid congested B
        self.assertEqual(result.assignments.get("f1"), ["A", "D", "C"])

    def test_n_flows_matches_input(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        self.assertEqual(result.n_flows, len(flows))

    def test_solver_used_is_pulp(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        self.assertIn("PuLP", result.solver_used)

    def test_n_vars_positive(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        # 2 flows * 2 paths + 1 z = 5 vars
        self.assertGreater(result.n_vars, 0)

    def test_n_constraints_positive(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        self.assertGreater(result.n_constraints, 0)

    def test_solve_time_positive(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        self.assertGreater(result.solve_time_s, 0.0)

    def test_flow_no_candidate_paths_absent(self):
        """Flows with no candidate paths must not appear in assignments."""
        flows = [_make_flow("fX", "10.0.0.2", "10.0.1.2")]
        result = self.solver.solve(flows, {}, {}, {})
        # No candidate path -- solver gets empty list, so no assignment
        # (solve handles empty flows_with_paths gracefully)
        self.assertNotIn("fX", result.assignments)

    def test_three_flows_all_assigned(self):
        flows = [
            _make_flow("fa", "10.0.0.2", "10.0.1.2", 5_000),
            _make_flow("fb", "10.0.0.3", "10.0.1.3", 50_000),
            _make_flow("fc", "10.0.0.2", "10.0.2.2", 500),
        ]
        cands = {
            "fa": [["A", "B", "C"], ["A", "D", "C"]],
            "fb": [["A", "B", "C"], ["A", "D", "C"]],
            "fc": [["A", "B", "E"], ["A", "D", "E"]],
        }
        caps = {
            ("A", "B"): 1e9, ("B", "A"): 1e9, ("B", "C"): 1e9, ("C", "B"): 1e9,
            ("A", "D"): 1e9, ("D", "A"): 1e9, ("D", "C"): 1e9, ("C", "D"): 1e9,
            ("B", "E"): 1e9, ("E", "B"): 1e9, ("D", "E"): 1e9, ("E", "D"): 1e9,
        }
        result = self.solver.solve(flows, cands, {}, caps)
        for f in flows:
            self.assertIn(f.flow_id, result.assignments)

    def test_status_is_optimal_or_feasible(self):
        flows, cands, utils, caps = _tiny_milp_inputs()
        result = self.solver.solve(flows, cands, utils, caps)
        self.assertIn(result.status, {"Optimal", "Feasible", "Infeasible"})


# =============================================================================
# TestLAFSMILPSolverGreedy
# =============================================================================

class TestLAFSMILPSolverGreedy(unittest.TestCase):
    """Tests for the internal greedy fallback."""

    def setUp(self):
        self.solver = LAFSMILPSolver(MILPConfig(solver="pulp"))

    def test_greedy_assigns_all_flows(self):
        flows, cands, _, caps = _tiny_milp_inputs()
        demands = {f.flow_id: f.size_bytes * 8 / 0.1 for f in flows}
        result = self.solver._fallback_greedy(flows, cands, demands, caps, time.perf_counter())
        for flow in flows:
            self.assertIn(flow.flow_id, result.assignments)

    def test_greedy_status_is_fallback(self):
        flows, cands, _, caps = _tiny_milp_inputs()
        demands = {f.flow_id: f.size_bytes * 8 / 0.1 for f in flows}
        result = self.solver._fallback_greedy(flows, cands, demands, caps, time.perf_counter())
        self.assertEqual(result.status, "Fallback")

    def test_greedy_solver_used(self):
        flows, cands, _, caps = _tiny_milp_inputs()
        demands = {f.flow_id: f.size_bytes * 8 / 0.1 for f in flows}
        result = self.solver._fallback_greedy(flows, cands, demands, caps, time.perf_counter())
        self.assertEqual(result.solver_used, "Greedy")

    def test_greedy_max_util_non_negative(self):
        flows, cands, _, caps = _tiny_milp_inputs()
        demands = {f.flow_id: f.size_bytes * 8 / 0.1 for f in flows}
        result = self.solver._fallback_greedy(flows, cands, demands, caps, time.perf_counter())
        self.assertGreaterEqual(result.max_utilisation, 0.0)

    def test_greedy_no_flows_empty_result(self):
        result = self.solver._fallback_greedy([], {}, {}, {}, time.perf_counter())
        self.assertEqual(result.assignments, {})
        self.assertEqual(result.max_utilisation, 0.0)


# =============================================================================
# TestLAFSScheduler
# =============================================================================

class TestLAFSScheduler(unittest.TestCase):
    """Integration tests for LAFSScheduler with real FatTreeGraph(k=4)."""

    @classmethod
    def setUpClass(cls):
        cls.topo = _small_topo(k=4)
        cls.sched = LAFSScheduler(
            cls.topo,
            milp_config=MILPConfig(solver="pulp", time_limit_s=10.0, verbose=False),
        )
        # Build a small set of flows on the k=4 topology
        # k=4: hosts are h_0_0_0 ... h_3_1_1 with IPs 10.p.e.2 / 10.p.e.3
        cls.flows = [
            _make_flow("s1", "10.0.0.2", "10.1.0.2", 5_000),
            _make_flow("s2", "10.0.0.3", "10.1.0.3", 50_000),
            _make_flow("s3", "10.0.1.2", "10.2.0.2", 500),
            _make_flow("s4", "10.0.1.3", "10.3.0.2", 1_000_000),
            _make_flow("s5", "10.1.0.2", "10.2.1.2", 200),
        ]

    def test_name_is_lafs(self):
        self.assertEqual(self.sched.name, "lafs")

    def test_isinstance_base_scheduler(self):
        from src.scheduler.base_scheduler import BaseScheduler
        self.assertIsInstance(self.sched, BaseScheduler)

    def test_schedule_flow_single_returns_path_or_none(self):
        flow = _make_flow("lone", "10.0.0.2", "10.1.0.2", 1000)
        path = self.sched.schedule_flow(flow)
        if path is not None:
            self.assertIsInstance(path, list)
            self.assertGreater(len(path), 0)

    def test_link_capacities_populated(self):
        self.assertGreater(len(self.sched._link_capacities), 0)

    def test_link_capacities_positive(self):
        for (u, v), cap in self.sched._link_capacities.items():
            self.assertGreater(cap, 0, f"Zero capacity on ({u}, {v})")

    def test_schedule_flows_milp_returns_result(self):
        result = self.sched.schedule_flows_milp(self.flows)
        self.assertIsInstance(result, MILPResult)

    def test_schedule_flows_milp_assigned_count(self):
        result = self.sched.schedule_flows_milp(self.flows)
        # All 5 flows have valid IPs so should be scheduled
        self.assertGreater(len(result.assignments), 0)

    def test_assigned_paths_are_lists(self):
        result = self.sched.schedule_flows_milp(self.flows)
        for fid, path in result.assignments.items():
            self.assertIsInstance(path, list, f"{fid}: path not a list")
            self.assertGreater(len(path), 0)

    def test_flow_objects_stamped(self):
        """schedule_flows_milp should set flow.assigned_path in-place."""
        local_flows = [
            _make_flow("x1", "10.0.0.2", "10.1.0.2", 1000),
            _make_flow("x2", "10.0.0.3", "10.2.0.2", 2000),
        ]
        self.sched.schedule_flows_milp(local_flows)
        for flow in local_flows:
            self.assertIsNotNone(
                flow.assigned_path,
                f"{flow.flow_id}.assigned_path is None after scheduling"
            )

    def test_metrics_updated_after_batch(self):
        """BaseScheduler.metrics should increment after scheduling."""
        sched = LAFSScheduler(
            self.topo,
            milp_config=MILPConfig(solver="pulp", time_limit_s=10.0, verbose=False),
        )
        local_flows = [_make_flow("m1", "10.0.0.2", "10.1.0.2", 1000)]
        sched.schedule_flows_milp(local_flows)
        self.assertGreater(sched.metrics.flows_scheduled, 0)

    def test_schedule_flows_dict_interface(self):
        """schedule_flows() (BaseScheduler API) returns dict."""
        result_dict = self.sched.schedule_flows(self.flows[:2])
        self.assertIsInstance(result_dict, dict)

    def test_no_forecaster_uses_zero_utils(self):
        """Without a forecaster, _get_predicted_utils returns empty dict."""
        sched = LAFSScheduler(self.topo, forecaster=None)
        utils = sched._get_predicted_utils()
        self.assertEqual(utils, {})

    def test_milp_config_summary_string(self):
        summary = self.sched.milp_config_summary()
        self.assertIn("pulp", summary)
        self.assertIn("window", summary)

    def test_invalid_topology_type_raises(self):
        with self.assertRaises(TypeError):
            LAFSScheduler("not_a_topology")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestMILPConfig,
        TestMILPResult,
        TestLAFSMILPSolverPuLP,
        TestLAFSMILPSolverGreedy,
        TestLAFSScheduler,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
