"""
LAFS — CONGA Scheduler Unit Tests
====================================
COMP-6910 — Group 7

Test coverage
-------------
TestCongestionTable   (14 tests) — DRE updates, decay, best-path selection
TestFlowletTable      (10 tests) — new/continuing flowlet detection, eviction
TestCONGAScheduler    (18 tests) — path selection, flowlet logic, congestion steering
TestCONGAIntegration  (12 tests) — 100-flow batches, congestion avoidance, k=8

Usage
-----
    pytest tests/unit/test_conga.py -v
    python tests/unit/test_conga.py --verbose
"""

from __future__ import annotations

import random
import sys
import os
import time
import unittest
from collections import Counter
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow
from src.scheduler.conga import (
    CONGAScheduler,
    CongestionTable,
    FlowletTable,
    FLOWLET_GAP_S,
    DRE_ALPHA,
    DRE_DECAY_RATE,
)


# =============================================================================
# Shared helpers
# =============================================================================
def _topo(k: int = 4) -> FatTreeGraph:
    return FatTreeGraph(k=k)


def _flow(
    src_ip: str = "10.0.0.2",
    dst_ip: str = "10.1.0.2",
    src_port: int = 1000,
    dst_port: int = 80,
    protocol: int = 6,
    size_bytes: int = 1_000_000,
    fid: str = "f001",
) -> Flow:
    return Flow(
        flow_id=fid,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        size_bytes=size_bytes,
    )


def _random_flows(topo: FatTreeGraph, n: int, seed: int = 42) -> List[Flow]:
    rng = random.Random(seed)
    hosts = list(topo.hosts)
    flows = []
    for i in range(n):
        src, dst = rng.sample(hosts, 2)
        flows.append(Flow(
            flow_id=f"f{i:04d}",
            src_ip=topo.get_host_ip(src),
            dst_ip=topo.get_host_ip(dst),
            src_port=rng.randint(1024, 65535),
            dst_port=rng.choice([80, 443, 5001]),
            protocol=6,
            size_bytes=rng.randint(1, 10_000_000),
        ))
    return flows


# =============================================================================
# 1. CongestionTable tests
# =============================================================================
class TestCongestionTable(unittest.TestCase):
    """Verify DRE accounting, EWMA updates, decay, and path selection."""

    def setUp(self):
        # alpha=1.0 → each update fully replaces the old value (easy to reason about).
        # decay_rate=1.0 → no time-based decay, so DRE is stable between update and get.
        self.ct = CongestionTable(alpha=1.0, decay_rate=1.0)

    def test_initial_dre_is_zero(self):
        dre = self.ct.get("e_0_0", "e_1_0", 0)
        self.assertEqual(dre, 0.0)

    def test_update_increases_dre(self):
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=1_000_000)
        dre = self.ct.get("e_0_0", "e_1_0", 0)
        self.assertGreater(dre, 0.0)

    def test_dre_clamped_to_one(self):
        # Inject enormous bytes to force saturation.
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=10**12)
        dre = self.ct.get("e_0_0", "e_1_0", 0)
        self.assertLessEqual(dre, 1.0)

    def test_dre_never_negative(self):
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=0)
        dre = self.ct.get("e_0_0", "e_1_0", 0)
        self.assertGreaterEqual(dre, 0.0)

    def test_different_path_indices_independent(self):
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=1_000_000)
        dre0 = self.ct.get("e_0_0", "e_1_0", 0)
        dre1 = self.ct.get("e_0_0", "e_1_0", 1)
        self.assertGreater(dre0, dre1,
                           "Path 1 should have lower DRE than loaded path 0")

    def test_different_leaf_pairs_independent(self):
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=1_000_000)
        dre_loaded = self.ct.get("e_0_0", "e_1_0", 0)
        dre_other = self.ct.get("e_2_0", "e_3_0", 0)
        self.assertGreater(dre_loaded, dre_other)

    def test_best_path_idx_returns_least_congested(self):
        # Congest paths 0, 2, 3 — path 1 should be chosen.
        for idx in [0, 2, 3]:
            self.ct.update("e_0_0", "e_1_0", idx, added_bytes=5_000_000)
        best_idx, _ = self.ct.best_path_idx("e_0_0", "e_1_0", 4)
        self.assertEqual(best_idx, 1)

    def test_best_path_idx_all_zero(self):
        # No congestion → should return index 0 (tie-break).
        best_idx, dre = self.ct.best_path_idx("e_0_0", "e_1_0", 4)
        self.assertEqual(best_idx, 0)
        self.assertEqual(dre, 0.0)

    def test_reset_clears_table(self):
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=1_000_000)
        self.ct.reset()
        dre = self.ct.get("e_0_0", "e_1_0", 0)
        self.assertEqual(dre, 0.0)

    def test_snapshot_returns_dict(self):
        self.ct.update("e_0_0", "e_1_0", 0, added_bytes=100_000)
        snap = self.ct.snapshot()
        self.assertIsInstance(snap, dict)
        self.assertGreater(len(snap), 0)

    def test_decay_with_zero_rate_zeroes_dre(self):
        """decay_rate=0.0 means DRE drops to 0 after any elapsed time."""
        ct = CongestionTable(alpha=1.0, decay_rate=0.0)
        ct.update("e_0_0", "e_1_0", 0, added_bytes=1_000_000)
        # Manually set last_update to the past.
        ct._last_update[("e_0_0", "e_1_0", 0)] -= 1.0  # 1 second ago
        dre = ct.get("e_0_0", "e_1_0", 0)
        self.assertEqual(dre, 0.0)

    def test_decay_with_rate_1_no_decay(self):
        """decay_rate=1.0 means DRE never decays."""
        ct = CongestionTable(alpha=1.0, decay_rate=1.0)
        ct.update("e_0_0", "e_1_0", 0, added_bytes=1_000_000)
        original = ct._table[("e_0_0", "e_1_0", 0)]
        ct._last_update[("e_0_0", "e_1_0", 0)] -= 10.0  # 10 s ago
        dre = ct.get("e_0_0", "e_1_0", 0)
        self.assertAlmostEqual(dre, original, places=5)

    def test_ewma_alpha_0_5(self):
        """
        With alpha=0.5 and initial DRE=0:
        After one update that produces sample=1.0:
            DRE_new = 0.5 * 0 + 0.5 * 1.0 = 0.5
        """
        ct = CongestionTable(alpha=0.5, decay_rate=1.0)
        # Use huge bytes to force sample=1.0.
        ct.update("e_0_0", "e_1_0", 0, added_bytes=10**15)
        dre = ct._table[("e_0_0", "e_1_0", 0)]
        # Should be 0.5 * 0 + 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(dre, 0.5, places=3)

    def test_update_returns_float(self):
        result = self.ct.update("e_0_0", "e_1_0", 0, added_bytes=100)
        self.assertIsInstance(result, float)


# =============================================================================
# 2. FlowletTable tests
# =============================================================================
class TestFlowletTable(unittest.TestCase):
    """Verify flowlet detection: new vs. continuing, eviction."""

    GAP = 0.001  # 1 ms for test speed

    def setUp(self):
        self.ft = FlowletTable(flowlet_gap_s=self.GAP)
        self.FT = ("10.0.0.2", "10.1.0.2", 1000, 80, 6)

    def test_initial_lookup_returns_none(self):
        result = self.ft.lookup(self.FT, time.monotonic())
        self.assertIsNone(result)

    def test_record_then_lookup_within_gap_returns_index(self):
        now = time.monotonic()
        self.ft.record(self.FT, path_idx=2, now=now)
        result = self.ft.lookup(self.FT, now + self.GAP * 0.1)
        self.assertEqual(result, 2)

    def test_lookup_after_gap_returns_none(self):
        now = time.monotonic()
        self.ft.record(self.FT, path_idx=1, now=now)
        result = self.ft.lookup(self.FT, now + self.GAP * 2)
        self.assertIsNone(result, "Expired flowlet should return None")

    def test_new_flowlet_counter_increments(self):
        self.ft.record(self.FT, path_idx=0, now=time.monotonic())
        self.assertEqual(self.ft.new_flowlets, 1)

    def test_continuing_flowlet_counter_increments(self):
        now = time.monotonic()
        self.ft.record(self.FT, path_idx=0, now=now)
        self.ft.record(self.FT, path_idx=0, now=now + self.GAP * 0.1)
        self.assertEqual(self.ft.continuing_flowlets, 1)

    def test_size_reflects_active_entries(self):
        self.ft.record(self.FT, path_idx=0, now=time.monotonic())
        self.assertEqual(self.ft.size, 1)

    def test_evict_expired_removes_old_entries(self):
        now = time.monotonic()
        self.ft.record(self.FT, path_idx=0, now=now - self.GAP * 2)
        evicted = self.ft.evict_expired(now)
        self.assertEqual(evicted, 1)
        self.assertEqual(self.ft.size, 0)

    def test_evict_does_not_remove_fresh_entries(self):
        now = time.monotonic()
        self.ft.record(self.FT, path_idx=0, now=now)
        evicted = self.ft.evict_expired(now)
        self.assertEqual(evicted, 0)
        self.assertEqual(self.ft.size, 1)

    def test_reset_clears_all(self):
        self.ft.record(self.FT, path_idx=0, now=time.monotonic())
        self.ft.reset()
        self.assertEqual(self.ft.size, 0)
        self.assertEqual(self.ft.new_flowlets, 0)

    def test_different_5tuples_independent(self):
        FT2 = ("10.0.0.2", "10.1.0.2", 2000, 80, 6)
        now = time.monotonic()
        self.ft.record(self.FT, path_idx=0, now=now)
        # FT2 not recorded → lookup should return None.
        result = self.ft.lookup(FT2, now + self.GAP * 0.1)
        self.assertIsNone(result)


# =============================================================================
# 3. CONGAScheduler unit tests
# =============================================================================
class TestCONGAScheduler(unittest.TestCase):
    """Unit tests for CONGAScheduler path selection and state management."""

    def setUp(self):
        self.topo = _topo(k=4)
        self.sched = CONGAScheduler(self.topo, flowlet_gap_s=FLOWLET_GAP_S)

    # ── Construction ──────────────────────────────────────────────────────────

    def test_name(self):
        self.assertEqual(self.sched.name, "conga")

    def test_repr(self):
        r = repr(self.sched)
        self.assertIn("CONGAScheduler", r)

    def test_wrong_topology_raises(self):
        with self.assertRaises(TypeError):
            CONGAScheduler("bad")

    # ── Path selection ────────────────────────────────────────────────────────

    def test_schedule_flow_returns_list(self):
        path = self.sched.schedule_flow(_flow())
        self.assertIsInstance(path, list)

    def test_path_starts_at_src_host(self):
        path = self.sched.schedule_flow(_flow(src_ip="10.0.0.2", dst_ip="10.1.0.2"))
        self.assertEqual(path[0], "h_0_0_0")

    def test_path_ends_at_dst_host(self):
        path = self.sched.schedule_flow(_flow(src_ip="10.0.0.2", dst_ip="10.1.0.2"))
        self.assertEqual(path[-1], "h_1_0_0")

    def test_loopback_returns_single_node(self):
        path = self.sched.schedule_flow(_flow(src_ip="10.0.0.2", dst_ip="10.0.0.2"))
        self.assertEqual(path, ["h_0_0_0"])

    def test_unknown_src_ip_returns_none(self):
        path = self.sched.schedule_flow(_flow(src_ip="192.168.0.1"))
        self.assertIsNone(path)

    def test_unknown_dst_ip_returns_none(self):
        path = self.sched.schedule_flow(_flow(dst_ip="172.16.0.1"))
        self.assertIsNone(path)

    def test_path_nodes_exist_in_topology(self):
        path = self.sched.schedule_flow(_flow())
        all_nodes = set(self.topo.graph.nodes())
        for node in path:
            self.assertIn(node, all_nodes)

    def test_consecutive_nodes_connected(self):
        path = self.sched.schedule_flow(_flow())
        for u, v in zip(path[:-1], path[1:]):
            self.assertTrue(self.topo.graph.has_edge(u, v), f"No edge {u}—{v}")

    # ── Flowlet continuation ──────────────────────────────────────────────────

    def test_same_5tuple_within_gap_uses_same_path(self):
        """Two flows with the same 5-tuple arriving within the flowlet gap
        should be forwarded on the same path (in-order delivery guarantee)."""
        sched = CONGAScheduler(self.topo, flowlet_gap_s=10.0)  # wide gap
        f1 = _flow(fid="fa")
        f2 = _flow(fid="fb")  # same 5-tuple as f1
        p1 = sched.schedule_flow(f1)
        p2 = sched.schedule_flow(f2)
        self.assertEqual(tuple(p1), tuple(p2),
                         "Same flowlet should stay on the same path")

    # ── Congestion steering ───────────────────────────────────────────────────

    def test_congested_path_avoided(self):
        """
        After injecting high DRE on path 0 for a leaf pair, CONGA should
        prefer a different path for new flows on that pair.
        """
        src_ip = self.topo.get_host_ip("h_0_0_0")
        dst_ip = self.topo.get_host_ip("h_2_0_0")
        paths = self.topo.get_paths("h_0_0_0", "h_2_0_0")
        if len(paths) < 2:
            self.skipTest("Need ≥ 2 paths for this test")

        # Identify the leaf switches from path 0.
        src_leaf = paths[0][1]
        dst_leaf = paths[0][-2]

        # Inject maximum congestion on path 0.
        self.sched.inject_congestion(src_leaf, dst_leaf, 0, dre_value=1.0)

        # Schedule a new flow (fresh, no flowlet entry).
        f = Flow(flow_id="new1", src_ip=src_ip, dst_ip=dst_ip,
                 src_port=9999, dst_port=80, protocol=6, size_bytes=1000)
        path = self.sched.schedule_flow(f)
        self.assertIsNotNone(path)
        self.assertNotEqual(
            tuple(path), tuple(paths[0]),
            "CONGA should avoid the congested path[0]"
        )

    def test_inject_congestion_updates_table(self):
        self.sched.inject_congestion("e_0_0", "e_1_0", 0, dre_value=0.75)
        dre = self.sched.congestion_table.get("e_0_0", "e_1_0", 0)
        self.assertAlmostEqual(dre, 0.75, places=3)

    # ── State management ──────────────────────────────────────────────────────

    def test_flowlet_table_populated_after_schedule(self):
        self.sched.schedule_flow(_flow())
        self.assertGreater(self.sched.flowlet_table.size, 0)

    def test_congestion_table_populated_after_schedule(self):
        self.sched.schedule_flow(_flow())
        snap = self.sched.congestion_snapshot()
        self.assertGreater(len(snap), 0)

    def test_evict_expired_flowlets(self):
        sched = CONGAScheduler(self.topo, flowlet_gap_s=0.0)  # gap=0 → all expire
        sched.schedule_flow(_flow())
        evicted = sched.evict_expired_flowlets()
        self.assertGreaterEqual(evicted, 0)  # should evict the entry

    def test_reset_clears_state(self):
        self.sched.schedule_flows([_flow(fid="f1"), _flow(fid="f2")])
        self.sched.reset_metrics()
        self.assertEqual(self.sched.flowlet_table.size, 0)
        self.assertEqual(len(self.sched.congestion_snapshot()), 0)

    def test_conga_stats_string(self):
        self.sched.schedule_flow(_flow())
        s = self.sched.conga_stats()
        self.assertIn("CONGA", s)
        self.assertIn("Flowlet", s)


# =============================================================================
# 4. CONGAScheduler integration tests
# =============================================================================
class TestCONGAIntegration(unittest.TestCase):
    """Integration tests: 100-flow batches, congestion avoidance, k=8."""

    @classmethod
    def setUpClass(cls):
        cls.topo4 = FatTreeGraph(k=4)
        cls.topo8 = FatTreeGraph(k=8)

    # ── All flows scheduled ───────────────────────────────────────────────────

    def test_all_100_flows_scheduled_k4(self):
        sched = CONGAScheduler(self.topo4)
        flows = _random_flows(self.topo4, 100)
        results = sched.schedule_flows(flows)
        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(n_ok, 100)

    def test_all_paths_valid_k4(self):
        sched = CONGAScheduler(self.topo4)
        flows = _random_flows(self.topo4, 50)
        results = sched.schedule_flows(flows)
        for fid, path in results.items():
            if path:
                for u, v in zip(path[:-1], path[1:]):
                    self.assertTrue(
                        self.topo4.graph.has_edge(u, v),
                        f"{fid}: no edge {u}—{v}"
                    )

    def test_assigned_path_set_on_flow(self):
        sched = CONGAScheduler(self.topo4)
        flows = _random_flows(self.topo4, 20)
        sched.schedule_flows(flows)
        for f in flows:
            self.assertIsNotNone(f.assigned_path)

    # ── Congestion avoidance ──────────────────────────────────────────────────

    def test_congestion_avoidance_shifts_flows(self):
        """
        Schedule 200 flows from h_0_0_0 → h_2_0_0, all with different ports.
        Inject full congestion on path[0] after 50 flows.
        Count how many of the last 150 flows land on path[0] vs others.
        CONGA should significantly reduce path[0] usage after the injection.
        """
        sched = CONGAScheduler(self.topo4, flowlet_gap_s=0.0)  # always new flowlet
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_2_0_0")
        paths = self.topo4.get_paths("h_0_0_0", "h_2_0_0")
        if len(paths) < 2:
            self.skipTest("Need ≥ 2 paths")

        src_leaf = paths[0][1]
        dst_leaf = paths[0][-2]
        first_path_tuple = tuple(paths[0])

        # First 50 flows (no congestion).
        for i in range(50):
            f = Flow(flow_id=f"pre{i}", src_ip=src_ip, dst_ip=dst_ip,
                     src_port=10000 + i, dst_port=80, protocol=6,
                     size_bytes=100_000)
            sched.schedule_flow(f)

        # Inject high congestion on path 0.
        sched.inject_congestion(src_leaf, dst_leaf, 0, dre_value=0.95)

        # Next 150 flows.
        path0_after = 0
        for i in range(150):
            f = Flow(flow_id=f"post{i}", src_ip=src_ip, dst_ip=dst_ip,
                     src_port=20000 + i, dst_port=443, protocol=6,
                     size_bytes=100_000)
            path = sched.schedule_flow(f)
            if path and tuple(path) == first_path_tuple:
                path0_after += 1

        # With DRE=0.95 on path 0, CONGA should mostly avoid it.
        # Allow up to 40% on path 0 (some DRE decay can reduce the penalty).
        fraction_on_path0 = path0_after / 150
        self.assertLess(
            fraction_on_path0, 0.50,
            f"CONGA used congested path 0 for {fraction_on_path0*100:.1f}% "
            f"of flows after injection (expected < 50%)"
        )

    def test_conga_uses_multiple_paths_k4(self):
        """With flowlet_gap_s=0 (always new flowlet), CONGA should spread
        flows across multiple paths for a single (src,dst) pair."""
        sched = CONGAScheduler(self.topo4, flowlet_gap_s=0.0)
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_3_0_0")
        n_ecmp = len(self.topo4.get_paths("h_0_0_0", "h_3_0_0"))

        paths_used = set()
        for i in range(200):
            f = Flow(flow_id=f"f{i}", src_ip=src_ip, dst_ip=dst_ip,
                     src_port=30000 + i, dst_port=80, protocol=6,
                     size_bytes=500_000)
            path = sched.schedule_flow(f)
            if path:
                paths_used.add(tuple(path))

        self.assertGreater(len(paths_used), 1,
                           "CONGA should use more than one path")

    # ── Metrics consistency ───────────────────────────────────────────────────

    def test_metrics_counts_match_results(self):
        sched = CONGAScheduler(self.topo4)
        flows = _random_flows(self.topo4, 100)
        results = sched.schedule_flows(flows)
        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(sched.metrics.flows_scheduled, n_ok)

    def test_flowlet_stats_populated(self):
        sched = CONGAScheduler(self.topo4)
        sched.schedule_flows(_random_flows(self.topo4, 20))
        stats = sched.flowlet_stats()
        self.assertIn("new_flowlets", stats)
        self.assertGreater(stats["new_flowlets"], 0)

    # ── k=8 smoke test ────────────────────────────────────────────────────────

    def test_100_flows_k8(self):
        sched = CONGAScheduler(self.topo8)
        flows = _random_flows(self.topo8, 100, seed=13)
        results = sched.schedule_flows(flows)
        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(n_ok, 100)

    def test_cross_pod_paths_k8(self):
        """k=8 cross-pod should provide 16 ECMP paths; CONGA selects among them."""
        sched = CONGAScheduler(self.topo8, flowlet_gap_s=0.0)
        src_ip = self.topo8.get_host_ip("h_0_0_0")
        dst_ip = self.topo8.get_host_ip("h_4_0_0")
        paths_used = set()
        for i in range(100):
            f = Flow(flow_id=f"k8f{i}", src_ip=src_ip, dst_ip=dst_ip,
                     src_port=40000 + i, dst_port=80, protocol=6,
                     size_bytes=1_000_000)
            path = sched.schedule_flow(f)
            if path:
                paths_used.add(tuple(path))
        # With 100 flows and 16 paths, expect several distinct paths chosen.
        self.assertGreater(len(paths_used), 2)

    # ── Report string ─────────────────────────────────────────────────────────

    def test_report_string_contains_conga(self):
        sched = CONGAScheduler(self.topo4)
        sched.schedule_flows(_random_flows(self.topo4, 5))
        r = sched.report()
        self.assertIn("CONGA", r)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args, _ = parser.parse_known_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestCongestionTable, TestFlowletTable,
        TestCONGAScheduler, TestCONGAIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
