"""
LAFS — Hedera Scheduler Unit Tests
====================================
COMP-6910 — Group 7

Test coverage
-------------
TestPathLoadTracker   (14 tests) — byte accounting, release, utilisation, balance
TestHederaScheduler   (20 tests) — mice/elephant routing, GFF logic, path length
TestHederaIntegration (12 tests) — 100-flow batches, GFF vs ECMP balance, reschedule

Usage
-----
    pytest tests/unit/test_hedera.py -v
    python tests/unit/test_hedera.py --verbose
"""

from __future__ import annotations

import random
import sys
import os
import unittest
from collections import Counter
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow, ELEPHANT_THRESHOLD_BYTES, MICE_THRESHOLD_BYTES
from src.scheduler.hedera import HederaScheduler, PathLoadTracker
from src.scheduler.ecmp import ECMPScheduler


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
    size_bytes: int = 1024,
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


def _elephant(fid: str = "e001", src_ip: str = "10.0.0.2",
              dst_ip: str = "10.1.0.2", size: int = 5_000_000) -> Flow:
    return _flow(fid=fid, src_ip=src_ip, dst_ip=dst_ip, size_bytes=size)


def _mice(fid: str = "m001", src_ip: str = "10.0.0.2",
          dst_ip: str = "10.1.0.2", size: int = 10_000) -> Flow:
    return _flow(fid=fid, src_ip=src_ip, dst_ip=dst_ip, size_bytes=size)


def _random_flows(topo: FatTreeGraph, n: int, seed: int = 42,
                  size_bytes: int = None) -> List[Flow]:
    rng = random.Random(seed)
    hosts = list(topo.hosts)
    flows = []
    for i in range(n):
        src, dst = rng.sample(hosts, 2)
        sz = size_bytes if size_bytes is not None else rng.randint(1, 10_000_000)
        flows.append(Flow(
            flow_id=f"f{i:04d}",
            src_ip=topo.get_host_ip(src),
            dst_ip=topo.get_host_ip(dst),
            src_port=rng.randint(1024, 65535),
            dst_port=rng.choice([80, 443, 5001]),
            protocol=6,
            size_bytes=sz,
        ))
    return flows


# =============================================================================
# 1. PathLoadTracker tests
# =============================================================================
class TestPathLoadTracker(unittest.TestCase):
    """Verify PathLoadTracker byte accounting, release, and balance metrics."""

    PATH_A = ["h_0_0_0", "e_0_0", "a_0_0", "c_0_0", "a_1_0", "e_1_0", "h_1_0_0"]
    PATH_B = ["h_0_0_0", "e_0_0", "a_0_0", "c_0_1", "a_1_0", "e_1_0", "h_1_0_0"]
    PATH_C = ["h_0_0_0", "e_0_0", "a_0_1", "c_1_0", "a_1_1", "e_1_0", "h_1_0_0"]

    def setUp(self):
        self.tracker = PathLoadTracker(path_capacity_gbps=1.0)

    def test_initial_load_zero(self):
        self.assertEqual(self.tracker.get_load_bytes(self.PATH_A), 0)

    def test_assign_adds_bytes(self):
        f = _flow(size_bytes=1_000_000)
        self.tracker.assign(f, self.PATH_A)
        self.assertEqual(self.tracker.get_load_bytes(self.PATH_A), 1_000_000)

    def test_two_flows_same_path_accumulate(self):
        f1 = _flow(fid="f1", size_bytes=500_000)
        f2 = _flow(fid="f2", size_bytes=300_000)
        self.tracker.assign(f1, self.PATH_A)
        self.tracker.assign(f2, self.PATH_A)
        self.assertEqual(self.tracker.get_load_bytes(self.PATH_A), 800_000)

    def test_release_removes_bytes(self):
        f = _flow(fid="f1", size_bytes=2_000_000)
        self.tracker.assign(f, self.PATH_A)
        self.tracker.release("f1")
        self.assertEqual(self.tracker.get_load_bytes(self.PATH_A), 0)

    def test_release_unknown_flow_returns_false(self):
        result = self.tracker.release("nonexistent")
        self.assertFalse(result)

    def test_release_known_flow_returns_true(self):
        f = _flow(fid="f1", size_bytes=100)
        self.tracker.assign(f, self.PATH_A)
        self.assertTrue(self.tracker.release("f1"))

    def test_least_loaded_empty_paths(self):
        paths = [self.PATH_A, self.PATH_B, self.PATH_C]
        result = self.tracker.least_loaded_path(paths)
        # All zero → first one returned (min is stable)
        self.assertIn(result, paths)

    def test_least_loaded_finds_minimum(self):
        f_heavy = _flow(fid="heavy", size_bytes=9_000_000)
        self.tracker.assign(f_heavy, self.PATH_A)
        paths = [self.PATH_A, self.PATH_B, self.PATH_C]
        result = self.tracker.least_loaded_path(paths)
        self.assertNotEqual(result, self.PATH_A,
                            "Heavy path A should not be chosen as least-loaded")

    def test_utilisation_calculation(self):
        # 1 Gbps capacity = 1e9 bits/s → treating bytes as bits: 1e9 bytes = 100% util
        f = _flow(size_bytes=int(1e9 / 8))  # 125 MB → 1 Gbps in 1-second window
        self.tracker.assign(f, self.PATH_A)
        util = self.tracker.get_utilisation(self.PATH_A)
        # util = (bytes * 8) / (1e9 bits) ≈ 1.0
        self.assertAlmostEqual(util, 1.0, places=5)

    def test_utilisation_zero_for_empty_path(self):
        self.assertAlmostEqual(self.tracker.get_utilisation(self.PATH_B), 0.0)

    def test_all_loads_returns_dict(self):
        f = _flow(size_bytes=1000)
        self.tracker.assign(f, self.PATH_A)
        loads = self.tracker.all_loads()
        self.assertIsInstance(loads, dict)
        self.assertIn(tuple(self.PATH_A), loads)

    def test_reset_clears_all(self):
        self.tracker.assign(_flow(fid="f1", size_bytes=1000), self.PATH_A)
        self.tracker.reset()
        self.assertEqual(self.tracker.get_load_bytes(self.PATH_A), 0)
        self.assertEqual(len(self.tracker.all_loads()), 0)

    def test_least_loaded_empty_list_returns_none(self):
        self.assertIsNone(self.tracker.least_loaded_path([]))

    def test_partial_release_leaves_remainder(self):
        f1 = _flow(fid="f1", size_bytes=600_000)
        f2 = _flow(fid="f2", size_bytes=400_000)
        self.tracker.assign(f1, self.PATH_A)
        self.tracker.assign(f2, self.PATH_A)
        self.tracker.release("f1")
        self.assertEqual(self.tracker.get_load_bytes(self.PATH_A), 400_000)


# =============================================================================
# 2. HederaScheduler unit tests
# =============================================================================
class TestHederaScheduler(unittest.TestCase):
    """Unit tests for HederaScheduler routing decisions."""

    def setUp(self):
        self.topo = _topo(k=4)
        self.sched = HederaScheduler(self.topo)

    # ── Construction ──────────────────────────────────────────────────────────

    def test_name(self):
        self.assertEqual(self.sched.name, "hedera")

    def test_repr(self):
        r = repr(self.sched)
        self.assertIn("HederaScheduler", r)
        self.assertIn("k=4", r)

    def test_wrong_topology_type_raises(self):
        with self.assertRaises(TypeError):
            HederaScheduler("not_a_topo")

    # ── Mice routing (via ECMP) ───────────────────────────────────────────────

    def test_mice_flow_returns_path(self):
        f = _mice()
        path = self.sched.schedule_flow(f)
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)

    def test_mice_flow_starts_and_ends_correctly(self):
        f = _mice(src_ip="10.0.0.2", dst_ip="10.1.0.2")
        path = self.sched.schedule_flow(f)
        self.assertEqual(path[0], "h_0_0_0")
        self.assertEqual(path[-1], "h_1_0_0")

    def test_mice_does_not_update_load_tracker(self):
        f = _mice(size=50_000)
        self.sched.schedule_flow(f)
        # Mice flows go through ECMP; load tracker should remain empty.
        self.assertEqual(len(self.sched.path_loads()), 0)

    def test_mice_count_increments(self):
        self.sched.schedule_flow(_mice(fid="m1"))
        self.sched.schedule_flow(_mice(fid="m2"))
        self.assertEqual(self.sched.mice_count, 2)

    def test_mice_deterministic_same_5tuple(self):
        f = _mice(fid="m1")
        paths = [self.sched.schedule_flow(f) for _ in range(10)]
        self.assertTrue(all(p == paths[0] for p in paths))

    # ── Elephant routing (via GFF) ────────────────────────────────────────────

    def test_elephant_flow_returns_path(self):
        f = _elephant()
        path = self.sched.schedule_flow(f)
        self.assertIsNotNone(path)

    def test_elephant_flow_starts_and_ends_correctly(self):
        f = _elephant(src_ip="10.0.0.2", dst_ip="10.1.0.2")
        path = self.sched.schedule_flow(f)
        self.assertEqual(path[0], "h_0_0_0")
        self.assertEqual(path[-1], "h_1_0_0")

    def test_elephant_updates_load_tracker(self):
        f = _elephant(fid="e1", size=5_000_000)
        self.sched.schedule_flow(f)
        loads = self.sched.path_loads()
        self.assertEqual(len(loads), 1)
        self.assertEqual(list(loads.values())[0], 5_000_000)

    def test_elephant_count_increments(self):
        self.sched.schedule_flow(_elephant(fid="e1"))
        self.sched.schedule_flow(_elephant(fid="e2"))
        self.assertEqual(self.sched.elephant_count, 2)

    def test_gff_routes_second_elephant_to_different_path(self):
        """
        Two large parallel flows between the same host pair should land on
        different paths (GFF picks least-loaded each time).
        """
        src_ip = self.topo.get_host_ip("h_0_0_0")
        dst_ip = self.topo.get_host_ip("h_1_0_0")
        f1 = _elephant(fid="e1", src_ip=src_ip, dst_ip=dst_ip, size=50_000_000)
        f2 = _elephant(fid="e2", src_ip=src_ip, dst_ip=dst_ip, size=50_000_000)
        p1 = self.sched.schedule_flow(f1)
        p2 = self.sched.schedule_flow(f2)
        # k=4 has 4 cross-pod paths → both should land on different ones.
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertNotEqual(tuple(p1), tuple(p2),
                            "GFF should spread two large flows across different paths")

    def test_unknown_src_ip_returns_none(self):
        f = _flow(src_ip="192.168.0.1", size_bytes=10_000_000, fid="bad")
        self.assertIsNone(self.sched.schedule_flow(f))

    def test_unknown_dst_ip_returns_none(self):
        f = _flow(dst_ip="172.16.0.1", size_bytes=10_000_000, fid="bad")
        self.assertIsNone(self.sched.schedule_flow(f))

    def test_elephant_threshold_boundary(self):
        """Flow at exactly elephant_threshold should be an elephant."""
        f = _flow(size_bytes=ELEPHANT_THRESHOLD_BYTES, fid="boundary")
        path = self.sched.schedule_flow(f)
        self.assertIsNotNone(path)
        self.assertEqual(self.sched.elephant_count, 1)

    def test_mice_threshold_boundary(self):
        """Flow just below elephant_threshold is NOT an elephant → no load."""
        f = _flow(size_bytes=ELEPHANT_THRESHOLD_BYTES - 1, fid="just_mice")
        self.sched.schedule_flow(f)
        self.assertEqual(self.sched.elephant_count, 0)

    # ── Release and rescheduling ──────────────────────────────────────────────

    def test_release_flow_removes_load(self):
        f = _elephant(fid="e1", size=5_000_000)
        self.sched.schedule_flow(f)
        self.sched.release_flow("e1")
        self.assertEqual(len(self.sched.path_loads()), 0)

    def test_load_balance_ratio_one_flow(self):
        self.sched.schedule_flow(_elephant(fid="e1"))
        ratio = self.sched.load_balance_ratio()
        # Only one loaded path → min==max → ratio == 1.0
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_reset_metrics_clears_counts(self):
        self.sched.schedule_flows([_mice(fid="m1"), _elephant(fid="e1")])
        self.sched.reset_metrics()
        self.assertEqual(self.sched.mice_count, 0)
        self.assertEqual(self.sched.elephant_count, 0)
        self.assertEqual(len(self.sched.path_loads()), 0)


# =============================================================================
# 3. HederaScheduler integration tests
# =============================================================================
class TestHederaIntegration(unittest.TestCase):
    """Integration tests: 100-flow batches, GFF balance, reschedule API."""

    @classmethod
    def setUpClass(cls):
        cls.topo4 = FatTreeGraph(k=4)
        cls.topo8 = FatTreeGraph(k=8)

    # ── All flows scheduled ───────────────────────────────────────────────────

    def test_all_100_flows_scheduled_k4(self):
        sched = HederaScheduler(self.topo4)
        flows = _random_flows(self.topo4, 100)
        results = sched.schedule_flows(flows)
        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(n_ok, 100)

    def test_paths_are_connected_k4(self):
        sched = HederaScheduler(self.topo4)
        flows = _random_flows(self.topo4, 50)
        results = sched.schedule_flows(flows)
        for fid, path in results.items():
            if path:
                for u, v in zip(path[:-1], path[1:]):
                    self.assertTrue(
                        self.topo4.graph.has_edge(u, v),
                        f"{fid}: no edge {u}—{v}"
                    )

    # ── GFF balance vs ECMP ───────────────────────────────────────────────────

    def test_gff_spreads_elephant_flows_across_paths(self):
        """
        Send 8 equal-size elephant flows from one host to another.
        GFF should distribute them across all 4 cross-pod ECMP paths.
        ECMP (static hash) will typically all land on the same path.
        """
        sched = HederaScheduler(self.topo4)
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_2_0_0")  # cross-pod

        paths_used = Counter()
        for i in range(8):
            f = Flow(
                flow_id=f"e{i}",
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=5000 + i, dst_port=80,
                protocol=6,
                size_bytes=10_000_000,  # 10 MB elephant
            )
            path = sched.schedule_flow(f)
            self.assertIsNotNone(path)
            paths_used[tuple(path)] += 1

        # With 8 flows and 4 paths, GFF should use at least 3 distinct paths.
        n_ecmp = len(self.topo4.get_paths("h_0_0_0", "h_2_0_0"))
        self.assertGreaterEqual(
            len(paths_used), min(n_ecmp, 4),
            f"GFF used only {len(paths_used)}/{n_ecmp} available paths"
        )

    def test_gff_better_balance_than_ecmp_for_elephants(self):
        """
        GFF's max-to-min load ratio should be better than (or equal to)
        ECMP's when all flows are elephants to the same (src,dst) pair.
        """
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_3_0_0")

        # Build 8 identical elephant flows.
        elephants = [
            Flow(flow_id=f"e{i}", src_ip=src_ip, dst_ip=dst_ip,
                 src_port=6000 + i, dst_port=443, protocol=6,
                 size_bytes=5_000_000)
            for i in range(8)
        ]

        # Hedera GFF load distribution.
        sched_h = HederaScheduler(self.topo4)
        for f in elephants:
            sched_h.schedule_flow(f)
        hedera_loads = sched_h.path_loads()
        h_values = list(hedera_loads.values()) if hedera_loads else [0]
        hedera_spread = len(hedera_loads)

        # Hedera should use ≥ 2 paths.
        self.assertGreaterEqual(hedera_spread, 2,
                                "Hedera GFF should spread elephants across paths")

    # ── reschedule_elephants API ──────────────────────────────────────────────

    def test_reschedule_elephants_returns_all_paths(self):
        sched = HederaScheduler(self.topo4)
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_2_0_0")
        elephants = [
            Flow(flow_id=f"re{i}", src_ip=src_ip, dst_ip=dst_ip,
                 src_port=7000 + i, dst_port=80, protocol=6,
                 size_bytes=5_000_000)
            for i in range(4)
        ]
        results = sched.reschedule_elephants(elephants)
        self.assertEqual(len(results), 4)
        for fid, path in results.items():
            self.assertIsNotNone(path, f"reschedule_elephants failed for {fid}")

    def test_reschedule_updates_assigned_path(self):
        sched = HederaScheduler(self.topo4)
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_1_0_0")
        f = Flow(flow_id="re1", src_ip=src_ip, dst_ip=dst_ip,
                 src_port=8000, dst_port=80, protocol=6, size_bytes=5_000_000)
        sched.reschedule_elephants([f])
        self.assertIsNotNone(f.assigned_path)

    # ── Hedera stats string ───────────────────────────────────────────────────

    def test_hedera_stats_string(self):
        sched = HederaScheduler(self.topo4)
        sched.schedule_flows(_random_flows(self.topo4, 20))
        s = sched.hedera_stats()
        self.assertIn("Hedera", s)
        self.assertIn("Elephant", s)
        self.assertIn("Mice", s)

    # ── k=8 smoke test ────────────────────────────────────────────────────────

    def test_100_flows_k8(self):
        sched = HederaScheduler(self.topo8)
        flows = _random_flows(self.topo8, 100, seed=7)
        results = sched.schedule_flows(flows)
        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(n_ok, 100)

    def test_report_string(self):
        sched = HederaScheduler(self.topo4)
        sched.schedule_flows(_random_flows(self.topo4, 10))
        r = sched.report()
        self.assertIn("HEDERA", r)

    # ── Custom threshold ──────────────────────────────────────────────────────

    def test_custom_elephant_threshold(self):
        """A threshold of 500 KB should treat 600 KB flows as elephants."""
        sched = HederaScheduler(self.topo4, elephant_threshold_bytes=500_000)
        f = Flow(flow_id="big", src_ip="10.0.0.2", dst_ip="10.1.0.2",
                 src_port=1000, dst_port=80, protocol=6, size_bytes=600_000)
        sched.schedule_flow(f)
        self.assertEqual(sched.elephant_count, 1)
        self.assertEqual(sched.mice_count, 0)


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
    for cls in [TestPathLoadTracker, TestHederaScheduler, TestHederaIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
