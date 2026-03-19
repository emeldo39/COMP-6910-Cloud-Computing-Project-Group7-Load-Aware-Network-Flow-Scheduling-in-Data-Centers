"""
LAFS — Scheduler Unit & Integration Tests
==========================================
COMP 6910 — Group 7

Test coverage
-------------
TestFlowDataclass          (20 tests) — Flow construction, validation, properties
TestEcmpHashFunction       (12 tests) — ecmp_hash correctness and distribution
TestBaseSchedulerInterface  (8 tests) — BaseScheduler contract via ECMPScheduler
TestEcmpSchedulerUnit      (18 tests) — ECMPScheduler path selection and stats
TestEcmpIntegration        (10 tests) — 100-flow batch, distribution, k=4 and k=8

Usage
-----
    pytest tests/unit/test_scheduler.py -v
    python tests/unit/test_scheduler.py --verbose
"""

from __future__ import annotations

import random
import sys
import time
import unittest
from collections import Counter
from typing import List

# ── Ensure project root is on sys.path when run directly ─────────────────────
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.topology.fattree import FatTreeGraph
from src.workload.flow import (
    Flow,
    MICE_THRESHOLD_BYTES,
    ELEPHANT_THRESHOLD_BYTES,
    VALID_PROTOCOLS,
)
from src.scheduler.ecmp import ECMPScheduler, ecmp_hash, _ip_to_uint32
from src.scheduler.base_scheduler import BaseScheduler, SchedulerMetrics


# =============================================================================
# Shared fixtures
# =============================================================================
def _make_topo(k: int = 4) -> FatTreeGraph:
    return FatTreeGraph(k=k)


def _make_flow(
    src_ip: str = "10.0.0.2",
    dst_ip: str = "10.1.0.2",
    src_port: int = 12345,
    dst_port: int = 80,
    protocol: int = 6,
    size_bytes: int = 1024,
    flow_id: str = "test0001",
) -> Flow:
    return Flow(
        flow_id=flow_id,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        size_bytes=size_bytes,
    )


def _random_flow(
    topo: FatTreeGraph,
    rng: random.Random,
    flow_id: str = None,
    protocol: int = None,
    size_bytes: int = None,
) -> Flow:
    """Return a Flow with source and destination randomly chosen from topology hosts."""
    hosts = list(topo.hosts)
    src_host, dst_host = rng.sample(hosts, 2)
    return Flow(
        flow_id=flow_id or Flow.new_id(),
        src_ip=topo.get_host_ip(src_host),
        dst_ip=topo.get_host_ip(dst_host),
        src_port=rng.randint(1024, 65535),
        dst_port=rng.choice([80, 443, 8080, 5001, 22]),
        protocol=protocol if protocol is not None else rng.choice([6, 17]),
        size_bytes=size_bytes if size_bytes is not None else rng.randint(1, 10_000_000),
    )


# =============================================================================
# 1. Flow dataclass tests
# =============================================================================
class TestFlowDataclass(unittest.TestCase):
    """Verify Flow construction, validation, and all derived properties."""

    # ── Construction ──────────────────────────────────────────────────────────

    def test_basic_construction(self):
        f = _make_flow()
        self.assertEqual(f.flow_id, "test0001")
        self.assertEqual(f.src_ip, "10.0.0.2")
        self.assertEqual(f.dst_ip, "10.1.0.2")
        self.assertEqual(f.src_port, 12345)
        self.assertEqual(f.dst_port, 80)
        self.assertEqual(f.protocol, 6)
        self.assertEqual(f.size_bytes, 1024)

    def test_default_scheduler_fields_are_none(self):
        f = _make_flow()
        self.assertIsNone(f.assigned_path)
        self.assertIsNone(f.schedule_time)
        self.assertIsNone(f.completion_time)

    def test_arrival_time_set_automatically(self):
        before = time.time()
        f = _make_flow()
        after = time.time()
        self.assertGreaterEqual(f.arrival_time, before)
        self.assertLessEqual(f.arrival_time, after)

    def test_create_factory(self):
        f = Flow.create("10.0.0.2", "10.2.0.2", 5000, 443)
        self.assertIsNotNone(f.flow_id)
        self.assertEqual(len(f.flow_id), 8)
        self.assertEqual(f.protocol, 6)   # default TCP
        self.assertEqual(f.size_bytes, 1024)  # default

    def test_new_id_unique(self):
        ids = {Flow.new_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    # ── 5-tuple ───────────────────────────────────────────────────────────────

    def test_five_tuple_content(self):
        f = _make_flow()
        self.assertEqual(
            f.five_tuple,
            ("10.0.0.2", "10.1.0.2", 12345, 80, 6),
        )

    def test_five_tuple_is_tuple(self):
        f = _make_flow()
        self.assertIsInstance(f.five_tuple, tuple)
        self.assertEqual(len(f.five_tuple), 5)

    # ── Flow-type classification ──────────────────────────────────────────────

    def test_mice_flow(self):
        f = _make_flow(size_bytes=MICE_THRESHOLD_BYTES - 1)
        self.assertTrue(f.is_mice)
        self.assertFalse(f.is_elephant)
        self.assertEqual(f.flow_type, "mice")

    def test_mice_boundary(self):
        f_mice = _make_flow(size_bytes=MICE_THRESHOLD_BYTES - 1)
        f_not = _make_flow(size_bytes=MICE_THRESHOLD_BYTES)
        self.assertTrue(f_mice.is_mice)
        self.assertFalse(f_not.is_mice)

    def test_elephant_flow(self):
        f = _make_flow(size_bytes=ELEPHANT_THRESHOLD_BYTES)
        self.assertFalse(f.is_mice)
        self.assertTrue(f.is_elephant)
        self.assertEqual(f.flow_type, "elephant")

    def test_medium_flow(self):
        f = _make_flow(size_bytes=500_000)
        self.assertFalse(f.is_mice)
        self.assertFalse(f.is_elephant)
        self.assertEqual(f.flow_type, "medium")

    # ── FCT / slowdown ────────────────────────────────────────────────────────

    def test_fct_none_before_completion(self):
        f = _make_flow()
        self.assertIsNone(f.fct)
        self.assertIsNone(f.slowdown)

    def test_fct_computed_correctly(self):
        f = _make_flow()
        f.schedule_time = 100.0
        f.completion_time = 100.5
        self.assertAlmostEqual(f.fct, 0.5, places=9)

    def test_slowdown_computed(self):
        f = _make_flow(size_bytes=1_000_000_000 // 8)  # exactly 1 second at 1 Gbps
        f.schedule_time = 0.0
        f.completion_time = 2.0   # 2× the ideal → slowdown = 2.0
        self.assertAlmostEqual(f.slowdown, 2.0, places=6)

    def test_ideal_fct_formula(self):
        # 1 Gbps → 1 bit/ns → 1,000,000,000 bits/s
        # 1,000,000 bytes × 8 bits = 8,000,000 bits → 8 ms
        f = _make_flow(size_bytes=1_000_000)
        self.assertAlmostEqual(f.ideal_fct, 0.008, places=9)

    def test_meets_deadline(self):
        f = _make_flow()
        f.arrival_time = 0.0
        f.deadline = 1.0
        f.completion_time = 0.9
        f.schedule_time = 0.0
        self.assertTrue(f.meets_deadline)
        f.completion_time = 1.1
        self.assertFalse(f.meets_deadline)

    # ── Protocol helpers ──────────────────────────────────────────────────────

    def test_protocol_names(self):
        self.assertEqual(_make_flow(protocol=6).protocol_name, "TCP")
        self.assertEqual(_make_flow(protocol=17).protocol_name, "UDP")
        self.assertEqual(_make_flow(protocol=1).protocol_name, "ICMP")

    # ── Validation ────────────────────────────────────────────────────────────

    def test_invalid_src_ip_raises(self):
        with self.assertRaises(ValueError):
            _make_flow(src_ip="999.0.0.1")

    def test_invalid_dst_ip_raises(self):
        with self.assertRaises(ValueError):
            _make_flow(dst_ip="not-an-ip")

    def test_port_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _make_flow(src_port=70000)
        with self.assertRaises(ValueError):
            _make_flow(dst_port=-1)

    def test_invalid_protocol_raises(self):
        with self.assertRaises(ValueError):
            _make_flow(protocol=99)

    def test_negative_size_raises(self):
        with self.assertRaises(ValueError):
            _make_flow(size_bytes=-1)


# =============================================================================
# 2. ECMP hash function tests
# =============================================================================
class TestEcmpHashFunction(unittest.TestCase):
    """Verify ecmp_hash correctness, determinism, and distribution."""

    # ── Correctness ───────────────────────────────────────────────────────────

    def test_returns_int(self):
        h = ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
        self.assertIsInstance(h, int)

    def test_range_is_uint32(self):
        for _ in range(500):
            h = ecmp_hash("10.0.0.2", "10.1.0.2",
                          random.randint(0, 65535),
                          random.randint(0, 65535), 6)
            self.assertGreaterEqual(h, 0)
            self.assertLessEqual(h, 0xFFFF_FFFF)

    def test_deterministic_same_input(self):
        h1 = ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
        h2 = ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
        self.assertEqual(h1, h2)

    def test_different_src_port_gives_different_hash(self):
        h1 = ecmp_hash("10.0.0.2", "10.1.0.2", 1000, 80, 6)
        h2 = ecmp_hash("10.0.0.2", "10.1.0.2", 1001, 80, 6)
        self.assertNotEqual(h1, h2)

    def test_different_dst_port_gives_different_hash(self):
        h1 = ecmp_hash("10.0.0.2", "10.1.0.2", 1000, 80, 6)
        h2 = ecmp_hash("10.0.0.2", "10.1.0.2", 1000, 81, 6)
        self.assertNotEqual(h1, h2)

    def test_different_protocol_gives_different_hash(self):
        h_tcp = ecmp_hash("10.0.0.2", "10.1.0.2", 1000, 80, 6)
        h_udp = ecmp_hash("10.0.0.2", "10.1.0.2", 1000, 80, 17)
        self.assertNotEqual(h_tcp, h_udp)

    def test_ip_to_uint32_known_value(self):
        # 10.0.0.2 = 0x0A000002
        self.assertEqual(_ip_to_uint32("10.0.0.2"), 0x0A000002)
        # 10.1.2.3 = 0x0A010203
        self.assertEqual(_ip_to_uint32("10.1.2.3"), 0x0A010203)

    def test_symmetry_broken(self):
        """Hash(A→B) should differ from Hash(B→A) because IP positions differ."""
        h_fwd = ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
        h_rev = ecmp_hash("10.1.0.2", "10.0.0.2", 80, 12345, 6)
        self.assertNotEqual(h_fwd, h_rev)

    def test_distribution_uniformity_4_buckets(self):
        """
        With 2000 flows and random ports, hash % 4 should produce roughly
        equal counts in each bucket (within ±25% of expected 500).
        """
        rng = random.Random(42)
        n_buckets = 4
        n_flows = 2000
        counts = Counter()
        for _ in range(n_flows):
            h = ecmp_hash(
                "10.0.0.2", "10.1.0.2",
                rng.randint(1024, 65535),
                rng.randint(1, 1023),
                6,
            )
            counts[h % n_buckets] += 1

        expected = n_flows / n_buckets
        tolerance = 0.30  # ±30% is acceptable for CRC32 with 4 buckets
        for bucket, count in counts.items():
            self.assertAlmostEqual(
                count, expected, delta=expected * tolerance,
                msg=f"Bucket {bucket} count={count} deviates > {tolerance*100:.0f}% from {expected}",
            )

    def test_distribution_uniformity_16_buckets(self):
        """With 4000 flows, hash % 16 should fill all 16 buckets."""
        rng = random.Random(99)
        n_buckets = 16
        counts = Counter()
        for _ in range(4000):
            h = ecmp_hash(
                f"10.{rng.randint(0,7)}.{rng.randint(0,3)}.{rng.randint(2,5)}",
                f"10.{rng.randint(0,7)}.{rng.randint(0,3)}.{rng.randint(2,5)}",
                rng.randint(1024, 65535),
                rng.randint(1, 1023),
                rng.choice([6, 17]),
            )
            counts[h % n_buckets] += 1
        # Every bucket should have been hit
        self.assertEqual(len(counts), n_buckets,
                         f"Only {len(counts)}/{n_buckets} buckets used")

    def test_known_hash_value_stability(self):
        """
        Regression: verify a specific hash value does not change across
        refactors (the exact value depends on zlib.crc32 which is stable).
        """
        h = ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
        # Re-compute expected value deterministically.
        import struct, zlib
        packed = struct.pack("!IIHHB", 0x0A000002, 0x0A010002, 12345, 80, 6)
        expected = zlib.crc32(packed) & 0xFFFF_FFFF
        self.assertEqual(h, expected)

    def test_zero_ports_valid(self):
        """Port 0 is technically valid; hash should not raise."""
        h = ecmp_hash("10.0.0.2", "10.1.0.2", 0, 0, 1)
        self.assertIsInstance(h, int)


# =============================================================================
# 3. BaseScheduler contract tests (exercised via ECMPScheduler)
# =============================================================================
class TestBaseSchedulerInterface(unittest.TestCase):
    """Verify BaseScheduler contract: init, repr, metrics, delegation."""

    def setUp(self):
        self.topo = _make_topo(k=4)
        self.sched = ECMPScheduler(self.topo)

    def test_name_property(self):
        self.assertEqual(self.sched.name, "ecmp")

    def test_repr_contains_class_name(self):
        r = repr(self.sched)
        self.assertIn("ECMPScheduler", r)
        self.assertIn("k=4", r)

    def test_wrong_topology_type_raises(self):
        with self.assertRaises(TypeError):
            ECMPScheduler("not_a_topology")  # type: ignore

    def test_metrics_initially_zero(self):
        m = self.sched.metrics
        self.assertEqual(m.flows_scheduled, 0)
        self.assertEqual(m.flows_failed, 0)
        self.assertEqual(m.total_bytes_scheduled, 0)

    def test_reset_metrics(self):
        flow = _make_flow()
        self.sched.schedule_flow(flow)
        self.sched.reset_metrics()
        self.assertEqual(self.sched.metrics.flows_scheduled, 0)

    def test_get_candidate_paths_valid(self):
        paths = self.sched.get_candidate_paths("10.0.0.2", "10.1.0.2")
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_get_candidate_paths_unknown_ip(self):
        paths = self.sched.get_candidate_paths("192.168.1.1", "10.1.0.2")
        self.assertEqual(paths, [])

    def test_report_string(self):
        r = self.sched.report()
        self.assertIn("ECMP", r)
        self.assertIn("Flows scheduled", r)


# =============================================================================
# 4. ECMPScheduler unit tests
# =============================================================================
class TestEcmpSchedulerUnit(unittest.TestCase):
    """Unit tests for ECMPScheduler path selection, caching, and stats."""

    def setUp(self):
        self.topo = _make_topo(k=4)
        self.sched = ECMPScheduler(self.topo)

    # ── Path validity ─────────────────────────────────────────────────────────

    def test_schedule_flow_returns_list(self):
        flow = _make_flow()
        path = self.sched.schedule_flow(flow)
        self.assertIsInstance(path, list)

    def test_path_starts_at_src_host(self):
        flow = _make_flow(src_ip="10.0.0.2", dst_ip="10.1.0.2")
        path = self.sched.schedule_flow(flow)
        self.assertEqual(path[0], "h_0_0_0")

    def test_path_ends_at_dst_host(self):
        flow = _make_flow(src_ip="10.0.0.2", dst_ip="10.1.0.2")
        path = self.sched.schedule_flow(flow)
        self.assertEqual(path[-1], "h_1_0_0")

    def test_path_length_cross_pod(self):
        """Cross-pod paths in k=4 have 7 nodes (4 hops in fabric + 2 host hops = 6 links)."""
        # h_0_0_0 → e_0_0 → a_0_? → c_?_? → a_1_? → e_1_0 → h_1_0_0
        flow = _make_flow(src_ip="10.0.0.2", dst_ip="10.1.0.2")
        path = self.sched.schedule_flow(flow)
        self.assertEqual(len(path), 7, f"Expected 7-node path, got: {path}")

    def test_path_length_within_pod(self):
        """Within-pod paths in k=4 have 5 nodes (h → e → a → e → h)."""
        # h_0_0_0 (10.0.0.2) → h_0_1_0 (10.0.1.2) — same pod, different edge
        flow = _make_flow(src_ip="10.0.0.2", dst_ip="10.0.1.2")
        path = self.sched.schedule_flow(flow)
        self.assertEqual(len(path), 5, f"Expected 5-node path, got: {path}")

    def test_same_host_loopback(self):
        flow = _make_flow(src_ip="10.0.0.2", dst_ip="10.0.0.2")
        path = self.sched.schedule_flow(flow)
        self.assertEqual(path, ["h_0_0_0"])

    def test_unknown_src_ip_returns_none(self):
        flow = _make_flow(src_ip="192.168.1.1")
        path = self.sched.schedule_flow(flow)
        self.assertIsNone(path)

    def test_unknown_dst_ip_returns_none(self):
        flow = _make_flow(dst_ip="172.16.0.1")
        path = self.sched.schedule_flow(flow)
        self.assertIsNone(path)

    # ── Determinism ───────────────────────────────────────────────────────────

    def test_same_5tuple_always_same_path(self):
        """ECMP must be fully deterministic: identical 5-tuple → identical path."""
        flow = _make_flow()
        paths = [self.sched.schedule_flow(flow) for _ in range(20)]
        self.assertTrue(
            all(p == paths[0] for p in paths),
            "Different paths returned for identical 5-tuple",
        )

    def test_different_src_port_may_give_different_path(self):
        """Two flows differing only in src_port should hash to potentially different paths."""
        paths_seen = set()
        for port in range(1024, 2048, 16):
            flow = _make_flow(src_ip="10.0.0.2", dst_ip="10.2.0.2",
                              src_port=port, flow_id=f"f{port}")
            path = self.sched.schedule_flow(flow)
            if path:
                paths_seen.add(tuple(path))
        # With 64 different ports, we expect at least 2 distinct paths (k=4 → 4 cross-pod paths)
        self.assertGreater(len(paths_seen), 1,
                           "ECMP should use multiple paths across different src ports")

    # ── Path caching ──────────────────────────────────────────────────────────

    def test_cache_populated_after_first_schedule(self):
        self.assertEqual(self.sched.cache_size(), 0)
        self.sched.schedule_flow(_make_flow())
        self.assertGreater(self.sched.cache_size(), 0)

    def test_clear_cache(self):
        self.sched.schedule_flow(_make_flow())
        self.sched.clear_cache()
        self.assertEqual(self.sched.cache_size(), 0)

    def test_no_cache_mode(self):
        sched_nocache = ECMPScheduler(self.topo, cache_paths=False)
        sched_nocache.schedule_flow(_make_flow())
        self.assertEqual(sched_nocache.cache_size(), 0)

    # ── Statistics ────────────────────────────────────────────────────────────

    def test_metrics_updated_after_schedule(self):
        # Metrics are populated by schedule_flows() (the public batch API).
        self.sched.schedule_flows([_make_flow(size_bytes=5000)])
        self.assertEqual(self.sched.metrics.flows_scheduled, 1)
        self.assertEqual(self.sched.metrics.total_bytes_scheduled, 5000)

    def test_metrics_mice_count(self):
        self.sched.schedule_flows([_make_flow(size_bytes=50_000)])   # mice
        self.assertEqual(self.sched.metrics.mice_flows, 1)
        self.assertEqual(self.sched.metrics.elephant_flows, 0)

    def test_metrics_elephant_count(self):
        self.sched.schedule_flows([_make_flow(size_bytes=5_000_000)])  # elephant
        self.assertEqual(self.sched.metrics.elephant_flows, 1)
        self.assertEqual(self.sched.metrics.mice_flows, 0)

    def test_hash_distribution_populated(self):
        flows = [_make_flow(src_port=port, flow_id=f"f{port}")
                 for port in range(1024, 1200)]
        self.sched.schedule_flows(flows)
        dist = self.sched.hash_distribution()
        self.assertGreater(len(dist), 0)

    def test_ecmp_stats_string(self):
        self.sched.schedule_flow(_make_flow())
        s = self.sched.ecmp_stats()
        self.assertIn("ECMP", s)
        self.assertIn("cache", s.lower())


# =============================================================================
# 5. Integration tests — 100 flows, distribution analysis
# =============================================================================
class TestEcmpIntegration(unittest.TestCase):
    """
    Integration-level tests that exercise the full scheduling pipeline with
    realistic synthetic workloads.

    Flow mix follows approximate data-centre distributions:
      80% mice  (<100 KB)
      15% medium (100 KB – 1 MB)
       5% elephant (>1 MB)
    """

    # ── k=4 topology: 16 hosts, 4 cross-pod ECMP paths ───────────────────────

    @classmethod
    def setUpClass(cls):
        cls.topo4 = FatTreeGraph(k=4)
        cls.topo8 = FatTreeGraph(k=8)

    def _generate_flows(
        self, topo: FatTreeGraph, n: int, seed: int = 42
    ) -> List[Flow]:
        rng = random.Random(seed)
        flows = []
        for i in range(n):
            # Size distribution: 80% mice, 15% medium, 5% elephant
            r = rng.random()
            if r < 0.80:
                size = rng.randint(1, MICE_THRESHOLD_BYTES - 1)
            elif r < 0.95:
                size = rng.randint(MICE_THRESHOLD_BYTES, ELEPHANT_THRESHOLD_BYTES - 1)
            else:
                size = rng.randint(ELEPHANT_THRESHOLD_BYTES, 100_000_000)
            flows.append(_random_flow(topo, rng,
                                      flow_id=f"flow_{i:04d}",
                                      size_bytes=size))
        return flows

    # ── Test 1: all 100 flows get valid paths ─────────────────────────────────

    def test_all_100_flows_get_valid_paths_k4(self):
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 100)
        results = sched.schedule_flows(flows)

        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(n_ok, 100,
                         f"Only {n_ok}/100 flows got paths")

    def test_each_path_starts_and_ends_at_host_k4(self):
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 100)
        results = sched.schedule_flows(flows)
        flows_by_id = {f.flow_id: f for f in flows}

        for flow_id, path in results.items():
            self.assertIsNotNone(path, f"Flow {flow_id} has None path")
            f = flows_by_id[flow_id]
            expected_src = self.topo4.node_for_ip(f.src_ip)
            expected_dst = self.topo4.node_for_ip(f.dst_ip)
            self.assertEqual(path[0], expected_src,
                             f"Flow {flow_id}: path starts at {path[0]}, expected {expected_src}")
            self.assertEqual(path[-1], expected_dst,
                             f"Flow {flow_id}: path ends at {path[-1]}, expected {expected_dst}")

    def test_path_nodes_exist_in_topology_k4(self):
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 50)
        results = sched.schedule_flows(flows)

        all_nodes = set(self.topo4.graph.nodes())
        for flow_id, path in results.items():
            for node in path:
                self.assertIn(node, all_nodes,
                              f"Flow {flow_id}: node {node!r} not in topology")

    def test_consecutive_nodes_in_path_are_connected_k4(self):
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 30)
        results = sched.schedule_flows(flows)

        for flow_id, path in results.items():
            for u, v in zip(path[:-1], path[1:]):
                self.assertTrue(
                    self.topo4.graph.has_edge(u, v),
                    f"Flow {flow_id}: no edge {u}—{v} in topology",
                )

    # ── Test 2: multi-path coverage ───────────────────────────────────────────

    def test_cross_pod_flows_use_multiple_paths_k4(self):
        """
        For cross-pod flows in k=4, there are 4 ECMP paths.
        With 400 flows from the same (src,dst) pair but varying ports,
        at least 2 distinct paths should be selected.
        """
        sched = ECMPScheduler(self.topo4)
        # h_0_0_0 (pod 0) → h_2_0_0 (pod 2): cross-pod
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_2_0_0")

        paths_seen = set()
        for port in range(1024, 1424):
            flow = Flow.create(src_ip, dst_ip, port, 80)
            path = sched.schedule_flow(flow)
            if path:
                paths_seen.add(tuple(path))

        n_ecmp = len(self.topo4.get_paths("h_0_0_0", "h_2_0_0"))
        self.assertGreaterEqual(
            len(paths_seen), min(2, n_ecmp),
            f"Expected ≥2 distinct paths, got {len(paths_seen)} (n_ecmp={n_ecmp})",
        )

    def test_within_pod_flows_use_correct_number_of_paths_k4(self):
        """Within-pod flows in k=4 have 2 ECMP paths (through a_0_0 or a_0_1)."""
        sched = ECMPScheduler(self.topo4)
        src_ip = self.topo4.get_host_ip("h_0_0_0")
        dst_ip = self.topo4.get_host_ip("h_0_1_0")

        paths_seen = set()
        for port in range(1024, 2024):
            flow = Flow.create(src_ip, dst_ip, port, 80)
            path = sched.schedule_flow(flow)
            if path:
                paths_seen.add(tuple(path))

        n_ecmp = len(self.topo4.get_paths("h_0_0_0", "h_0_1_0"))
        self.assertEqual(n_ecmp, 2, "k=4 within-pod should have 2 ECMP paths")
        self.assertGreaterEqual(len(paths_seen), 2,
                                "Should use both within-pod paths")

    # ── Test 3: load distribution quality ────────────────────────────────────

    def test_hash_distribution_reasonable_k4(self):
        """
        With 400 flows spread across the full k=4 host set, path-index
        distribution should not be completely lopsided (no bucket > 60%).
        """
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 400, seed=123)
        sched.schedule_flows(flows)

        dist = sched.hash_distribution()
        total = sum(dist.values())
        if total > 0:
            max_fraction = max(dist.values()) / total
            self.assertLess(
                max_fraction, 0.60,
                f"One path bucket got {max_fraction*100:.1f}% of flows — too imbalanced",
            )

    # ── Test 4: metrics accuracy ──────────────────────────────────────────────

    def test_metrics_counts_match_results_k4(self):
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 100)
        results = sched.schedule_flows(flows)

        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(sched.metrics.flows_scheduled, n_ok)
        self.assertEqual(
            sched.metrics.mice_flows
            + sched.metrics.medium_flows
            + sched.metrics.elephant_flows,
            n_ok,
        )

    # ── Test 5: k=8 smoke test (128 hosts) ───────────────────────────────────

    def test_100_flows_k8(self):
        """Smoke test: 100 flows on k=8 (128-host) topology, all should succeed."""
        sched = ECMPScheduler(self.topo8)
        flows = self._generate_flows(self.topo8, 100, seed=7)
        results = sched.schedule_flows(flows)

        n_ok = sum(1 for p in results.values() if p is not None)
        self.assertEqual(n_ok, 100, f"Only {n_ok}/100 flows got paths on k=8")

    def test_cross_pod_paths_have_16_ecmp_options_k8(self):
        """k=8 cross-pod host pair should expose 16 ECMP paths."""
        src_ip = self.topo8.get_host_ip("h_0_0_0")
        dst_ip = self.topo8.get_host_ip("h_4_0_0")
        n_ecmp = len(self.topo8.get_paths("h_0_0_0", "h_4_0_0"))
        self.assertEqual(n_ecmp, 16,
                         f"Expected 16 cross-pod ECMP paths on k=8, got {n_ecmp}")

    def test_schedule_flows_populates_assigned_path(self):
        """schedule_flows must set flow.assigned_path after scheduling."""
        sched = ECMPScheduler(self.topo4)
        flows = self._generate_flows(self.topo4, 20)
        sched.schedule_flows(flows)
        for f in flows:
            self.assertIsNotNone(
                f.assigned_path,
                f"Flow {f.flow_id}: assigned_path not set after schedule_flows",
            )


# =============================================================================
# Entry point (run directly or via pytest)
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LAFS Scheduler Tests")
    parser.add_argument("--verbose", "-v", action="store_true")
    args, remaining = parser.parse_known_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestFlowDataclass,
        TestEcmpHashFunction,
        TestBaseSchedulerInterface,
        TestEcmpSchedulerUnit,
        TestEcmpIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
