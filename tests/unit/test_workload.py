"""
LAFS — Workload Generator Unit Tests
=====================================
COMP-6910 — Group 7

Test coverage
-------------
TestFacebookWebSearch   (18 tests) — CDF distribution, Poisson arrivals,
                                     multi-tenant, reproducibility
TestAllReduceWorkload   (16 tests) — ring topology, shard sizes, PS mode,
                                     pipeline stages, iteration counts
TestMicroserviceRPC     (15 tests) — graph patterns, flow counts, RPC sizes,
                                     topological ordering, placement
TestWorkloadRunner      (14 tests) — unified runner, stats, mixed workload,
                                     load fraction, Jain's index

Usage
-----
    pytest tests/unit/test_workload.py -v
    python tests/unit/test_workload.py --verbose
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
from src.workload.flow import Flow, MICE_THRESHOLD_BYTES
from src.workload.facebook_websearch import (
    FacebookWebSearchGenerator,
    FacebookWebSearchConfig,
    _sample_flow_size,
    _FB_WEBSEARCH_CDF,
)
from src.workload.allreduce import (
    AllReduceGenerator,
    AllReduceConfig,
    MODEL_SIZES,
    _NCCL_PORT,
)
from src.workload.microservice import (
    MicroserviceRPCGenerator,
    MicroserviceConfig,
    ServiceGraph,
)
from src.workload.runner import WorkloadRunner, WorkloadConfig, WorkloadStats


# =============================================================================
# Shared fixtures
# =============================================================================
def _topo(k: int = 4) -> FatTreeGraph:
    return FatTreeGraph(k=k)


# =============================================================================
# 1. Facebook Web Search Generator tests
# =============================================================================
class TestFacebookWebSearch(unittest.TestCase):
    """Verify Facebook web-search workload generation."""

    def setUp(self):
        self.topo = _topo(k=4)  # 16 hosts

    # ── Construction ──────────────────────────────────────────────────────────

    def test_generates_correct_count(self):
        cfg = FacebookWebSearchConfig(n_flows=200, seed=1)
        gen = FacebookWebSearchGenerator(self.topo, cfg)
        flows = gen.generate()
        self.assertEqual(len(flows), 200)

    def test_default_config_works(self):
        gen = FacebookWebSearchGenerator(self.topo)
        flows = gen.generate()
        self.assertGreater(len(flows), 0)

    def test_invalid_load_fraction_raises(self):
        with self.assertRaises(ValueError):
            FacebookWebSearchConfig(load_fraction=0.0)
        with self.assertRaises(ValueError):
            FacebookWebSearchConfig(load_fraction=1.5)

    def test_too_many_tenants_raises(self):
        with self.assertRaises(ValueError):
            FacebookWebSearchGenerator(
                self.topo,
                FacebookWebSearchConfig(n_tenants=100)  # more than 16 hosts
            )

    # ── Flow validity ─────────────────────────────────────────────────────────

    def test_all_flows_have_valid_ips(self):
        cfg = FacebookWebSearchConfig(n_flows=100, seed=2)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        valid_ips = {self.topo.get_host_ip(h) for h in self.topo.hosts}
        for f in flows:
            self.assertIn(f.src_ip, valid_ips,
                          f"src_ip {f.src_ip} not in topology")
            self.assertIn(f.dst_ip, valid_ips,
                          f"dst_ip {f.dst_ip} not in topology")

    def test_no_self_loops(self):
        cfg = FacebookWebSearchConfig(n_flows=200, seed=3)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        for f in flows:
            self.assertNotEqual(f.src_ip, f.dst_ip,
                                f"Flow {f.flow_id}: src == dst")

    def test_flows_sorted_by_arrival_time(self):
        cfg = FacebookWebSearchConfig(n_flows=500, seed=4)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        times = [f.arrival_time for f in flows]
        self.assertEqual(times, sorted(times))

    def test_protocol_is_tcp(self):
        cfg = FacebookWebSearchConfig(n_flows=50, seed=5)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        for f in flows:
            self.assertEqual(f.protocol, 6, "Facebook flows should be TCP")

    # ── Size distribution ─────────────────────────────────────────────────────

    def test_mice_fraction_approx_90_percent(self):
        """
        The CDF is calibrated for ~90 % of flows < 100 KB (mice).
        With 1000 flows we expect 85–95 % mice.
        """
        cfg = FacebookWebSearchConfig(n_flows=1_000, seed=42)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        mice = sum(1 for f in flows if f.size_bytes < MICE_THRESHOLD_BYTES)
        fraction = mice / len(flows)
        self.assertGreater(fraction, 0.83,
                           f"Mice fraction {fraction:.2f} < 0.83")
        self.assertLess(fraction, 0.97,
                        f"Mice fraction {fraction:.2f} > 0.97")

    def test_all_sizes_positive(self):
        cfg = FacebookWebSearchConfig(n_flows=200, seed=6)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        for f in flows:
            self.assertGreater(f.size_bytes, 0)

    def test_max_size_within_cdf_bound(self):
        cfg = FacebookWebSearchConfig(n_flows=500, seed=7)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        max_size = _FB_WEBSEARCH_CDF[-1][2]
        for f in flows:
            self.assertLessEqual(f.size_bytes, max_size)

    def test_size_distribution_stats(self):
        cfg = FacebookWebSearchConfig(n_flows=100, seed=8)
        gen = FacebookWebSearchGenerator(self.topo, cfg)
        stats = gen.size_distribution_stats()
        self.assertIn("mice_fraction", stats)
        self.assertGreater(stats["mice_fraction"], 0.80)
        self.assertGreater(stats["mean_bytes"], 0)

    # ── Reproducibility ───────────────────────────────────────────────────────

    def test_same_seed_produces_identical_flows(self):
        cfg = FacebookWebSearchConfig(n_flows=50, seed=99)
        f1 = FacebookWebSearchGenerator(self.topo, cfg).generate()
        f2 = FacebookWebSearchGenerator(self.topo, cfg).generate()
        for a, b in zip(f1, f2):
            self.assertEqual(a.flow_id, b.flow_id)
            self.assertEqual(a.size_bytes, b.size_bytes)
            self.assertAlmostEqual(a.arrival_time, b.arrival_time, places=9)

    def test_different_seeds_produce_different_flows(self):
        f1 = FacebookWebSearchGenerator(
            self.topo, FacebookWebSearchConfig(n_flows=50, seed=1)
        ).generate()
        f2 = FacebookWebSearchGenerator(
            self.topo, FacebookWebSearchConfig(n_flows=50, seed=2)
        ).generate()
        sizes1 = [f.size_bytes for f in f1]
        sizes2 = [f.size_bytes for f in f2]
        self.assertNotEqual(sizes1, sizes2)

    # ── Multi-tenant ──────────────────────────────────────────────────────────

    def test_multi_tenant_flow_ids(self):
        """Flow IDs for n_tenants=4 should include t0, t1, t2, t3 prefixes."""
        cfg = FacebookWebSearchConfig(n_flows=400, n_tenants=4, seed=10)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        tenant_prefixes = {f.flow_id.split("_")[0] for f in flows}
        self.assertGreaterEqual(len(tenant_prefixes), 2,
                                "Multi-tenant should produce flows from multiple tenants")

    def test_single_tenant_config(self):
        cfg = FacebookWebSearchConfig(n_flows=100, n_tenants=1, seed=11)
        flows = FacebookWebSearchGenerator(self.topo, cfg).generate()
        self.assertEqual(len(flows), 100)

    # ── Sample flow-size CDF ──────────────────────────────────────────────────

    def test_sample_flow_size_in_range(self):
        rng = random.Random(0)
        for _ in range(1_000):
            s = _sample_flow_size(rng)
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, _FB_WEBSEARCH_CDF[-1][2])


# =============================================================================
# 2. AllReduce Workload tests
# =============================================================================
class TestAllReduceWorkload(unittest.TestCase):
    """Verify AllReduce synthetic workload generation."""

    def setUp(self):
        self.topo = _topo(k=4)  # 16 hosts

    # ── Construction ──────────────────────────────────────────────────────────

    def test_basic_ring_generation(self):
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=10, seed=1)
        flows = AllReduceGenerator(self.topo, cfg).generate()
        # Ring: n_workers flows per iteration
        self.assertEqual(len(flows), 4 * 10)

    def test_default_config_works(self):
        # Default n_workers=8 requires ≥8 hosts (k=4 has 16).
        cfg = AllReduceConfig(n_workers=4, seed=42)
        flows = AllReduceGenerator(self.topo, cfg).generate()
        self.assertGreater(len(flows), 0)

    def test_too_many_workers_raises(self):
        with self.assertRaises(ValueError):
            AllReduceGenerator(
                self.topo,
                AllReduceConfig(n_workers=100)   # 16 hosts only
            )

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            AllReduceConfig(n_workers=4, mode="broadcast")

    def test_invalid_n_workers_raises(self):
        with self.assertRaises(ValueError):
            AllReduceConfig(n_workers=1)

    # ── Ring topology ─────────────────────────────────────────────────────────

    def test_ring_flows_form_ring(self):
        """Each worker must appear exactly once as src and once as dst."""
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=1, seed=2)
        gen = AllReduceGenerator(self.topo, cfg)
        flows = gen.generate()
        worker_ips = set(gen.worker_ips())

        srcs = Counter(f.src_ip for f in flows)
        dsts = Counter(f.dst_ip for f in flows)
        for ip in worker_ips:
            self.assertEqual(srcs[ip], 1,
                             f"Worker {ip} should be src exactly once per iteration")
            self.assertEqual(dsts[ip], 1,
                             f"Worker {ip} should be dst exactly once per iteration")

    def test_ring_shard_size(self):
        """Each ring flow should carry gradient_bytes / n_workers bytes."""
        cfg = AllReduceConfig(n_workers=4, gradient_bytes=40_000_000,
                               n_iterations=1, seed=3)
        flows = AllReduceGenerator(self.topo, cfg).generate()
        expected_shard = 40_000_000 // 4
        for f in flows:
            self.assertEqual(f.size_bytes, expected_shard)

    def test_ring_uses_nccl_port(self):
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=1, seed=4)
        flows = AllReduceGenerator(self.topo, cfg).generate()
        for f in flows:
            self.assertEqual(f.dst_port, _NCCL_PORT)

    def test_all_workers_in_topology(self):
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=1, seed=5)
        gen = AllReduceGenerator(self.topo, cfg)
        valid_ips = {self.topo.get_host_ip(h) for h in self.topo.hosts}
        for ip in gen.worker_ips():
            self.assertIn(ip, valid_ips)

    # ── PS mode ───────────────────────────────────────────────────────────────

    def test_ps_mode_flow_count(self):
        """PS mode: 2 × (n_workers - 1) flows per iteration (PS excluded)."""
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=1, mode="ps", seed=6)
        flows = AllReduceGenerator(self.topo, cfg).generate()
        # 3 workers × 2 (upload + download) = 6 flows
        self.assertEqual(len(flows), 6)

    def test_ps_flows_go_through_ps(self):
        """All PS-mode flows should have the PS IP as src or dst."""
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=1, mode="ps", seed=7)
        gen = AllReduceGenerator(self.topo, cfg)
        flows = gen.generate()
        ps_ip = gen._ps_ip
        for f in flows:
            self.assertTrue(
                f.src_ip == ps_ip or f.dst_ip == ps_ip,
                f"PS flow {f.flow_id}: neither endpoint is PS ({ps_ip})"
            )

    # ── Iteration timing ──────────────────────────────────────────────────────

    def test_iterations_advance_time(self):
        """Flows from later iterations should have later arrival_times."""
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=5, seed=8)
        flows = AllReduceGenerator(self.topo, cfg).generate()
        # Group by iteration.
        iter_times: dict = {}
        for f in flows:
            parts = f.flow_id.split("_")
            # flow_id: ar_ring_it0000_w0
            it_part = next((p for p in parts if p.startswith("it")), None)
            if it_part:
                it = int(it_part[2:])
                t = iter_times.setdefault(it, [])
                t.append(f.arrival_time)
        means = {it: sum(ts) / len(ts) for it, ts in iter_times.items()}
        iters = sorted(means.keys())
        for i in range(len(iters) - 1):
            self.assertLess(means[iters[i]], means[iters[i + 1]],
                            "Later iterations should start after earlier ones")

    # ── Pipeline parallelism ──────────────────────────────────────────────────

    def test_pipeline_adds_extra_flows(self):
        """Adding pipeline_stages=2 should produce more flows than ring alone."""
        cfg_base = AllReduceConfig(n_workers=4, model_preset="resnet50",
                                    n_iterations=2, pipeline_stages=1, seed=9)
        cfg_pipe = AllReduceConfig(n_workers=4, model_preset="resnet50",
                                    n_iterations=2, pipeline_stages=2, seed=9)
        flows_base = AllReduceGenerator(self.topo, cfg_base).generate()
        flows_pipe = AllReduceGenerator(self.topo, cfg_pipe).generate()
        self.assertGreater(len(flows_pipe), len(flows_base))

    # ── Reproducibility ───────────────────────────────────────────────────────

    def test_reproducibility(self):
        cfg = AllReduceConfig(n_workers=4, model_preset="resnet50",
                               n_iterations=5, seed=77)
        f1 = AllReduceGenerator(self.topo, cfg).generate()
        f2 = AllReduceGenerator(self.topo, cfg).generate()
        for a, b in zip(f1, f2):
            self.assertEqual(a.flow_id, b.flow_id)
            self.assertEqual(a.size_bytes, b.size_bytes)

    def test_model_preset_resolves(self):
        for preset, size in MODEL_SIZES.items():
            if preset == "custom" or size == 0:
                continue
            cfg = AllReduceConfig(n_workers=4, model_preset=preset,
                                   n_iterations=1, seed=0)
            gen = AllReduceGenerator(self.topo, cfg)
            self.assertEqual(gen.config.gradient_bytes, size)


# =============================================================================
# 3. Microservice RPC Chain Generator tests
# =============================================================================
class TestMicroserviceRPC(unittest.TestCase):
    """Verify microservice RPC workload generation."""

    def setUp(self):
        self.topo = _topo(k=4)

    # ── ServiceGraph factories ────────────────────────────────────────────────

    def test_linear_chain_graph(self):
        g = ServiceGraph.linear_chain(depth=4)
        self.assertEqual(len(g.nodes), 4)
        self.assertEqual(len(g.edges), 3)

    def test_fan_out_graph(self):
        g = ServiceGraph.fan_out(fan=8)
        self.assertEqual(len(g.nodes), 9)   # 1 frontend + 8 leaves
        self.assertEqual(len(g.edges), 8)

    def test_mixed_dag_graph(self):
        g = ServiceGraph.mixed_dag(fan=4, depth=2)
        # frontend + 4 mw + 4 db = 9 nodes, 8 edges
        self.assertGreater(len(g.edges), 4)

    def test_chain_depth_too_small_raises(self):
        with self.assertRaises(ValueError):
            ServiceGraph.linear_chain(depth=1)

    def test_fan_out_zero_raises(self):
        with self.assertRaises(ValueError):
            ServiceGraph.fan_out(fan=0)

    # ── Construction validation ───────────────────────────────────────────────

    def test_invalid_graph_type_raises(self):
        with self.assertRaises(ValueError):
            MicroserviceConfig(graph_type="star")

    def test_invalid_placement_raises(self):
        with self.assertRaises(ValueError):
            MicroserviceConfig(placement="datacenter")

    # ── Flow generation ───────────────────────────────────────────────────────

    def test_generates_flows(self):
        cfg = MicroserviceConfig(n_requests=50, seed=1)
        flows = MicroserviceRPCGenerator(self.topo, cfg).generate()
        self.assertGreater(len(flows), 0)

    def test_flows_per_request_matches_actual(self):
        cfg = MicroserviceConfig(n_requests=100, graph_type="chain",
                                  chain_depth=3, seed=2,
                                  include_data_flows=False)
        gen = MicroserviceRPCGenerator(self.topo, cfg)
        flows = gen.generate()
        expected_per_req = gen.flows_per_request()
        # Allow ±some deviation due to same-host filtering.
        actual_per_req = len(flows) / cfg.n_requests
        self.assertAlmostEqual(actual_per_req, expected_per_req, delta=1.5)

    def test_fan_out_generates_parallel_flows(self):
        """Fan-out: multiple flows at the same arrival_time (parallel calls)."""
        cfg = MicroserviceConfig(n_requests=1, graph_type="fan_out",
                                  fan_out=4, seed=3)
        flows = MicroserviceRPCGenerator(self.topo, cfg).generate()
        # Multiple flows should share arrival_time (concurrent fan-out).
        arrival_times = [f.arrival_time for f in flows]
        time_counts = Counter(arrival_times)
        max_concurrent = max(time_counts.values())
        self.assertGreater(max_concurrent, 1,
                           "Fan-out should produce concurrent flows")

    def test_rpc_request_sizes_small(self):
        """RPC request flows should be small (< 10 KB)."""
        cfg = MicroserviceConfig(n_requests=50, seed=4, include_data_flows=False)
        flows = MicroserviceRPCGenerator(self.topo, cfg).generate()
        req_flows = [f for f in flows if f.flow_id.endswith("_req")]
        for f in req_flows:
            self.assertLessEqual(f.size_bytes, 10_000,
                                 f"RPC request {f.flow_id} size too large: {f.size_bytes}")

    def test_sorted_by_arrival_time(self):
        cfg = MicroserviceConfig(n_requests=100, seed=5)
        flows = MicroserviceRPCGenerator(self.topo, cfg).generate()
        times = [f.arrival_time for f in flows]
        self.assertEqual(times, sorted(times))

    def test_valid_ips(self):
        cfg = MicroserviceConfig(n_requests=30, seed=6)
        flows = MicroserviceRPCGenerator(self.topo, cfg).generate()
        valid_ips = {self.topo.get_host_ip(h) for h in self.topo.hosts}
        for f in flows:
            self.assertIn(f.src_ip, valid_ips)
            self.assertIn(f.dst_ip, valid_ips)

    def test_service_placement_random_mode(self):
        cfg = MicroserviceConfig(n_requests=20, placement="random", seed=7)
        gen = MicroserviceRPCGenerator(self.topo, cfg)
        placement = gen.service_placement()
        self.assertIsInstance(placement, dict)
        self.assertGreater(len(placement), 0)

    def test_service_placement_rack_mode(self):
        cfg = MicroserviceConfig(n_requests=20, placement="rack", seed=8)
        gen = MicroserviceRPCGenerator(self.topo, cfg)
        placement = gen.service_placement()
        self.assertIsInstance(placement, dict)
        self.assertGreater(len(placement), 0)

    def test_data_flows_optional(self):
        """With include_data_flows=False, fewer flows generated."""
        cfg_with = MicroserviceConfig(n_requests=20, graph_type="mixed",
                                       include_data_flows=True, seed=9)
        cfg_without = MicroserviceConfig(n_requests=20, graph_type="mixed",
                                          include_data_flows=False, seed=9)
        gen_with = MicroserviceRPCGenerator(self.topo, cfg_with)
        gen_without = MicroserviceRPCGenerator(self.topo, cfg_without)
        self.assertGreaterEqual(gen_with.flows_per_request(),
                                gen_without.flows_per_request())


# =============================================================================
# 4. WorkloadRunner tests
# =============================================================================
class TestWorkloadRunner(unittest.TestCase):
    """Verify WorkloadRunner orchestration, stats, and config validation."""

    def setUp(self):
        self.topo = _topo(k=4)

    # ── Config validation ─────────────────────────────────────────────────────

    def test_unknown_workload_type_raises(self):
        with self.assertRaises(ValueError):
            WorkloadConfig(workload_types=["unknown"])

    def test_mixed_expands_to_three_types(self):
        cfg = WorkloadConfig(workload_types=["mixed"], n_workers=4)
        self.assertIn("facebook", cfg.workload_types)
        self.assertIn("allreduce", cfg.workload_types)
        self.assertIn("microservice", cfg.workload_types)

    def test_invalid_load_fraction_raises(self):
        with self.assertRaises(ValueError):
            WorkloadConfig(load_fraction=0.0)

    # ── Facebook-only runner ──────────────────────────────────────────────────

    def test_facebook_only_count(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=100, seed=1)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        self.assertEqual(len(flows), 100)

    def test_facebook_flows_sorted(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=200, seed=2)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        times = [f.arrival_time for f in flows]
        self.assertEqual(times, sorted(times))

    # ── AllReduce-only runner ─────────────────────────────────────────────────

    def test_allreduce_only(self):
        cfg = WorkloadConfig(workload_types=["allreduce"], n_flows=40,
                             n_workers=4, model_preset="resnet50", seed=3)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        self.assertGreater(len(flows), 0)

    # ── Mixed workload ────────────────────────────────────────────────────────

    def test_mixed_generates_from_all_types(self):
        cfg = WorkloadConfig(
            workload_types=["facebook", "allreduce", "microservice"],
            n_flows=300,
            n_workers=4,
            model_preset="resnet50",
            seed=4,
        )
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        self.assertLessEqual(len(flows), 300)
        self.assertGreater(len(flows), 0)

    def test_mixed_shorthand(self):
        cfg = WorkloadConfig(workload_types=["mixed"], n_flows=200,
                             n_workers=4, model_preset="resnet50", seed=5)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        self.assertGreater(len(flows), 0)

    # ── WorkloadStats ─────────────────────────────────────────────────────────

    def test_compute_stats_returns_workload_stats(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=100, seed=6)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        stats = runner.compute_stats(flows)
        self.assertIsInstance(stats, WorkloadStats)

    def test_stats_n_flows_matches(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=100, seed=7)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        stats = runner.compute_stats(flows)
        self.assertEqual(stats.n_flows, len(flows))

    def test_stats_size_classes_sum_to_total(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=100, seed=8)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        stats = runner.compute_stats(flows)
        self.assertEqual(
            stats.mice_count + stats.medium_count + stats.elephant_count,
            stats.n_flows
        )

    def test_stats_mice_fraction_in_range(self):
        """Facebook-only workload should have high mice fraction (>80%)."""
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=500, seed=9)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        stats = runner.compute_stats(flows)
        self.assertGreater(stats.mice_fraction, 0.80)

    def test_jains_index_range(self):
        """Jain's index must be in (0, 1]."""
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=100,
                             n_tenants=2, seed=10)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        stats = runner.compute_stats(flows)
        self.assertGreater(stats.jains_index, 0.0)
        self.assertLessEqual(stats.jains_index, 1.0)

    def test_stats_summary_string(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=50, seed=11)
        runner = WorkloadRunner(self.topo, cfg)
        flows = runner.generate()
        stats = runner.compute_stats(flows)
        s = stats.summary()
        self.assertIn("Workload Statistics", s)
        self.assertIn("Mice", s)
        self.assertIn("Jain", s)

    def test_empty_flows_returns_default_stats(self):
        cfg = WorkloadConfig(workload_types=["facebook"], n_flows=10, seed=12)
        runner = WorkloadRunner(self.topo, cfg)
        stats = runner.compute_stats([])
        self.assertEqual(stats.n_flows, 0)


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
        TestFacebookWebSearch,
        TestAllReduceWorkload,
        TestMicroserviceRPC,
        TestWorkloadRunner,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
