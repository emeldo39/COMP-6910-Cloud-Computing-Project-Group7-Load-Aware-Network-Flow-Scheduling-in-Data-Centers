"""
LAFS Project — Unit Tests: Fat-Tree Topology
=============================================
COMP 6910 — Group 7

Tests FatTreeGraph (pure NetworkX, no Mininet) and all naming/IP helpers.
These tests run on ANY platform (Windows, Linux, macOS) without root.

Test categories:
  TestNamingHelpers       — node name and address functions
  TestFatTreeGraphK4      — k=4 fat-tree (16 hosts)
  TestFatTreeGraphK8      — k=8 fat-tree (128 hosts)
  TestFatTreeGraphPaths   — path computation between all host scenarios
  TestFatTreeGraphEdges   — link structure and port numbering
  TestFatTreeGraphTopology— graph theory properties (connectivity, diameter)
  TestFatTreeTopoStub     — FatTreeTopo graph delegation (no Mininet needed)

Usage:
    pytest tests/unit/test_topology.py -v
    pytest tests/unit/test_topology.py -v -k "k8"     # only k=8 tests
    pytest tests/unit/test_topology.py -v -k "paths"  # only path tests
"""

import sys
import os
import math
import unittest
from typing import List

import networkx as nx
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.topology.fattree import (
    FatTreeGraph,
    FatTreeTopo,
    agg_name,
    core_name,
    edge_name,
    host_ip,
    host_mac,
    host_name,
)


# =============================================================================
# TestNamingHelpers
# =============================================================================
class TestNamingHelpers(unittest.TestCase):
    """Verify the naming and addressing helper functions."""

    # ── core_name ─────────────────────────────────────────────────────────────
    def test_core_name_format(self):
        self.assertEqual(core_name(0, 0), "c_0_0")
        self.assertEqual(core_name(1, 2), "c_1_2")
        self.assertEqual(core_name(3, 3), "c_3_3")

    def test_core_name_type(self):
        self.assertIsInstance(core_name(0, 0), str)

    # ── agg_name ──────────────────────────────────────────────────────────────
    def test_agg_name_format(self):
        self.assertEqual(agg_name(0, 0), "a_0_0")
        self.assertEqual(agg_name(7, 3), "a_7_3")

    # ── edge_name ─────────────────────────────────────────────────────────────
    def test_edge_name_format(self):
        self.assertEqual(edge_name(0, 0), "e_0_0")
        self.assertEqual(edge_name(7, 3), "e_7_3")

    # ── host_name ─────────────────────────────────────────────────────────────
    def test_host_name_format(self):
        self.assertEqual(host_name(0, 0, 0), "h_0_0_0")
        self.assertEqual(host_name(7, 3, 3), "h_7_3_3")

    # ── host_ip ───────────────────────────────────────────────────────────────
    def test_host_ip_first(self):
        """First host in pod 0 should be 10.0.0.2."""
        self.assertEqual(host_ip(0, 0, 0), "10.0.0.2")

    def test_host_ip_offset(self):
        """idx=1 → .3, idx=2 → .4, etc."""
        self.assertEqual(host_ip(0, 0, 1), "10.0.0.3")
        self.assertEqual(host_ip(0, 0, 2), "10.0.0.4")

    def test_host_ip_different_pod_edge(self):
        self.assertEqual(host_ip(3, 2, 1), "10.3.2.3")

    def test_host_ip_last_k8(self):
        """k=8: last host h_7_3_3 → 10.7.3.5."""
        self.assertEqual(host_ip(7, 3, 3), "10.7.3.5")

    # ── host_mac ──────────────────────────────────────────────────────────────
    def test_host_mac_format(self):
        mac = host_mac(0, 0, 0)
        parts = mac.split(":")
        self.assertEqual(len(parts), 6, "MAC must have 6 octets")

    def test_host_mac_first(self):
        """h_0_0_0 → 00:00:0a:00:00:02."""
        self.assertEqual(host_mac(0, 0, 0), "00:00:0a:00:00:02")

    def test_host_mac_matches_ip(self):
        """MAC octet values should mirror the IP address digits."""
        for pod in range(4):
            for edge in range(2):
                for idx in range(2):
                    ip  = host_ip(pod, edge, idx)
                    mac = host_mac(pod, edge, idx)
                    # Last three octets of MAC = pod, edge, idx+2 in hex
                    ip_parts  = [int(x) for x in ip.split(".")]
                    mac_parts = [int(x, 16) for x in mac.split(":")]
                    self.assertEqual(
                        mac_parts[3], ip_parts[1],
                        f"MAC pod octet mismatch for h_{pod}_{edge}_{idx}"
                    )
                    self.assertEqual(
                        mac_parts[4], ip_parts[2],
                        f"MAC edge octet mismatch for h_{pod}_{edge}_{idx}"
                    )
                    self.assertEqual(
                        mac_parts[5], ip_parts[3],
                        f"MAC host octet mismatch for h_{pod}_{edge}_{idx}"
                    )

    # ── Validation ────────────────────────────────────────────────────────────
    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            FatTreeGraph(k=3)   # odd

    def test_invalid_k_zero_raises(self):
        with self.assertRaises(ValueError):
            FatTreeGraph(k=0)

    def test_invalid_k_negative_raises(self):
        with self.assertRaises(ValueError):
            FatTreeGraph(k=-2)


# =============================================================================
# TestFatTreeGraphK4
# =============================================================================
class TestFatTreeGraphK4(unittest.TestCase):
    """Structural tests for k=4 Fat-Tree (16 hosts, 20 switches)."""

    @classmethod
    def setUpClass(cls):
        cls.g = FatTreeGraph(k=4)

    # ── Counts ────────────────────────────────────────────────────────────────
    def test_host_count(self):
        """k=4: k³/4 = 16 hosts."""
        self.assertEqual(len(self.g.hosts), 16)
        self.assertEqual(self.g.n_hosts, 16)

    def test_core_switch_count(self):
        """k=4: (k/2)² = 4 core switches."""
        self.assertEqual(len(self.g.core_switches), 4)
        self.assertEqual(self.g.n_core, 4)

    def test_agg_switch_count(self):
        """k=4: k*(k/2) = 8 aggregation switches."""
        self.assertEqual(len(self.g.agg_switches), 8)
        self.assertEqual(self.g.n_agg, 8)

    def test_edge_switch_count(self):
        """k=4: k*(k/2) = 8 edge switches."""
        self.assertEqual(len(self.g.edge_switches), 8)
        self.assertEqual(self.g.n_edge, 8)

    def test_total_switch_count(self):
        """k=4: 4 + 8 + 8 = 20 switches total."""
        self.assertEqual(self.g.n_switches, 20)
        self.assertEqual(len(self.g.all_switches), 20)

    def test_total_node_count(self):
        """k=4: 20 switches + 16 hosts = 36 nodes."""
        self.assertEqual(self.g.graph.number_of_nodes(), 36)

    def test_link_count(self):
        """k=4: core-agg = (k/2)²*k=16, agg-edge = (k/2)²*k=16, edge-host = k*(k/2)*(k/2)=16."""
        half_k = 2
        k      = 4
        expected = (
            half_k ** 2 * k           # core-agg: 4 * 4 = 16
            + k * half_k * half_k     # agg-edge: 4 * 2 * 2 = 16
            + k * half_k * half_k     # edge-host: 4 * 2 * 2 = 16
        )
        self.assertEqual(self.g.graph.number_of_edges(), expected)

    # ── Node names ────────────────────────────────────────────────────────────
    def test_core_node_names(self):
        """Core switches for k=4: c_0_0, c_0_1, c_1_0, c_1_1."""
        expected = {"c_0_0", "c_0_1", "c_1_0", "c_1_1"}
        self.assertEqual(set(self.g.core_switches), expected)

    def test_first_host_name(self):
        self.assertIn("h_0_0_0", self.g.hosts)

    def test_last_host_name_k4(self):
        """Last host for k=4 is h_3_1_1."""
        self.assertIn("h_3_1_1", self.g.hosts)

    # ── IPs ───────────────────────────────────────────────────────────────────
    def test_all_ips_unique(self):
        """Every host must have a unique IP address."""
        ips = [self.g.get_host_ip(h) for h in self.g.hosts]
        self.assertEqual(len(ips), len(set(ips)), "Duplicate IP addresses detected")

    def test_ip_lookup(self):
        self.assertEqual(self.g.get_host_ip("h_0_0_0"), "10.0.0.2")

    def test_reverse_ip_lookup(self):
        self.assertEqual(self.g.node_for_ip("10.0.0.2"), "h_0_0_0")

    def test_unknown_ip_raises(self):
        with self.assertRaises(KeyError):
            self.g.node_for_ip("192.168.1.1")

    # ── MACs ──────────────────────────────────────────────────────────────────
    def test_all_macs_unique(self):
        macs = [self.g.get_host_mac(h) for h in self.g.hosts]
        self.assertEqual(len(macs), len(set(macs)), "Duplicate MACs detected")

    # ── repr / summary ────────────────────────────────────────────────────────
    def test_repr(self):
        r = repr(self.g)
        self.assertIn("k=4", r)
        self.assertIn("20", r)   # switches
        self.assertIn("16", r)   # hosts

    def test_summary_contains_counts(self):
        s = self.g.summary()
        self.assertIn("16", s)   # n_hosts
        self.assertIn("20", s)   # n_switches


# =============================================================================
# TestFatTreeGraphK8
# =============================================================================
class TestFatTreeGraphK8(unittest.TestCase):
    """Structural tests for k=8 Fat-Tree (128 hosts, 80 switches)."""

    @classmethod
    def setUpClass(cls):
        cls.g = FatTreeGraph(k=8)

    def test_host_count(self):
        """k=8: k³/4 = 128 hosts."""
        self.assertEqual(self.g.n_hosts, 128)
        self.assertEqual(len(self.g.hosts), 128)

    def test_core_switch_count(self):
        """k=8: (k/2)² = 16 core switches."""
        self.assertEqual(self.g.n_core, 16)

    def test_agg_switch_count(self):
        """k=8: k*(k/2) = 32 aggregation switches."""
        self.assertEqual(self.g.n_agg, 32)

    def test_edge_switch_count(self):
        """k=8: k*(k/2) = 32 edge switches."""
        self.assertEqual(self.g.n_edge, 32)

    def test_total_switch_count(self):
        """k=8: 16 + 32 + 32 = 80 switches."""
        self.assertEqual(self.g.n_switches, 80)

    def test_total_node_count(self):
        """k=8: 80 switches + 128 hosts = 208 nodes."""
        self.assertEqual(self.g.graph.number_of_nodes(), 208)

    def test_link_count(self):
        """k=8: 16*8 + 8*4*4 + 8*4*4 = 128+128+128 = 384 links."""
        half_k = 4
        k      = 8
        expected = (
            half_k ** 2 * k       # core-agg: 16 * 8 = 128
            + k * half_k * half_k  # agg-edge: 8 * 4 * 4 = 128
            + k * half_k * half_k  # edge-host: 8 * 4 * 4 = 128
        )
        self.assertEqual(self.g.graph.number_of_edges(), expected)

    def test_all_core_switches_exist(self):
        """All 16 core switch names must be present."""
        for row in range(4):
            for col in range(4):
                self.assertIn(core_name(row, col), self.g.core_switches)

    def test_all_agg_switches_exist(self):
        for pod in range(8):
            for idx in range(4):
                self.assertIn(agg_name(pod, idx), self.g.agg_switches)

    def test_all_edge_switches_exist(self):
        for pod in range(8):
            for idx in range(4):
                self.assertIn(edge_name(pod, idx), self.g.edge_switches)

    def test_all_hosts_exist(self):
        for pod in range(8):
            for e in range(4):
                for h in range(4):
                    self.assertIn(host_name(pod, e, h), self.g.hosts)

    def test_all_ips_unique_k8(self):
        ips = [self.g.get_host_ip(h) for h in self.g.hosts]
        self.assertEqual(len(ips), len(set(ips)), "k=8: Duplicate IPs")

    def test_all_macs_unique_k8(self):
        macs = [self.g.get_host_mac(h) for h in self.g.hosts]
        self.assertEqual(len(macs), len(set(macs)), "k=8: Duplicate MACs")

    def test_last_host_ip(self):
        """h_7_3_3 should have IP 10.7.3.5."""
        self.assertEqual(self.g.get_host_ip("h_7_3_3"), "10.7.3.5")


# =============================================================================
# TestFatTreeGraphPaths
# =============================================================================
class TestFatTreeGraphPaths(unittest.TestCase):
    """Path computation tests for various host pair scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.g4 = FatTreeGraph(k=4)
        cls.g8 = FatTreeGraph(k=8)

    # ── Self-path ─────────────────────────────────────────────────────────────
    def test_same_host_path(self):
        """Path from a host to itself is [[host]]."""
        paths = self.g4.get_paths("h_0_0_0", "h_0_0_0")
        self.assertEqual(paths, [["h_0_0_0"]])

    # ── Same edge switch (k=4: 1 path through the switch) ────────────────────
    def test_same_edge_switch_hosts(self):
        """Two hosts on the same edge switch share that switch on all paths."""
        paths = self.g4.get_paths("h_0_0_0", "h_0_0_1")
        self.assertGreater(len(paths), 0, "Expected at least 1 path")
        for path in paths:
            # Path must pass through the shared edge switch e_0_0
            self.assertIn("e_0_0", path,
                          f"Path {path} does not traverse e_0_0")

    # ── Same pod, different edge switches ─────────────────────────────────────
    def test_within_pod_paths_k4(self):
        """
        k=4: hosts in same pod but different edge switches.
        Expected shortest path: host → edge → agg → edge → host (length 4).
        Number of paths = k/2 = 2 (one through each agg switch).
        """
        paths = self.g4.get_paths("h_0_0_0", "h_0_1_0")
        self.assertEqual(len(paths), 2,
                         f"Expected 2 within-pod paths, got {len(paths)}: {paths}")
        for path in paths:
            self.assertEqual(len(path), 5,
                             f"Within-pod path length should be 5, got {path}")

    def test_within_pod_paths_k8(self):
        """
        k=8: within-pod paths.
        Expected paths = k/2 = 4.
        """
        paths = self.g8.get_paths("h_0_0_0", "h_0_1_0")
        self.assertEqual(len(paths), 4,
                         f"Expected 4 within-pod paths, got {len(paths)}")

    # ── Cross-pod paths ───────────────────────────────────────────────────────
    def test_cross_pod_paths_k4(self):
        """
        k=4: cross-pod path length = 6 (host→edge→agg→core→agg→edge→host).
        Expected paths = (k/2)² = 4.
        """
        paths = self.g4.get_paths("h_0_0_0", "h_1_0_0")
        self.assertEqual(len(paths), 4,
                         f"Expected 4 cross-pod paths, got {len(paths)}")
        for path in paths:
            self.assertEqual(len(path), 7,
                             f"Cross-pod path should be 7 hops, got {path}")

    def test_cross_pod_paths_k8(self):
        """
        k=8: cross-pod paths.
        Expected paths = (k/2)² = 16.
        """
        paths = self.g8.get_paths("h_0_0_0", "h_1_0_0")
        self.assertEqual(len(paths), 16,
                         f"Expected 16 cross-pod paths, got {len(paths)}")
        for path in paths:
            self.assertEqual(len(path), 7,
                             f"Cross-pod path should be 7 hops, got {path}")

    # ── Path validity ─────────────────────────────────────────────────────────
    def test_paths_start_and_end_correctly(self):
        src, dst = "h_0_0_0", "h_1_0_0"
        for path in self.g4.get_paths(src, dst):
            self.assertEqual(path[0], src)
            self.assertEqual(path[-1], dst)

    def test_paths_are_simple(self):
        """No node should appear twice in any path."""
        for path in self.g4.get_paths("h_0_0_0", "h_3_1_1"):
            self.assertEqual(len(path), len(set(path)),
                             f"Path has repeated nodes: {path}")

    def test_paths_use_valid_nodes(self):
        """Every node in every path must exist in the graph."""
        valid_nodes = set(self.g4.graph.nodes())
        for path in self.g4.get_paths("h_0_0_0", "h_1_0_0"):
            for node in path:
                self.assertIn(node, valid_nodes,
                              f"Unknown node '{node}' in path")

    def test_all_paths_are_connected(self):
        """Consecutive nodes in a path must share an edge."""
        for path in self.g4.get_paths("h_0_0_0", "h_2_1_0"):
            for i in range(len(path) - 1):
                self.assertTrue(
                    self.g4.graph.has_edge(path[i], path[i + 1]),
                    f"No edge between {path[i]} and {path[i+1]} in path {path}"
                )

    # ── IP-addressed path lookup ──────────────────────────────────────────────
    def test_path_by_ip(self):
        """get_paths should accept IP strings as well as node names."""
        paths_by_name = self.g4.get_paths("h_0_0_0", "h_1_0_0")
        paths_by_ip   = self.g4.get_paths("10.0.0.2", "10.1.0.2")
        self.assertEqual(len(paths_by_name), len(paths_by_ip))

    def test_unknown_node_raises(self):
        with self.assertRaises(KeyError):
            self.g4.get_paths("h_99_0_0", "h_0_0_0")

    # ── max_paths parameter ───────────────────────────────────────────────────
    def test_max_paths_limit(self):
        paths = self.g8.get_paths("h_0_0_0", "h_1_0_0", max_paths=4)
        self.assertLessEqual(len(paths), 4)
        self.assertGreater(len(paths), 0)

    # ── ECMP alias ────────────────────────────────────────────────────────────
    def test_ecmp_paths_same_as_get_paths(self):
        p1 = self.g4.get_paths("h_0_0_0", "h_1_0_0")
        p2 = self.g4.get_ecmp_paths("h_0_0_0", "h_1_0_0")
        self.assertEqual(p1, p2)

    # ── k-shortest paths ──────────────────────────────────────────────────────
    def test_k_shortest_paths_count(self):
        paths = self.g4.get_k_shortest_paths("h_0_0_0", "h_1_0_0", k=4)
        self.assertGreater(len(paths), 0)
        self.assertLessEqual(len(paths), 4)

    def test_k_shortest_paths_sorted_by_length(self):
        paths = self.g4.get_k_shortest_paths("h_0_0_0", "h_1_0_0", k=8)
        lengths = [len(p) for p in paths]
        self.assertEqual(lengths, sorted(lengths),
                         "Paths should be sorted shortest-first")

    # ── get_all_paths ─────────────────────────────────────────────────────────
    def test_get_all_paths_k4(self):
        """All-pairs paths for k=4 (16 hosts → 120 ordered pairs)."""
        all_paths = self.g4.get_all_paths()
        expected_pairs = 16 * 15  # H*(H-1) ordered pairs
        self.assertEqual(len(all_paths), expected_pairs,
                         f"Expected {expected_pairs} pairs, got {len(all_paths)}")

    def test_get_all_paths_contains_key(self):
        all_paths = self.g4.get_all_paths()
        self.assertIn(("h_0_0_0", "h_1_0_0"), all_paths)
        self.assertIn(("h_1_0_0", "h_0_0_0"), all_paths)  # reverse also present

    def test_all_paths_reverse_are_reversed(self):
        """Forward and reverse paths should be each other's reversal."""
        all_paths = self.g4.get_all_paths()
        fwd = all_paths[("h_0_0_0", "h_1_0_0")]
        rev = all_paths[("h_1_0_0", "h_0_0_0")]
        for p_fwd, p_rev in zip(fwd, rev):
            self.assertEqual(p_fwd, list(reversed(p_rev)))


# =============================================================================
# TestFatTreeGraphEdges
# =============================================================================
class TestFatTreeGraphEdges(unittest.TestCase):
    """Test link structure, port numbering, and capacities."""

    @classmethod
    def setUpClass(cls):
        cls.g = FatTreeGraph(k=4)

    def test_edge_host_link_exists(self):
        """e_0_0 — h_0_0_0 must be an edge."""
        self.assertTrue(
            self.g.graph.has_edge("e_0_0", "h_0_0_0"),
            "Missing edge: e_0_0 — h_0_0_0"
        )

    def test_agg_edge_link_exists(self):
        """a_0_0 — e_0_0 must be an edge."""
        self.assertTrue(
            self.g.graph.has_edge("a_0_0", "e_0_0"),
            "Missing edge: a_0_0 — e_0_0"
        )

    def test_core_agg_link_exists(self):
        """c_0_0 — a_0_0 must be an edge."""
        self.assertTrue(
            self.g.graph.has_edge("c_0_0", "a_0_0"),
            "Missing edge: c_0_0 — a_0_0"
        )

    def test_no_core_edge_link(self):
        """Core switches do NOT connect directly to edge switches."""
        self.assertFalse(
            self.g.graph.has_edge("c_0_0", "e_0_0"),
            "Core should not connect directly to edge switch"
        )

    def test_no_host_to_host_link(self):
        """Hosts are never directly linked to each other."""
        for h1 in self.g.hosts:
            for h2 in self.g.hosts:
                if h1 != h2:
                    self.assertFalse(
                        self.g.graph.has_edge(h1, h2),
                        f"Unexpected host–host link: {h1} — {h2}"
                    )

    def test_edge_host_port_numbering(self):
        """Port on edge side of host link should be h_idx + 1."""
        data = self.g.graph.get_edge_data("e_0_0", "h_0_0_0")
        self.assertIsNotNone(data)
        self.assertEqual(data["port_edge"], 1)   # h_idx=0 → port 1
        self.assertEqual(data["port_host"], 1)

        data2 = self.g.graph.get_edge_data("e_0_0", "h_0_0_1")
        self.assertIsNotNone(data2)
        self.assertEqual(data2["port_edge"], 2)  # h_idx=1 → port 2

    def test_agg_edge_port_numbering(self):
        """
        Agg port to edge e_idx: port = e_idx + 1.
        Edge port to agg a_idx: port = half_k + a_idx + 1.
        """
        # a_0_0 → e_0_0: agg port 1, edge port half_k+1 = 3
        data = self.g.graph.get_edge_data("a_0_0", "e_0_0")
        self.assertIsNotNone(data)
        self.assertEqual(data["port_agg"],  1)   # e_idx=0 → 1
        self.assertEqual(data["port_edge"], 3)   # half_k + a_idx + 1 = 2+0+1 = 3

    def test_core_agg_port_numbering(self):
        """
        Core port to agg in pod `pod`: port = pod + 1.
        Agg port to core c_i_j: port = half_k + j + 1.
        """
        # c_0_0 → a_0_0: core port = pod+1 = 1, agg port = half_k+col+1 = 2+0+1 = 3
        data = self.g.graph.get_edge_data("c_0_0", "a_0_0")
        self.assertIsNotNone(data)
        self.assertEqual(data["port_core"], 1)   # pod=0 → 1
        self.assertEqual(data["port_agg"],  3)   # half_k + col + 1 = 2+0+1 = 3

    def test_link_capacities(self):
        """Host-edge links: 1 Gbps. Fabric links: 10 Gbps."""
        from src.topology.fattree import BW_HOST_GBPS, BW_FABRIC_GBPS

        # Host link
        cap_host = self.g.get_link_capacity("e_0_0", "h_0_0_0")
        self.assertAlmostEqual(cap_host, BW_HOST_GBPS)

        # Fabric link
        cap_fab = self.g.get_link_capacity("a_0_0", "e_0_0")
        self.assertAlmostEqual(cap_fab, BW_FABRIC_GBPS)

    def test_link_type_attributes(self):
        """Link-type attributes must be set correctly."""
        self.assertEqual(
            self.g.graph["c_0_0"]["a_0_0"]["link_type"], "core-agg"
        )
        self.assertEqual(
            self.g.graph["a_0_0"]["e_0_0"]["link_type"], "agg-edge"
        )
        self.assertEqual(
            self.g.graph["e_0_0"]["h_0_0_0"]["link_type"], "edge-host"
        )

    def test_switch_link_filter(self):
        """get_switch_links() must exclude all edge-host links."""
        switch_links = self.g.get_switch_links()
        for u, v, d in switch_links:
            self.assertNotEqual(
                d.get("link_type"), "edge-host",
                f"edge-host link found in switch_links: {u}—{v}"
            )

    def test_all_links_have_capacity(self):
        for u, v, d in self.g.get_all_links():
            self.assertIn("capacity", d, f"Missing capacity on {u}—{v}")
            self.assertGreater(d["capacity"], 0)


# =============================================================================
# TestFatTreeGraphTopology (graph theory)
# =============================================================================
class TestFatTreeGraphTopology(unittest.TestCase):
    """Graph-theory properties: connectivity, diameter, degree."""

    @classmethod
    def setUpClass(cls):
        cls.g4 = FatTreeGraph(k=4)
        cls.g8 = FatTreeGraph(k=8)

    def test_fully_connected_k4(self):
        """The k=4 graph must be fully connected (single component)."""
        self.assertTrue(
            nx.is_connected(self.g4.graph),
            "k=4 Fat-tree graph is not connected!"
        )

    def test_fully_connected_k8(self):
        """The k=8 graph must be fully connected."""
        self.assertTrue(
            nx.is_connected(self.g8.graph),
            "k=8 Fat-tree graph is not connected!"
        )

    def test_diameter_k4(self):
        """
        k=4 diameter: longest shortest path.
        Cross-pod worst case: host→edge→agg→core→agg→edge→host = 6 hops.
        """
        # Use host subgraph to avoid counting switch-internal diameter
        # Diameter in the full graph = 6 (6 edges = 7 nodes in path)
        diam = nx.diameter(self.g4.graph)
        self.assertEqual(diam, 6,
                         f"k=4 Fat-tree diameter should be 6, got {diam}")

    def test_diameter_k8(self):
        """k=8: same diameter = 6 (cross-pod path is always 6 hops)."""
        diam = nx.diameter(self.g8.graph)
        self.assertEqual(diam, 6,
                         f"k=8 Fat-tree diameter should be 6, got {diam}")

    def test_host_degree_is_1(self):
        """Every host connects to exactly one edge switch."""
        for h in self.g4.hosts:
            deg = self.g4.graph.degree(h)
            self.assertEqual(deg, 1,
                             f"Host {h} has degree {deg}, expected 1")

    def test_edge_switch_degree_k4(self):
        """k=4 edge switch: k/2 downlinks (hosts) + k/2 uplinks (agg) = k."""
        for e in self.g4.edge_switches:
            self.assertEqual(self.g4.graph.degree(e), 4)

    def test_agg_switch_degree_k4(self):
        """k=4 agg switch: k/2 downlinks (edge) + k/2 uplinks (core) = k."""
        for a in self.g4.agg_switches:
            self.assertEqual(self.g4.graph.degree(a), 4)

    def test_core_switch_degree_k4(self):
        """k=4 core switch: k uplinks (one per pod) = k."""
        for c in self.g4.core_switches:
            self.assertEqual(self.g4.graph.degree(c), 4)

    def test_pod_membership(self):
        """same_pod() should be correct for known hosts."""
        # h_0_0_0 and h_0_1_0 are in pod 0 → same_pod = True
        self.assertTrue(self.g4.same_pod("h_0_0_0", "h_0_1_0"))
        # h_0_0_0 and h_1_0_0 are in different pods → False
        self.assertFalse(self.g4.same_pod("h_0_0_0", "h_1_0_0"))

    def test_get_pod_of_host(self):
        self.assertEqual(self.g4.get_pod_of_host("h_0_0_0"), 0)
        self.assertEqual(self.g4.get_pod_of_host("h_3_1_1"), 3)


# =============================================================================
# TestFatTreeTopoStub  (FatTreeTopo graph delegation, no Mininet)
# =============================================================================
class TestFatTreeTopoStub(unittest.TestCase):
    """Test FatTreeTopo via graph delegation (Mininet may or may not be present)."""

    @classmethod
    def setUpClass(cls):
        # FatTreeTopo can be constructed without Mininet (uses stub Topo)
        cls.topo = FatTreeTopo(k=4)

    def test_topo_graph_is_fattreegraph(self):
        self.assertIsInstance(self.topo.graph, FatTreeGraph)

    def test_topo_delegates_get_paths(self):
        paths = self.topo.get_paths("h_0_0_0", "h_1_0_0")
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_topo_delegates_get_host_ip(self):
        ip = self.topo.get_host_ip("h_0_0_0")
        self.assertEqual(ip, "10.0.0.2")

    def test_topo_delegates_get_host_mac(self):
        mac = self.topo.get_host_mac("h_0_0_0")
        self.assertEqual(mac, "00:00:0a:00:00:02")

    def test_topo_summary(self):
        s = self.topo.summary()
        self.assertIn("k=4", s)

    def test_topo_repr(self):
        r = repr(self.topo)
        self.assertIn("FatTreeTopo", r)
        self.assertIn("k=4", r)

    def test_topo_invalid_k(self):
        with self.assertRaises(ValueError):
            FatTreeTopo(k=5)

    def test_topo_k8_host_count(self):
        t = FatTreeTopo(k=8)
        self.assertEqual(t.graph.n_hosts, 128)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LAFS topology unit tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", "--filter", default="",
                        help="Run only test methods matching this substring")
    args, remaining = parser.parse_known_args()

    verbosity = 2 if args.verbose else 1

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestNamingHelpers,
        TestFatTreeGraphK4,
        TestFatTreeGraphK8,
        TestFatTreeGraphPaths,
        TestFatTreeGraphEdges,
        TestFatTreeGraphTopology,
        TestFatTreeTopoStub,
    ]

    for cls in test_classes:
        tests = loader.loadTestsFromTestCase(cls)
        if args.filter:
            tests = unittest.TestSuite(
                t for t in tests
                if args.filter.lower() in t._testMethodName.lower()
            )
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
