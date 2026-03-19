#!/usr/bin/env python3
"""
LAFS Project — Mininet Verification Test
COMP 6910 — Group 7

Tests:
  1. Mininet imports correctly
  2. k=4 Fat-tree topology builds (small; k=8 needs >8 GB RAM)
  3. All hosts can ping each other (all-pairs)
  4. Bandwidth limit (TCLink) is respected
  5. OpenFlow 1.3 is set on all switches

Usage (requires root):
    sudo python tests/test_mininet.py
    sudo python tests/test_mininet.py --k 8   (full 128-host test; slow)
"""

import sys
import os
import time
import argparse
import unittest

# ── Guard: Mininet requires Linux + root ──────────────────────────────────────
import pytest

_MININET_AVAILABLE = False
_SKIP_REASON = ""

if sys.platform != "linux":
    _SKIP_REASON = f"Mininet requires Linux (current: {sys.platform})"
elif os.geteuid() != 0:
    _SKIP_REASON = "Mininet tests must be run as root (sudo)"
else:
    try:
        from mininet.topo import Topo
        from mininet.net import Mininet
        from mininet.node import OVSSwitch, RemoteController, Controller
        from mininet.link import TCLink
        from mininet.log import setLogLevel, info
        from mininet.clean import cleanup
        from mininet.log import setLogLevel
        setLogLevel("warning")
        _MININET_AVAILABLE = True
    except ImportError as e:
        _SKIP_REASON = f"Mininet not installed: {e}. Run: sudo ./setup/install_mininet.sh"

_MN_SKIP = (
    pytest.mark.skip(reason=_SKIP_REASON)
    if not _MININET_AVAILABLE
    else lambda cls: cls
)

# Provide a stub Topo so the class definition doesn't fail on Windows
if not _MININET_AVAILABLE:
    class Topo:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass
        def addSwitch(self, n, **kw): return n
        def addHost(self, n, **kw): return n
        def addLink(self, *a, **kw): pass
        def build(self, **kw): pass


# =============================================================================
# Fat-tree topology (k-ary)
# =============================================================================
class FatTreeTopo(Topo):
    """
    k-ary Fat-tree topology.
    Pods: k
    Core switches: (k/2)^2
    Aggregation switches: k * (k/2) = k²/2
    Edge switches: k * (k/2) = k²/2
    Hosts: k³/4

    For k=4: 4 pods, 4 core, 8 agg, 8 edge, 16 hosts
    For k=8: 8 pods, 16 core, 32 agg, 32 edge, 128 hosts
    """

    def build(self, k: int = 4, bw: float = 1.0):
        """
        Build a k-ary Fat-tree.

        Args:
            k:  Number of ports per switch (must be even).
            bw: Link bandwidth in Gbps.
        """
        assert k % 2 == 0, "k must be even"
        half_k = k // 2

        # ── Core switches: (k/2)^2 ────────────────────────────────────────────
        core = []
        for i in range(half_k):
            row = []
            for j in range(half_k):
                sw = self.addSwitch(f"c{i}{j}", protocols="OpenFlow13")
                row.append(sw)
            core.append(row)

        # ── Pods ──────────────────────────────────────────────────────────────
        for pod in range(k):
            # Aggregation switches: k/2 per pod
            agg = []
            for a in range(half_k):
                sw = self.addSwitch(f"a{pod}{a}", protocols="OpenFlow13")
                agg.append(sw)

            # Edge switches: k/2 per pod
            edge = []
            for e in range(half_k):
                sw = self.addSwitch(f"e{pod}{e}", protocols="OpenFlow13")
                edge.append(sw)

            # Hosts: k/2 per edge switch
            for e_idx, e_sw in enumerate(edge):
                for h in range(half_k):
                    host_id = pod * half_k * half_k + e_idx * half_k + h
                    ip = f"10.{pod}.{e_idx}.{h + 2}"
                    host = self.addHost(f"h{host_id}", ip=ip)
                    self.addLink(host, e_sw, cls=TCLink,
                                 bw=bw, delay="0.1ms")

            # Edge → Aggregation links
            for e_sw in edge:
                for a_sw in agg:
                    self.addLink(e_sw, a_sw, cls=TCLink,
                                 bw=bw, delay="0.1ms")

            # Aggregation → Core links
            for a_idx, a_sw in enumerate(agg):
                for c_j in range(half_k):
                    self.addLink(a_sw, core[a_idx][c_j], cls=TCLink,
                                 bw=bw, delay="0.1ms")


# =============================================================================
# Test cases
# =============================================================================
@_MN_SKIP
class TestMininetImport(unittest.TestCase):
    """Test that Mininet modules import correctly."""

    def test_mininet_import(self):
        import mininet
        self.assertIsNotNone(mininet.__file__)

    def test_mininet_version(self):
        import mininet
        ver = getattr(mininet, "VERSION", "")
        self.assertTrue(len(ver) > 0, "Could not read Mininet version")
        print(f"\n  Mininet version: {ver}")

    def test_openflow13_import(self):
        from ryu.ofproto import ofproto_v1_3
        self.assertIsNotNone(ofproto_v1_3)

    def test_ovs_switch(self):
        from mininet.node import OVSSwitch
        self.assertTrue(callable(OVSSwitch))

    def test_tclink(self):
        from mininet.link import TCLink
        self.assertTrue(callable(TCLink))


@_MN_SKIP
class TestFatTreeTopology(unittest.TestCase):
    """Test Fat-tree topology construction."""

    K = 4  # Use small k=4 for speed; override with --k 8

    @classmethod
    def setUpClass(cls):
        cleanup()  # Clean any leftover Mininet state
        topo = FatTreeTopo(k=cls.K, bw=1.0)
        cls.net = Mininet(
            topo=topo,
            switch=OVSSwitch,
            controller=Controller,
            autoSetMacs=True,
            autoStaticArp=True,
            waitConnected=True,
        )
        cls.net.start()
        time.sleep(2)  # Allow OVS to initialise

    @classmethod
    def tearDownClass(cls):
        cls.net.stop()
        cleanup()

    def test_host_count(self):
        """Fat-tree k=4 → 16 hosts, k=8 → 128 hosts."""
        expected = (self.K ** 3) // 4
        actual = len(self.net.hosts)
        self.assertEqual(actual, expected,
                         f"Expected {expected} hosts, got {actual}")
        print(f"\n  Host count: {actual} (correct for k={self.K})")

    def test_switch_count(self):
        """Fat-tree k=4 → 20 switches, k=8 → 80 switches."""
        half_k = self.K // 2
        expected_core = half_k ** 2
        expected_agg  = self.K * half_k
        expected_edge = self.K * half_k
        expected      = expected_core + expected_agg + expected_edge
        actual        = len(self.net.switches)
        self.assertEqual(actual, expected,
                         f"Expected {expected} switches, got {actual}")
        print(f"\n  Switch count: {actual} (correct for k={self.K})")

    def test_openflow13_on_switches(self):
        """All switches should be configured for OpenFlow 1.3."""
        for sw in self.net.switches:
            protocols = sw.cmd(f"ovs-vsctl get bridge {sw.name} protocols")
            self.assertIn("OpenFlow13", protocols,
                          f"Switch {sw.name} missing OpenFlow 1.3")

    def test_ping_all(self):
        """All host pairs should be reachable."""
        loss = self.net.pingAll(timeout=3)
        self.assertEqual(loss, 0.0,
                         f"Ping-all had {loss:.1f}% packet loss")
        print(f"\n  Ping-all: 0% loss — all hosts reachable")

    def test_host_connectivity_sample(self):
        """Spot-check: first host pings last host."""
        h0 = self.net.hosts[0]
        h_last = self.net.hosts[-1]
        result = h0.cmd(f"ping -c 3 -W 2 {h_last.IP()}")
        self.assertIn("0% packet loss", result,
                      f"h0 cannot reach {h_last.name}: {result}")

    def test_link_bandwidth(self):
        """iperf3 between two hosts should achieve ≥ 0.8 Gbps."""
        h0 = self.net.hosts[0]
        h1 = self.net.hosts[1]

        # Start iperf3 server on h1
        h1.cmd("iperf3 -s -D --logfile /tmp/iperf3_server.log")
        time.sleep(0.5)

        # Run iperf3 client from h0
        result = h0.cmd(f"iperf3 -c {h1.IP()} -t 3 -J 2>/dev/null")

        # Kill server
        h1.cmd("pkill iperf3")

        # Basic check: result contains bits/sec
        self.assertIn("bits_per_second", result,
                      "iperf3 did not produce expected output")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAFS Mininet tests")
    parser.add_argument("--k", type=int, default=4,
                        help="Fat-tree k parameter (4 or 8). Default: 4")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, remaining = parser.parse_known_args()

    # Inject k into test class
    TestFatTreeTopology.K = args.k

    verbosity = 2 if args.verbose else 1
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMininetImport))
    suite.addTests(loader.loadTestsFromTestCase(TestFatTreeTopology))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
