"""
LAFS Project — Integration Tests: Fat-Tree Topology
=====================================================
COMP-6910 — Group 7

End-to-end tests that start a real Mininet network, connect it to a Ryu
controller, install flow rules, and verify correctness.

Requirements:
  - Linux (Ubuntu 20.04 / 22.04)
  - Root privileges (sudo)
  - Mininet 2.3.0 installed (run setup/install_mininet.sh)
  - Ryu controller installed
  - Gurobi or PuLP for MILP verification (optional)

Test Categories:
  TestTopologyStartup       — network creation, switch counts, OF 1.3 config
  TestControllerConnectivity— all switches connect to remote controller
  TestHostConnectivity      — ping-all, cross-pod, within-pod
  TestFlowRules             — flow rule installation and correctness
  TestPerformance128Hosts   — iperf3, BW, scalability
  TestNetworkBuilder        — NetworkBuilder convenience API

Usage:
    sudo python tests/integration/test_topology_integration.py
    sudo python tests/integration/test_topology_integration.py --k 4
    sudo python tests/integration/test_topology_integration.py --k 8 --verbose

    # With pytest (requires pytest-timeout):
    sudo pytest tests/integration/test_topology_integration.py -v --timeout=300
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from typing import ClassVar, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Platform guards ───────────────────────────────────────────────────────────
IS_LINUX = sys.platform == "linux"
IS_ROOT  = os.geteuid() == 0 if IS_LINUX else False


def skip_non_linux(cls):
    """Class decorator: skip entire class on non-Linux."""
    if not IS_LINUX:
        return unittest.skip("Requires Linux")(cls)
    return cls


def skip_no_root(cls):
    """Class decorator: skip entire class without root."""
    if not IS_ROOT:
        return unittest.skip("Requires root (sudo)")(cls)
    return cls


def skip_no_mininet(cls):
    """Class decorator: skip entire class if Mininet not installed."""
    try:
        import mininet  # noqa: F401
    except ImportError:
        return unittest.skip("Mininet not installed")(cls)
    return cls


# ── Minimal Ryu app written to /tmp ──────────────────────────────────────────
_RYU_L2_APP = '''
"""
Minimal Ryu L2 Learning Switch for LAFS integration tests.
Installs exact-match flow rules using OpenFlow 1.3.
"""
from collections import defaultdict
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types


class LAFSL2Switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = defaultdict(dict)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self._add_flow(dp, 0, match, actions)

    def _add_flow(self, dp, priority, match, actions, idle=0, hard=0):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        instr = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp, priority=priority, match=match,
            instructions=instr, idle_timeout=idle, hard_timeout=hard,
        )
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst, src, dpid = eth.dst, eth.src, dp.id
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(dp, 1, match, actions, idle=60)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data,
        )
        dp.send_msg(out)
'''

CONTROLLER_PORT = 6635   # non-standard port to avoid conflicting with production


# ─────────────────────────────────────────────────────────────────────────────
#  Shared test fixture
# ─────────────────────────────────────────────────────────────────────────────
class MininetFixture:
    """Shared setup/teardown for Mininet-based test classes.

    Starts a Ryu controller process and a Mininet k=4 Fat-tree network
    before all tests in a class run, then shuts them down after.

    Subclass and set ``K`` to change the Fat-tree size.
    """

    K: ClassVar[int] = 4          # Override in subclasses for k=8
    net   = None
    ryu_proc = None
    app_path: ClassVar[str] = ""

    @classmethod
    def start_controller(cls):
        """Write the Ryu app to /tmp and start ryu-manager."""
        fd, cls.app_path = tempfile.mkstemp(suffix=".py", prefix="lafs_ryu_")
        with os.fdopen(fd, "w") as f:
            f.write(_RYU_L2_APP)

        ryu_bin = shutil.which("ryu-manager")
        if ryu_bin is None:
            return False

        cls.ryu_proc = subprocess.Popen(
            [ryu_bin, cls.app_path,
             f"--ofp-tcp-listen-port={CONTROLLER_PORT}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)   # Allow Ryu to bind
        return True

    @classmethod
    def start_network(cls):
        """Build and start the Mininet Fat-tree."""
        from mininet.net import Mininet
        from mininet.node import OVSSwitch, RemoteController
        from mininet.log import setLogLevel
        from mininet.clean import cleanup
        from src.topology.fattree import FatTreeTopo

        setLogLevel("warning")
        cleanup()

        topo = FatTreeTopo(k=cls.K, bw_host=0.1, bw_core=1.0)
        cls.net = Mininet(
            topo=topo,
            switch=OVSSwitch,
            controller=None,
            autoSetMacs=True,
            autoStaticArp=True,
        )
        cls.net.addController(
            "ctrl",
            controller=RemoteController,
            ip="127.0.0.1",
            port=CONTROLLER_PORT,
        )
        cls.net.start()

        # Force OF 1.3
        for sw in cls.net.switches:
            sw.cmd(f"ovs-vsctl set bridge {sw.name} protocols=OpenFlow13")

        time.sleep(3)   # Allow switches to connect and L2 to learn

    @classmethod
    def teardown(cls):
        if cls.net:
            cls.net.stop()
        if cls.ryu_proc and cls.ryu_proc.poll() is None:
            cls.ryu_proc.terminate()
            cls.ryu_proc.wait()
        try:
            if cls.app_path:
                os.remove(cls.app_path)
        except Exception:
            pass
        try:
            from mininet.clean import cleanup
            cleanup()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1 — Topology Startup
# ─────────────────────────────────────────────────────────────────────────────
@skip_no_mininet
@skip_non_linux
@skip_no_root
class TestTopologyStartup(MininetFixture, unittest.TestCase):
    """Verify Mininet network starts with correct node counts and configuration."""

    K = 4   # Faster for startup test

    @classmethod
    def setUpClass(cls):
        cls.start_controller()
        cls.start_network()

    @classmethod
    def tearDownClass(cls):
        cls.teardown()

    def test_switch_count_k4(self):
        """k=4: 20 switches (4 core + 8 agg + 8 edge)."""
        self.assertEqual(len(self.net.switches), 20)

    def test_host_count_k4(self):
        """k=4: 16 hosts."""
        self.assertEqual(len(self.net.hosts), 16)

    def test_switches_are_ovs(self):
        """All switches should be OVSSwitch instances."""
        from mininet.node import OVSSwitch
        for sw in self.net.switches:
            self.assertIsInstance(sw, OVSSwitch,
                                  f"{sw.name} is not OVSSwitch")

    def test_openflow13_on_all_switches(self):
        """Every switch must have OpenFlow 1.3 protocol set."""
        for sw in self.net.switches:
            proto = sw.cmd(f"ovs-vsctl get bridge {sw.name} protocols")
            self.assertIn("OpenFlow13", proto,
                          f"{sw.name} missing OpenFlow 1.3: {proto}")

    def test_host_ips_set(self):
        """All hosts should have a 10.x.x.x IP address."""
        for h in self.net.hosts:
            ip = h.IP()
            self.assertTrue(
                ip.startswith("10."),
                f"Host {h.name} has unexpected IP: {ip}"
            )

    def test_no_duplicate_ips(self):
        ips = [h.IP() for h in self.net.hosts]
        self.assertEqual(len(ips), len(set(ips)), "Duplicate IPs detected")

    def test_switch_names_follow_convention(self):
        """Switch names should match c_*, a_*, or e_* pattern."""
        import re
        pattern = re.compile(r"^(c|a|e)_\d+_\d+$")
        for sw in self.net.switches:
            self.assertTrue(
                pattern.match(sw.name),
                f"Switch '{sw.name}' doesn't follow naming convention"
            )

    def test_host_names_follow_convention(self):
        """Host names should match h_*_*_* pattern."""
        import re
        pattern = re.compile(r"^h_\d+_\d+_\d+$")
        for h in self.net.hosts:
            self.assertTrue(
                pattern.match(h.name),
                f"Host '{h.name}' doesn't follow naming convention"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2 — Controller Connectivity
# ─────────────────────────────────────────────────────────────────────────────
@skip_no_mininet
@skip_non_linux
@skip_no_root
class TestControllerConnectivity(MininetFixture, unittest.TestCase):
    """All switches must connect to the Ryu remote controller."""

    K = 4

    @classmethod
    def setUpClass(cls):
        ok = cls.start_controller()
        if not ok:
            return
        cls.start_network()
        time.sleep(3)   # Extra wait for controller handshake

    @classmethod
    def tearDownClass(cls):
        cls.teardown()

    def _controller_running(self):
        return self.ryu_proc is not None and self.ryu_proc.poll() is None

    def test_ryu_process_running(self):
        if not self._controller_running():
            self.skipTest("ryu-manager not found")
        self.assertIsNone(
            self.ryu_proc.poll(),
            "Ryu process died during test"
        )

    def test_all_switches_connected(self):
        """All switches must report is_connected = true."""
        if not self._controller_running():
            self.skipTest("ryu-manager not found")

        disconnected = []
        for sw in self.net.switches:
            result = sw.cmd(
                f"ovs-vsctl get controller {sw.name} is_connected"
            ).strip()
            if result != "true":
                disconnected.append(sw.name)

        self.assertEqual(
            disconnected, [],
            f"Switches not connected to controller: {disconnected}"
        )

    def test_controller_port_open(self):
        """TCP port CONTROLLER_PORT must be open and accepting connections."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("127.0.0.1", CONTROLLER_PORT))
        sock.close()
        self.assertEqual(result, 0,
                         f"Controller port {CONTROLLER_PORT} not reachable")

    def test_switches_have_controller_configured(self):
        """Each switch should have a controller entry in OVS."""
        for sw in self.net.switches:
            ctrl = sw.cmd(f"ovs-vsctl get-controller {sw.name}").strip()
            self.assertIn(
                str(CONTROLLER_PORT), ctrl,
                f"{sw.name} controller not pointing to port {CONTROLLER_PORT}: {ctrl}"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3 — Host Connectivity
# ─────────────────────────────────────────────────────────────────────────────
@skip_no_mininet
@skip_non_linux
@skip_no_root
class TestHostConnectivity(MininetFixture, unittest.TestCase):
    """Ping tests between host pairs in various topology positions."""

    K = 4

    @classmethod
    def setUpClass(cls):
        cls.start_controller()
        cls.start_network()
        # Warm-up: trigger L2 learning with initial pings
        if cls.net:
            cls.net.pingAll(timeout=2)
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.teardown()

    def _get_host(self, name: str):
        return self.net.get(name)

    def test_same_edge_switch_ping(self):
        """Two hosts on the same edge switch should ping successfully."""
        h0 = self._get_host("h_0_0_0")
        h1 = self._get_host("h_0_0_1")
        result = h0.cmd(f"ping -c 5 -W 2 {h1.IP()}")
        self.assertIn("0% packet loss", result,
                      f"Same-edge ping failed:\n{result}")

    def test_within_pod_ping(self):
        """Hosts in the same pod, different edge switches, should ping."""
        h0 = self._get_host("h_0_0_0")
        h1 = self._get_host("h_0_1_0")
        result = h0.cmd(f"ping -c 5 -W 2 {h1.IP()}")
        self.assertIn("0% packet loss", result,
                      f"Within-pod ping failed:\n{result}")

    def test_cross_pod_ping(self):
        """Hosts in different pods should ping (uses core switches)."""
        h0 = self._get_host("h_0_0_0")
        h1 = self._get_host("h_1_0_0")
        result = h0.cmd(f"ping -c 5 -W 3 {h1.IP()}")
        self.assertIn("0% packet loss", result,
                      f"Cross-pod ping failed:\n{result}")

    def test_ping_all(self):
        """All host pairs should be reachable (pingAll)."""
        loss = self.net.pingAll(timeout=3)
        self.assertEqual(loss, 0.0,
                         f"PingAll: {loss:.1f}% packet loss")

    def test_ping_latency_reasonable(self):
        """RTT between two hosts should be < 50ms (network is virtual)."""
        import re
        h0 = self._get_host("h_0_0_0")
        h1 = self._get_host("h_1_0_0")
        result = h0.cmd(f"ping -c 5 -W 2 {h1.IP()}")
        m = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", result)
        if m:
            avg_rtt = float(m.group(1))
            self.assertLess(avg_rtt, 50.0,
                            f"Cross-pod RTT too high: {avg_rtt}ms")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4 — Flow Rules
# ─────────────────────────────────────────────────────────────────────────────
@skip_no_mininet
@skip_non_linux
@skip_no_root
class TestFlowRules(MininetFixture, unittest.TestCase):
    """Verify flow rules are installed correctly after traffic."""

    K = 4

    @classmethod
    def setUpClass(cls):
        cls.start_controller()
        cls.start_network()
        # Generate traffic to trigger flow rule installation
        if cls.net:
            h0 = cls.net.get("h_0_0_0")
            h1 = cls.net.get("h_1_0_0")
            h0.cmd(f"ping -c 3 {h1.IP()}")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.teardown()

    def test_table_miss_entry_exists(self):
        """Every switch must have the table-miss flow entry (priority 0)."""
        for sw in self.net.switches:
            flows = sw.cmd(f"ovs-ofctl -O OpenFlow13 dump-flows {sw.name}")
            self.assertIn(
                "priority=0", flows,
                f"Table-miss (priority=0) not found on {sw.name}"
            )

    def test_forwarding_rules_installed(self):
        """After traffic, at least some switches should have priority-1 rules."""
        rules_found = False
        for sw in self.net.switches:
            flows = sw.cmd(f"ovs-ofctl -O OpenFlow13 dump-flows {sw.name}")
            if "priority=1" in flows:
                rules_found = True
                break
        self.assertTrue(rules_found,
                        "No priority-1 forwarding rules installed after traffic")

    def test_flow_rules_have_correct_version(self):
        """Flow rules should be OpenFlow 1.3 (checked via ovs-ofctl -O flag)."""
        sw = self.net.switches[0]
        result = sw.cmd(
            f"ovs-ofctl -O OpenFlow13 dump-flows {sw.name} 2>&1"
        )
        # If OF version is wrong, ovs-ofctl would print an error
        self.assertNotIn("unsupported", result.lower(),
                         f"OF 1.3 version issue on {sw.name}: {result}")

    def test_dump_flows_all_switches(self):
        """dump-flows should succeed on every switch without error."""
        for sw in self.net.switches:
            result = sw.cmd(
                f"ovs-ofctl -O OpenFlow13 dump-flows {sw.name} 2>&1"
            )
            self.assertNotIn("Error", result,
                             f"Error in dump-flows for {sw.name}: {result}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5 — Performance: 128 Hosts (k=8)
# ─────────────────────────────────────────────────────────────────────────────
@skip_no_mininet
@skip_non_linux
@skip_no_root
class TestPerformance128Hosts(MininetFixture, unittest.TestCase):
    """Performance and scalability tests for the full k=8 (128-host) topology.

    These are the slowest tests.  Only run on a machine with >= 8 GB RAM.
    """

    K = 8   # Full 128-host topology

    @classmethod
    def setUpClass(cls):
        # Check available RAM (128-host Mininet needs ~6 GB)
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            import re
            m = re.search(r"MemAvailable:\s+(\d+)", meminfo)
            if m and int(m.group(1)) < 4_000_000:   # < 4 GB free
                cls._skip_reason = "Insufficient RAM for k=8 (need >= 4 GB free)"
                return
        except Exception:
            pass

        cls._skip_reason = None
        cls.start_controller()
        cls.start_network()

    @classmethod
    def tearDownClass(cls):
        cls.teardown()

    def _maybe_skip(self):
        if getattr(self.__class__, "_skip_reason", None):
            self.skipTest(self.__class__._skip_reason)

    def test_host_count_k8(self):
        """k=8: exactly 128 hosts."""
        self._maybe_skip()
        self.assertEqual(len(self.net.hosts), 128)

    def test_switch_count_k8(self):
        """k=8: exactly 80 switches."""
        self._maybe_skip()
        self.assertEqual(len(self.net.switches), 80)

    def test_all_switches_of13_k8(self):
        """All 80 switches must have OpenFlow 1.3."""
        self._maybe_skip()
        for sw in self.net.switches:
            proto = sw.cmd(f"ovs-vsctl get bridge {sw.name} protocols")
            self.assertIn("OpenFlow13", proto)

    def test_sample_ping_k8(self):
        """Spot-check: a sample of host pairs should ping successfully."""
        self._maybe_skip()
        import random
        random.seed(42)
        hosts = self.net.hosts
        pairs = random.sample(range(len(hosts)), min(10, len(hosts)))

        for i in range(0, len(pairs) - 1, 2):
            h0 = hosts[pairs[i]]
            h1 = hosts[pairs[i + 1]]
            result = h0.cmd(f"ping -c 3 -W 2 {h1.IP()}")
            # Partial packet loss acceptable on first ping (ARP)
            packets_rcvd = int(
                __import__("re").search(
                    r"(\d+) received", result
                ).group(1)
                if __import__("re").search(r"(\d+) received", result)
                else 0
            )
            self.assertGreater(
                packets_rcvd, 0,
                f"All pings lost: {h0.name} → {h1.name}\n{result}"
            )

    def test_iperf3_single_pair_k8(self):
        """Single iperf3 pair should achieve measurable throughput."""
        self._maybe_skip()
        h0 = self.net.get("h_0_0_0")
        h1 = self.net.get("h_1_0_0")

        h1.cmd("iperf3 -s -D --logfile /tmp/iperf3_h1.log")
        time.sleep(0.5)

        result = h0.cmd(f"iperf3 -c {h1.IP()} -t 3 --json 2>/dev/null")
        h1.cmd("pkill iperf3")

        self.assertIn(
            "bits_per_second", result,
            f"iperf3 did not produce expected output:\n{result}"
        )

    def test_controller_handles_all_switches(self):
        """Verify the controller is connected to all 80 switches."""
        self._maybe_skip()
        if self.ryu_proc is None or self.ryu_proc.poll() is not None:
            self.skipTest("ryu-manager not running")

        connected = sum(
            1 for sw in self.net.switches
            if sw.cmd(
                f"ovs-vsctl get controller {sw.name} is_connected"
            ).strip() == "true"
        )
        # Require >= 95% connection rate (network startup timing)
        total = len(self.net.switches)
        self.assertGreaterEqual(
            connected / total, 0.95,
            f"Only {connected}/{total} switches connected"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6 — NetworkBuilder API
# ─────────────────────────────────────────────────────────────────────────────
@skip_no_mininet
@skip_non_linux
@skip_no_root
class TestNetworkBuilderAPI(unittest.TestCase):
    """Test NetworkBuilder convenience methods using a k=4 network."""

    @classmethod
    def setUpClass(cls):
        from src.topology.network_builder import NetworkConfig, ControllerConfig

        # Write Ryu app
        fd, cls.app_path = tempfile.mkstemp(suffix=".py", prefix="lafs_nb_")
        with os.fdopen(fd, "w") as f:
            f.write(_RYU_L2_APP)

        ryu_bin = shutil.which("ryu-manager")
        if ryu_bin is None:
            cls.ryu_proc = None
        else:
            cls.ryu_proc = subprocess.Popen(
                [ryu_bin, cls.app_path,
                 f"--ofp-tcp-listen-port={CONTROLLER_PORT}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)

        from src.topology.network_builder import NetworkBuilder
        cfg = NetworkConfig(
            k=4,
            bw_host=0.1,
            bw_core=1.0,
            controller=ControllerConfig(
                host="127.0.0.1",
                port=CONTROLLER_PORT,
                timeout=20,
            ),
        )
        cls.nb = NetworkBuilder(cfg)
        try:
            cls.nb.start()
            time.sleep(3)
        except Exception as exc:
            cls._start_error = str(exc)
        else:
            cls._start_error = None

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "nb"):
            cls.nb.stop()
        if hasattr(cls, "ryu_proc") and cls.ryu_proc:
            cls.ryu_proc.terminate()
            cls.ryu_proc.wait()
        try:
            if cls.app_path:
                os.remove(cls.app_path)
        except Exception:
            pass
        try:
            from mininet.clean import cleanup
            cleanup()
        except Exception:
            pass

    def _require_started(self):
        if self.__class__._start_error:
            self.skipTest(f"NetworkBuilder failed to start: {self._start_error}")

    def test_controller_reachable(self):
        self._require_started()
        reachable = self.nb.test_controller_reachable()
        self.assertTrue(reachable)

    def test_switch_connectivity(self):
        self._require_started()
        connected, total = self.nb.test_all_switches_connected()
        self.assertEqual(total, 20)   # k=4: 20 switches
        self.assertGreaterEqual(connected, 18)  # Allow minor startup lag

    def test_pingall_via_builder(self):
        self._require_started()
        loss = self.nb.pingall(timeout=3.0)
        self.assertLessEqual(loss, 5.0,   # Allow up to 5% loss
                             f"PingAll loss too high: {loss}%")

    def test_ping_pair_via_builder(self):
        self._require_started()
        sent, rcvd, rtt = self.nb.ping_pair("h_0_0_0", "h_1_0_0", count=5)
        self.assertGreater(rcvd, 0,
                           "ping_pair returned 0 received packets")

    def test_count_flow_rules(self):
        self._require_started()
        # Generate traffic first
        self.nb.host_cmd("h_0_0_0",
                         f"ping -c 3 {self.nb.graph.get_host_ip('h_1_0_0')}")
        time.sleep(1)

        counts = self.nb.count_flow_rules()
        self.assertEqual(len(counts), 20,  # k=4: 20 switches
                         f"Expected 20 switches in count_flow_rules")
        # At least some switches should have installed forwarding rules
        total_rules = sum(counts.values())
        self.assertGreater(total_rules, 20,  # More than just table-miss
                           f"Only {total_rules} flow rules installed (expected > 20)")

    def test_dump_flows(self):
        self._require_started()
        flows = self.nb.dump_flows("e_0_0")
        self.assertIsInstance(flows, str)
        self.assertGreater(len(flows), 0)

    def test_get_host_raises_if_not_found(self):
        self._require_started()
        with self.assertRaises(KeyError):
            self.nb.get_host("h_99_99_99")

    def test_get_switch_raises_if_not_found(self):
        self._require_started()
        with self.assertRaises(KeyError):
            self.nb.get_switch("c_99_99")

    def test_host_cmd(self):
        self._require_started()
        result = self.nb.host_cmd("h_0_0_0", "hostname")
        self.assertIn("h_0_0_0", result)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAFS topology integration tests")
    parser.add_argument("--k", type=int, default=4,
                        help="Fat-tree k parameter (4 or 8). Default: 4")
    parser.add_argument("--skip-k8", action="store_true",
                        help="Skip the k=8 performance tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args()

    # Override K for the performance test class if requested
    if args.k == 8:
        TestTopologyStartup.K = 8
        TestControllerConnectivity.K = 8
        TestHostConnectivity.K = 8
        TestFlowRules.K = 8

    verbosity = 2 if args.verbose else 1
    loader    = unittest.TestLoader()
    suite     = unittest.TestSuite()

    classes = [
        TestTopologyStartup,
        TestControllerConnectivity,
        TestHostConnectivity,
        TestFlowRules,
        TestNetworkBuilderAPI,
    ]
    if not args.skip_k8:
        classes.append(TestPerformance128Hosts)

    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
