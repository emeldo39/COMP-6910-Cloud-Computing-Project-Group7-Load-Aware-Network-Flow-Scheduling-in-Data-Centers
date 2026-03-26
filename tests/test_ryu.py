#!/usr/bin/env python3
"""
LAFS Project — Ryu Controller Verification Test
COMP-6910 — Group 7

Tests:
  1. Ryu imports correctly
  2. OpenFlow 1.3 protocol module present
  3. Ryu controller can start and bind to a port
  4. REST API app is accessible
  5. Basic OpenFlow 1.3 message construction

Usage:
    python tests/test_ryu.py
    pytest tests/test_ryu.py -v
"""

import sys
import os
import time
import socket
import unittest
import subprocess
import signal
import threading

# ── Ryu is Linux-only — skip entire file on Windows ──────────────────────────
_RYU_AVAILABLE = False
try:
    import ryu  # noqa: F401
    _RYU_AVAILABLE = True
except ImportError:
    pass

if not _RYU_AVAILABLE:
    # Mark every test class as skipped at collection time
    _RYU_SKIP = unittest.skip(
        "Ryu not installed on this platform (Linux/Ubuntu only). "
        "Run on Ubuntu after: pip install ryu==4.34"
    )
else:
    _RYU_SKIP = lambda cls: cls  # no-op decorator

# =============================================================================
# Test 1 — Ryu module imports
# =============================================================================
@_RYU_SKIP
class TestRyuImports(unittest.TestCase):
    """Verify all required Ryu modules can be imported."""

    def test_ryu_base(self):
        import ryu
        self.assertIsNotNone(ryu.__version__)
        print(f"\n  Ryu version: {ryu.__version__}")

    def test_ofproto_v1_3(self):
        from ryu.ofproto import ofproto_v1_3
        self.assertIsNotNone(ofproto_v1_3.OFP_VERSION)
        self.assertEqual(ofproto_v1_3.OFP_VERSION, 0x04,
                         "OpenFlow 1.3 version byte should be 0x04")

    def test_ofproto_v1_3_parser(self):
        from ryu.ofproto import ofproto_v1_3_parser
        self.assertTrue(hasattr(ofproto_v1_3_parser, "OFPFlowMod"))
        self.assertTrue(hasattr(ofproto_v1_3_parser, "OFPMatch"))
        self.assertTrue(hasattr(ofproto_v1_3_parser, "OFPActionOutput"))

    def test_ryu_app_manager(self):
        from ryu.base import app_manager
        self.assertTrue(hasattr(app_manager, "RyuApp"))

    def test_ryu_controller(self):
        from ryu.controller import ofp_event
        from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER
        self.assertIsNotNone(ofp_event)

    def test_ryu_topology(self):
        from ryu.topology import api as topo_api
        self.assertTrue(hasattr(topo_api, "get_switch"))
        self.assertTrue(hasattr(topo_api, "get_link"))

    def test_ryu_ofctl_rest(self):
        from ryu.app import ofctl_rest
        self.assertIsNotNone(ofctl_rest)

    def test_ryu_lib_packet(self):
        from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
        self.assertIsNotNone(packet.Packet)

    def test_ryu_lib_hub(self):
        from ryu.lib import hub
        self.assertIsNotNone(hub)


# =============================================================================
# Test 2 — OpenFlow 1.3 message construction
# =============================================================================
@_RYU_SKIP
class TestOpenFlow13Messages(unittest.TestCase):
    """Verify OF 1.3 messages can be constructed without errors."""

    def setUp(self):
        from ryu.ofproto import ofproto_v1_3, ofproto_v1_3_parser
        self.ofproto = ofproto_v1_3
        self.parser  = ofproto_v1_3_parser

    def test_ofp_match_constructor(self):
        """Build an OFPMatch for IP traffic."""
        match = self.parser.OFPMatch(
            eth_type=0x0800,    # IPv4
            ip_proto=6,          # TCP
            ipv4_src="10.0.0.1",
            ipv4_dst="10.0.0.2",
            tcp_dst=80,
        )
        self.assertIsNotNone(match)

    def test_action_output(self):
        """Build an output action to port 1."""
        action = self.parser.OFPActionOutput(port=1, max_len=0xffff)
        self.assertIsNotNone(action)
        self.assertEqual(action.port, 1)

    def test_action_set_queue(self):
        """Build a set-queue action for QoS."""
        action = self.parser.OFPActionSetQueue(queue_id=1)
        self.assertIsNotNone(action)

    def test_group_action(self):
        """Build a group action for multipath forwarding."""
        action = self.parser.OFPActionGroup(group_id=0)
        self.assertIsNotNone(action)

    def test_meter_instruction(self):
        """Build a meter instruction for rate limiting."""
        instr = self.parser.OFPInstructionMeter(meter_id=1)
        self.assertIsNotNone(instr)

    def test_flow_mod_structure(self):
        """Verify OFPFlowMod has required fields."""
        self.assertTrue(hasattr(self.parser, "OFPFlowMod"))
        # Check the command constants
        self.assertTrue(hasattr(self.ofproto, "OFPFC_ADD"))
        self.assertTrue(hasattr(self.ofproto, "OFPFC_MODIFY"))
        self.assertTrue(hasattr(self.ofproto, "OFPFC_DELETE"))

    def test_port_stats_request(self):
        """Build a port stats request."""
        self.assertTrue(hasattr(self.parser, "OFPPortStatsRequest"))

    def test_flow_stats_request(self):
        """Build a flow stats request."""
        self.assertTrue(hasattr(self.parser, "OFPFlowStatsRequest"))

    def test_group_stats_request(self):
        """Build a group stats request."""
        self.assertTrue(hasattr(self.parser, "OFPGroupStatsRequest"))


# =============================================================================
# Test 3 — Ryu controller process startup
# =============================================================================
@_RYU_SKIP
class TestRyuControllerProcess(unittest.TestCase):
    """
    Starts a minimal Ryu controller process and verifies it binds to port 6633.
    NOTE: This test needs ryu-manager in PATH and an available port 6633.
    """

    CONTROLLER_PORT = 6634   # Use 6634 to avoid conflicting with a running controller
    process = None

    @classmethod
    def setUpClass(cls):
        """Write a minimal Ryu app to /tmp and start ryu-manager."""
        cls.app_path = "/tmp/lafs_test_app.py"
        with open(cls.app_path, "w") as f:
            f.write("""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3

class LAFSTestApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        pass  # Minimal: just accept connections
""")
        import shutil
        ryu_bin = shutil.which("ryu-manager")
        if not ryu_bin:
            cls.process = None
            return

        cls.process = subprocess.Popen(
            [ryu_bin, cls.app_path,
             f"--ofp-tcp-listen-port={cls.CONTROLLER_PORT}",
             "--log-config-file=/dev/null"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)  # Give Ryu time to start

    @classmethod
    def tearDownClass(cls):
        if cls.process and cls.process.poll() is None:
            cls.process.terminate()
            cls.process.wait()
        try:
            os.remove("/tmp/lafs_test_app.py")
        except FileNotFoundError:
            pass

    def _controller_available(self) -> bool:
        return self.__class__.process is not None

    def test_ryu_manager_in_path(self):
        import shutil
        found = shutil.which("ryu-manager") is not None
        self.assertTrue(found,
                        "ryu-manager not found in PATH — check Ryu installation")

    def test_controller_process_running(self):
        if not self._controller_available():
            self.skipTest("ryu-manager not in PATH")
        self.assertIsNone(
            self.process.poll(),
            f"ryu-manager process died. stderr: {self.process.stderr.read(200).decode()}"
        )

    def test_controller_port_open(self):
        """Check that the controller is listening on the expected port."""
        if not self._controller_available():
            self.skipTest("ryu-manager not in PATH")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("127.0.0.1", self.CONTROLLER_PORT))
        sock.close()
        self.assertEqual(result, 0,
                         f"Could not connect to Ryu on port {self.CONTROLLER_PORT}")


# =============================================================================
# Test 4 — Ryu packet library
# =============================================================================
@_RYU_SKIP
class TestRyuPacketLibrary(unittest.TestCase):
    """Verify packet parsing and construction works correctly."""

    def test_ethernet_packet_build(self):
        from ryu.lib.packet import packet, ethernet, ipv4, tcp
        from ryu.lib.packet import ether_types

        pkt = packet.Packet()
        eth = ethernet.ethernet(
            dst="ff:ff:ff:ff:ff:ff",
            src="00:00:00:00:00:01",
            ethertype=ether_types.ETH_TYPE_IP,
        )
        ip = ipv4.ipv4(
            src="10.0.0.1",
            dst="10.0.0.2",
            proto=6,
        )
        t = tcp.tcp(src_port=12345, dst_port=80)

        pkt.add_protocol(eth)
        pkt.add_protocol(ip)
        pkt.add_protocol(t)
        pkt.serialize()

        self.assertTrue(len(pkt.data) > 0, "Packet serialisation failed")

    def test_packet_parse(self):
        from ryu.lib.packet import packet, ethernet

        # Minimal raw Ethernet frame (14 bytes header)
        raw = bytes(14)
        pkt = packet.Packet(raw)
        self.assertIsNotNone(pkt)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    verbosity = 2 if "--verbose" in sys.argv or "-v" in sys.argv else 1
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestRyuImports, TestOpenFlow13Messages,
                TestRyuControllerProcess, TestRyuPacketLibrary]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
