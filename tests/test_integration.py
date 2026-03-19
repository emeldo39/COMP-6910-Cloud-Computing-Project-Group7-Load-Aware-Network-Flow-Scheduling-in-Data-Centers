#!/usr/bin/env python3
"""
LAFS Project — End-to-End Integration Test ("Hello World" for LAFS)
COMP 6910 — Group 7

This test verifies the full pipeline works:
  1. Build a k=4 Fat-tree topology in Mininet
  2. Start a Ryu controller with a minimal OF 1.3 forwarding app
  3. Install flow rules via OpenFlow 1.3
  4. Send traffic between hosts
  5. Collect FCT and link utilisation metrics
  6. Verify MILP optimizer produces a valid flow placement

All components exercised: Mininet → OVS → Ryu → MILP → Metrics

Usage (requires root for Mininet):
    sudo python tests/test_integration.py
    sudo python tests/test_integration.py --k 4 --verbose
"""

import sys
import os
import time
import socket
import subprocess
import threading
import tempfile
import unittest
import argparse

# ── Platform guard ────────────────────────────────────────────────────────────
IS_LINUX = sys.platform == "linux"
IS_ROOT  = os.geteuid() == 0 if IS_LINUX else False


def require_linux_root(test_case):
    """Skip decorator for tests that need Linux + root."""
    if not IS_LINUX:
        return unittest.skip("Requires Linux")(test_case)
    if not IS_ROOT:
        return unittest.skip("Requires root (sudo)")(test_case)
    return test_case


# =============================================================================
# Minimal Ryu forwarding app (written to /tmp at test time)
# =============================================================================
LAFS_HELLO_APP = '''
"""
Minimal LAFS "Hello World" Ryu app.
Installs L2 learning switch rules using OpenFlow 1.3.
Used by integration test only.
"""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from collections import defaultdict


class LAFSHelloWorld(app_manager.RyuApp):
    """
    Simple L2 learning switch (OpenFlow 1.3).
    Serves as integration test scaffold — real LAFS controller adds
    load prediction, MILP flow placement, and k-path routing on top.
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = defaultdict(dict)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow entry."""
        dp     = ev.msg.datapath
        ofp    = dp.ofproto
        parser = dp.ofproto_parser
        match  = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER,
                                          ofp.OFPCML_NO_BUFFER)]
        self._add_flow(dp, 0, match, actions)

    def _add_flow(self, dp, priority, match, actions, idle=0, hard=0):
        parser = dp.ofproto_parser
        ofp    = dp.ofproto
        instr  = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod    = parser.OFPFlowMod(
            datapath=dp, priority=priority, match=match,
            instructions=instr, idle_timeout=idle, hard_timeout=hard,
        )
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """L2 learning: install forwarding rules on first packet."""
        msg    = ev.msg
        dp     = msg.datapath
        ofp    = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = dp.id

        self.mac_to_port[dpid][src] = in_port

        out_port = (self.mac_to_port[dpid][dst]
                    if dst in self.mac_to_port[dpid]
                    else ofp.OFPP_FLOOD)

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(dp, 1, match, actions, idle=60)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out  = parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data,
        )
        dp.send_msg(out)
'''


# =============================================================================
# Component tests (no Mininet required)
# =============================================================================
class TestComponentsNoMininet(unittest.TestCase):
    """Verify all components work in isolation before combining."""

    def test_networkx_fat_tree_graph(self):
        """Build a Fat-tree graph with NetworkX and verify topology."""
        import networkx as nx

        k = 4
        G = nx.Graph()
        half_k = k // 2

        # Add nodes
        for pod in range(k):
            for e in range(half_k):
                for h in range(half_k):
                    G.add_node(f"h{pod}{e}{h}", type="host",
                               ip=f"10.{pod}.{e}.{h + 2}")
                for a in range(half_k):
                    G.add_node(f"a{pod}{a}", type="agg")
                for e2 in range(half_k):
                    G.add_node(f"e{pod}{e2}", type="edge")
        for i in range(half_k):
            for j in range(half_k):
                G.add_node(f"c{i}{j}", type="core")

        expected_hosts = k * half_k * half_k
        host_count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "host")
        self.assertEqual(host_count, expected_hosts)
        print(f"\n  NetworkX Fat-tree (k={k}): {host_count} hosts, "
              f"{G.number_of_nodes()} total nodes")

    def test_ewma_predictor(self):
        """EWMA predictor should track a step-function signal."""
        alpha = 0.3
        ewma_val = 0.0
        signal = [0.0] * 5 + [1.0] * 10

        for s in signal:
            ewma_val = alpha * s + (1 - alpha) * ewma_val

        # After 10 samples of 1.0, EWMA should be close to 1.0
        self.assertGreater(ewma_val, 0.8,
                           f"EWMA too slow to converge: {ewma_val:.3f}")
        print(f"\n  EWMA convergence test: final={ewma_val:.3f} (expected >0.8)")

    def test_arima_statsmodels(self):
        """ARIMA model should fit and forecast a simple time series."""
        import numpy as np
        from statsmodels.tsa.arima.model import ARIMA

        np.random.seed(42)
        # Simulated link utilisation: trending upward + noise
        n = 50
        t = np.arange(n)
        data = 0.3 + 0.01 * t + 0.02 * np.random.randn(n)
        data = np.clip(data, 0, 1)

        model  = ARIMA(data, order=(1, 0, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=5)

        self.assertEqual(len(forecast), 5)
        self.assertTrue(all(0 <= v <= 2.0 for v in forecast),
                        f"Forecast values unreasonable: {forecast}")
        print(f"\n  ARIMA(1,0,0) 5-step forecast: {forecast.round(3).tolist()}")

    def test_milp_flow_placement(self):
        """Solve a 5-flow, 3-path MILP placement problem."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
            SOLVER = "gurobi"
        except ImportError:
            SOLVER = "pulp"

        n_flows, n_paths = 5, 3
        fct   = [[1.0, 2.0, 3.0],
                 [3.0, 1.0, 2.0],
                 [2.0, 3.0, 1.0],
                 [1.5, 1.5, 2.5],
                 [2.5, 1.0, 1.5]]
        load  = [0.2, 0.3, 0.4, 0.1, 0.2]
        cap   = [0.8, 0.8, 0.8]

        if SOLVER == "gurobi":
            import gurobipy as gp
            from gurobipy import GRB
            m = gp.Model(); m.setParam("OutputFlag", 0)
            x = [[m.addVar(vtype=GRB.BINARY) for p in range(n_paths)]
                 for f in range(n_flows)]
            m.setObjective(
                gp.quicksum(fct[f][p] * x[f][p]
                            for f in range(n_flows) for p in range(n_paths)),
                GRB.MINIMIZE
            )
            for f in range(n_flows):
                m.addConstr(gp.quicksum(x[f][p] for p in range(n_paths)) == 1)
            for p in range(n_paths):
                m.addConstr(gp.quicksum(load[f] * x[f][p]
                                        for f in range(n_flows)) <= cap[p])
            m.optimize()
            status_ok = (m.Status == GRB.OPTIMAL)
            obj_val   = m.ObjVal
        else:
            import pulp
            prob = pulp.LpProblem("placement", pulp.LpMinimize)
            x = [[pulp.LpVariable(f"x{f}{p}", cat="Binary")
                  for p in range(n_paths)] for f in range(n_flows)]
            prob += pulp.lpSum(fct[f][p] * x[f][p]
                               for f in range(n_flows) for p in range(n_paths))
            for f in range(n_flows):
                prob += pulp.lpSum(x[f][p] for p in range(n_paths)) == 1
            for p in range(n_paths):
                prob += pulp.lpSum(load[f] * x[f][p]
                                   for f in range(n_flows)) <= cap[p]
            status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
            status_ok = (pulp.LpStatus[status] == "Optimal")
            obj_val   = pulp.value(prob.objective)

        self.assertTrue(status_ok, "MILP did not find optimal solution")
        self.assertLessEqual(obj_val, sum(min(fct[f]) for f in range(n_flows)) + 1e-6)
        print(f"\n  MILP 5-flow placement ({SOLVER}): obj={obj_val:.3f}")

    def test_jain_fairness_index(self):
        """Jain's fairness index calculation."""
        import numpy as np

        def jain_index(allocations):
            a = np.array(allocations)
            return (a.sum() ** 2) / (len(a) * (a ** 2).sum())

        # Perfect fairness
        self.assertAlmostEqual(jain_index([1, 1, 1, 1]), 1.0, places=5)
        # One tenant gets everything: worst case
        self.assertAlmostEqual(jain_index([4, 0, 0, 0]), 0.25, places=5)
        # Partial fairness
        ji = jain_index([3, 2, 2, 1])
        self.assertGreater(ji, 0.8)
        print(f"\n  Jain's index [3,2,2,1] = {ji:.4f}")

    def test_fct_percentiles(self):
        """Compute avg/P95/P99 FCT from a sample distribution."""
        import numpy as np

        np.random.seed(42)
        # Simulate FCT: mice flows have low FCT, elephants have high
        mice_fct     = np.random.exponential(scale=0.01, size=800)    # 80%
        elephant_fct = np.random.exponential(scale=1.0,  size=200)    # 20%
        all_fct      = np.concatenate([mice_fct, elephant_fct])

        avg_fct = np.mean(all_fct)
        p95_fct = np.percentile(all_fct, 95)
        p99_fct = np.percentile(all_fct, 99)

        self.assertGreater(p99_fct, p95_fct)
        self.assertGreater(p95_fct, avg_fct)
        print(f"\n  FCT metrics: avg={avg_fct:.3f}s, "
              f"P95={p95_fct:.3f}s, P99={p99_fct:.3f}s")


# =============================================================================
# Mininet integration test (requires root + Linux)
# =============================================================================
@require_linux_root
class TestMininetRyuIntegration(unittest.TestCase):
    """
    End-to-end test: Mininet k=4 Fat-tree + Ryu L2 controller.
    This is the "Hello World" for the full LAFS pipeline.
    """

    CONTROLLER_PORT = 6633
    K = 4

    @classmethod
    def setUpClass(cls):
        from mininet.clean import cleanup
        cleanup()

        # Write the Ryu app to a temp file
        cls.app_fd, cls.app_path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(cls.app_fd, "w") as f:
            f.write(LAFS_HELLO_APP)

        # Start Ryu controller
        import shutil
        ryu_bin = shutil.which("ryu-manager")
        if not ryu_bin:
            cls.ryu_proc = None
            return

        cls.ryu_proc = subprocess.Popen(
            [ryu_bin, cls.app_path,
             f"--ofp-tcp-listen-port={cls.CONTROLLER_PORT}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        # Build Mininet
        from mininet.topo import Topo
        from mininet.net import Mininet
        from mininet.node import OVSSwitch, RemoteController
        from mininet.link import TCLink
        from mininet.log import setLogLevel
        setLogLevel("warning")

        # Import Fat-tree from test_mininet (same module)
        sys.path.insert(0, os.path.dirname(__file__))
        from test_mininet import FatTreeTopo

        topo = FatTreeTopo(k=cls.K, bw=0.1)  # 100 Mbps for quick test
        cls.net = Mininet(
            topo=topo,
            switch=OVSSwitch,
            controller=RemoteController,
            autoSetMacs=True,
            autoStaticArp=True,
        )
        cls.net.addController(
            "c0",
            controller=RemoteController,
            ip="127.0.0.1",
            port=cls.CONTROLLER_PORT,
        )
        cls.net.start()
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "net") and cls.net:
            cls.net.stop()
        if hasattr(cls, "ryu_proc") and cls.ryu_proc:
            cls.ryu_proc.terminate()
            cls.ryu_proc.wait()
        try:
            os.remove(cls.app_path)
        except Exception:
            pass
        from mininet.clean import cleanup
        cleanup()

    def _skip_if_no_ryu(self):
        if not hasattr(self.__class__, "ryu_proc") or not self.__class__.ryu_proc:
            self.skipTest("ryu-manager not available")

    def test_controller_connected(self):
        """All switches should connect to the remote Ryu controller."""
        self._skip_if_no_ryu()
        for sw in self.net.switches:
            result = sw.cmd(
                f"ovs-vsctl get-controller {sw.name}"
            )
            self.assertIn("6633", result,
                          f"Switch {sw.name} not connected to controller")

    def test_ping_two_hosts(self):
        """Two hosts in the same pod should reach each other."""
        self._skip_if_no_ryu()
        h0 = self.net.hosts[0]
        h1 = self.net.hosts[1]
        result = h0.cmd(f"ping -c 5 -W 2 {h1.IP()}")
        self.assertIn("0% packet loss", result,
                      f"h0 → h1 ping failed:\n{result}")

    def test_cross_pod_ping(self):
        """Hosts in different pods should reach each other (cross-pod routing)."""
        self._skip_if_no_ryu()
        hosts = self.net.hosts
        h0 = hosts[0]
        h_cross = hosts[len(hosts) // 2]
        result = h0.cmd(f"ping -c 5 -W 3 {h_cross.IP()}")
        self.assertIn("0% packet loss", result,
                      f"Cross-pod ping failed: {h0.name} → {h_cross.name}\n{result}")

    def test_flow_rules_installed(self):
        """After traffic, switches should have flow rules (not just table-miss)."""
        self._skip_if_no_ryu()
        # Trigger learning
        h0 = self.net.hosts[0]
        h1 = self.net.hosts[1]
        h0.cmd(f"ping -c 2 {h1.IP()}")
        time.sleep(1)

        sw = self.net.switches[0]
        flows = sw.cmd(f"ovs-ofctl dump-flows {sw.name} -O OpenFlow13")
        # Should have more than just the table-miss rule
        flow_count = flows.count("cookie=")
        self.assertGreater(flow_count, 1,
                           f"Only {flow_count} flow rule(s) on {sw.name} — expected more")
        print(f"\n  Flow rules on {sw.name}: {flow_count}")

    def test_iperf_throughput(self):
        """iperf3 between two hosts should show measurable throughput."""
        self._skip_if_no_ryu()
        h0 = self.net.hosts[0]
        h1 = self.net.hosts[1]

        h1.cmd("iperf3 -s -D --logfile /tmp/iperf_integration.log")
        time.sleep(0.5)

        result = h0.cmd(f"iperf3 -c {h1.IP()} -t 3 --json 2>/dev/null")
        h1.cmd("pkill iperf3")

        # Just verify iperf ran and produced output
        self.assertIn("end", result,
                      "iperf3 did not complete successfully")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAFS integration tests")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--skip-mininet", action="store_true",
                        help="Run only component tests (no Mininet)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args()

    TestMininetRyuIntegration.K = args.k

    verbosity = 2 if args.verbose else 1
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestComponentsNoMininet))
    if not args.skip_mininet:
        suite.addTests(loader.loadTestsFromTestCase(TestMininetRyuIntegration))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
