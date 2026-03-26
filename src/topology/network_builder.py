"""
LAFS Project — Network Builder
================================
COMP-6910 — Group 7

Provides :class:`NetworkBuilder`: a high-level wrapper that combines
:class:`~src.topology.fattree.FatTreeTopo` with a Ryu remote controller
to create, start, verify, and tear down LAFS experiment networks.

Responsibilities
----------------
  * Instantiate a Mininet network from a :class:`FatTreeTopo`.
  * Connect all OVS switches to a Ryu remote controller on a configurable
    address/port.
  * Force OpenFlow 1.3 on every switch.
  * Optionally set per-link traffic control parameters (bandwidth, delay, loss).
  * Provide testing utilities: ping-all, iperf pair, flow-rule dump, link-
    utilisation snapshot, and performance scaling tests.
  * Cleanly shut down Mininet and kill any spawned processes.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

from src.topology.fattree import (
    FatTreeGraph,
    FatTreeTopo,
    agg_name,
    core_name,
    edge_name,
    host_ip,
    host_name,
)

logger = logging.getLogger(__name__)

# ── Mininet / Linux guard ─────────────────────────────────────────────────────
_IS_LINUX = sys.platform == "linux"
_IS_ROOT  = os.geteuid() == 0 if _IS_LINUX else False

if _IS_LINUX:
    try:
        from mininet.net import Mininet
        from mininet.node import OVSSwitch, RemoteController, Controller
        from mininet.link import TCLink
        from mininet.log import setLogLevel
        from mininet.clean import cleanup
        _MININET_AVAILABLE = True
    except ImportError:
        _MININET_AVAILABLE = False
        logger.warning("Mininet not installed — NetworkBuilder in stub mode.")
else:
    _MININET_AVAILABLE = False
    logger.info(
        "Non-Linux platform (%s) — NetworkBuilder in stub mode.", sys.platform
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ControllerConfig:
    """Configuration for the Ryu remote controller.

    Attributes:
        host:     Controller IP address or hostname.
        port:     OpenFlow TCP port (default 6633).
        timeout:  Seconds to wait for switches to connect (default 30).
    """
    host:    str = "127.0.0.1"
    port:    int = 6633
    timeout: int = 30


@dataclass
class NetworkConfig:
    """Full configuration for a LAFS experiment network.

    Attributes:
        k:          Fat-tree k parameter (must be even).
        bw_host:    Host ↔ Edge bandwidth in Gbps.
        bw_core:    Fabric link bandwidth in Gbps.
        delay:      Link propagation delay string (e.g. ``'0.1ms'``).
        loss:       Link loss percentage.
        controller: :class:`ControllerConfig` instance.
        autoarp:    If True, pre-populate ARP tables (faster experiments).
        verbose:    If True, set Mininet log level to 'info'.
    """
    k:          int              = 8
    bw_host:    float            = 1.0
    bw_core:    float            = 10.0
    delay:      str              = "0.1ms"
    loss:       int              = 0
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    autoarp:    bool             = True
    verbose:    bool             = False


@dataclass
class LinkStats:
    """Snapshot of link statistics collected from OVS.

    Attributes:
        src:           Source node name.
        dst:           Destination node name.
        tx_bytes:      Bytes transmitted.
        rx_bytes:      Bytes received.
        tx_packets:    Packets transmitted.
        rx_packets:    Packets received.
        tx_errors:     Transmit error count.
        rx_errors:     Receive error count.
        utilisation:   Estimated utilisation fraction (0–1).
        bw_gbps:       Link capacity in Gbps.
    """
    src:         str
    dst:         str
    tx_bytes:    int   = 0
    rx_bytes:    int   = 0
    tx_packets:  int   = 0
    rx_packets:  int   = 0
    tx_errors:   int   = 0
    rx_errors:   int   = 0
    utilisation: float = 0.0
    bw_gbps:     float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  NetworkBuilder
# ─────────────────────────────────────────────────────────────────────────────
class NetworkBuilder:
    """High-level interface for building and managing LAFS experiment networks.

    Combines :class:`FatTreeTopo` with a Mininet :class:`~mininet.net.Mininet`
    instance and a Ryu :class:`~mininet.node.RemoteController`.

    On non-Linux platforms the builder operates in **stub mode**: all topology
    graph methods work correctly but Mininet methods raise
    :exc:`RuntimeError`.

    Args:
        config: :class:`NetworkConfig` instance.  If omitted, defaults are used.

    Example::

        cfg = NetworkConfig(k=8, controller=ControllerConfig(host="127.0.0.1"))
        with NetworkBuilder.managed(cfg) as nb:
            nb.pingall()
            stats = nb.collect_link_stats()

    Attributes:
        config  (NetworkConfig): Active configuration.
        topo    (FatTreeTopo):   Topology instance.
        graph   (FatTreeGraph):  Graph representation for path computation.
        net     (Mininet | None): Running Mininet instance, or None before start.
    """

    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        self.config = config or NetworkConfig()
        self.topo   = FatTreeTopo(
            k       = self.config.k,
            bw_host = self.config.bw_host,
            bw_core = self.config.bw_core,
            delay   = self.config.delay,
            loss    = self.config.loss,
        )
        self.graph: FatTreeGraph = self.topo.graph
        self.net:   Optional[object] = None  # Mininet instance after start()
        self._started = False

        logger.info(
            "NetworkBuilder initialised: k=%d, controller=%s:%d",
            self.config.k,
            self.config.controller.host,
            self.config.controller.port,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Build and start the Mininet network.

        Raises:
            RuntimeError: If not running on Linux with root privileges.
            RuntimeError: If Mininet is not installed.
            TimeoutError: If switches do not connect to the controller within
                          ``config.controller.timeout`` seconds.
        """
        self._require_linux_root()

        if self._started:
            logger.warning("Network already started — call stop() first.")
            return

        if self.config.verbose:
            setLogLevel("info")
        else:
            setLogLevel("warning")

        # Clean up any stale Mininet state
        logger.info("Cleaning up stale Mininet state…")
        cleanup()

        ctrl_cfg = self.config.controller
        logger.info(
            "Starting Mininet k=%d, controller=%s:%d…",
            self.config.k, ctrl_cfg.host, ctrl_cfg.port,
        )

        self.net = Mininet(
            topo=self.topo,
            switch=OVSSwitch,
            controller=None,       # We add manually for precise config
            autoSetMacs=True,
            autoStaticArp=self.config.autoarp,
            waitConnected=False,   # We poll ourselves for better error reporting
        )

        # Add Ryu remote controller
        self.net.addController(
            "ryu_ctrl",
            controller=RemoteController,
            ip=ctrl_cfg.host,
            port=ctrl_cfg.port,
        )

        self.net.start()

        # Force OpenFlow 1.3 on all switches after start
        self._force_openflow13()

        # Wait for all switches to connect
        self._wait_for_controller(ctrl_cfg.timeout)

        self._started = True
        logger.info("Network started successfully.")

    def stop(self) -> None:
        """Stop the Mininet network and clean up resources.

        Safe to call even if the network was never started.
        """
        if self.net is not None and self._started:
            logger.info("Stopping Mininet network…")
            self.net.stop()
            cleanup()
            self._started = False
            logger.info("Network stopped.")
        else:
            logger.debug("stop() called but network not running — no-op.")

    @classmethod
    @contextmanager
    def managed(
        cls, config: Optional[NetworkConfig] = None
    ) -> Generator["NetworkBuilder", None, None]:
        """Context manager that starts and stops the network automatically.

        Args:
            config: Optional :class:`NetworkConfig`.

        Yields:
            A started :class:`NetworkBuilder` instance.

        Example::

            with NetworkBuilder.managed(NetworkConfig(k=4)) as nb:
                loss = nb.pingall()
                print(f"Packet loss: {loss}%")
        """
        nb = cls(config)
        try:
            nb.start()
            yield nb
        finally:
            nb.stop()

    # ── Switch configuration ──────────────────────────────────────────────────
    def _force_openflow13(self) -> None:
        """Set OpenFlow 1.3 protocol on every OVS switch."""
        self._require_started()
        for sw in self.net.switches:
            try:
                sw.cmd(f"ovs-vsctl set bridge {sw.name} protocols=OpenFlow13")
                logger.debug("OF1.3 set on %s", sw.name)
            except Exception as exc:
                logger.warning("Failed to set OF1.3 on %s: %s", sw.name, exc)

    def _wait_for_controller(self, timeout: int) -> None:
        """Block until all switches report ``is_connected`` or timeout.

        Args:
            timeout: Maximum seconds to wait.

        Raises:
            TimeoutError: If switches do not connect within timeout.
        """
        deadline = time.time() + timeout
        n_switches = len(self.net.switches)

        logger.info(
            "Waiting for %d switches to connect (timeout=%ds)…",
            n_switches, timeout,
        )

        while time.time() < deadline:
            connected = sum(
                1 for sw in self.net.switches
                if self._switch_connected(sw)
            )
            if connected == n_switches:
                logger.info("All %d switches connected.", n_switches)
                return
            logger.debug("%d/%d switches connected…", connected, n_switches)
            time.sleep(1)

        # Partial connection — warn but do not abort (some tests may still work)
        connected = sum(
            1 for sw in self.net.switches
            if self._switch_connected(sw)
        )
        if connected < n_switches:
            logger.warning(
                "Timeout: only %d/%d switches connected after %ds.",
                connected, n_switches, timeout,
            )

    def _switch_connected(self, sw) -> bool:
        """Check whether a switch is connected to its controller.

        Args:
            sw: Mininet OVSSwitch instance.

        Returns:
            True if connected.
        """
        try:
            result = sw.cmd(
                f"ovs-vsctl get controller {sw.name} is_connected"
            )
            return result.strip() == "true"
        except Exception:
            return False

    def configure_switch(
        self,
        sw_name: str,
        fail_mode: str = "secure",
        protocols: str = "OpenFlow13",
    ) -> None:
        """Reconfigure a single switch at runtime.

        Args:
            sw_name:   Switch name (e.g. ``'c_0_0'``).
            fail_mode: OVS fail-mode (``'secure'`` or ``'standalone'``).
            protocols: OpenFlow protocol string.

        Raises:
            KeyError: If sw_name not found.
        """
        self._require_started()
        sw = self.net.get(sw_name)
        if sw is None:
            raise KeyError(f"Switch '{sw_name}' not found in Mininet network")
        sw.cmd(f"ovs-vsctl set bridge {sw_name} protocols={protocols}")
        sw.cmd(f"ovs-vsctl set bridge {sw_name} fail_mode={fail_mode}")
        logger.info("Switch %s reconfigured: %s, %s", sw_name, protocols, fail_mode)

    # ── Testing utilities ─────────────────────────────────────────────────────
    def pingall(self, timeout: float = 3.0) -> float:
        """Run all-pairs ping and return packet loss percentage.

        Args:
            timeout: Seconds to wait for each ping response.

        Returns:
            Packet loss as a percentage (0.0 = perfect, 100.0 = all failed).
        """
        self._require_started()
        logger.info("Running pingAll (timeout=%.1fs)…", timeout)
        loss = self.net.pingAll(timeout=timeout)
        logger.info("PingAll complete — packet loss: %.1f%%", loss)
        return loss

    def ping_pair(
        self,
        src_name: str,
        dst_name: str,
        count: int = 5,
        timeout: float = 2.0,
    ) -> Tuple[int, int, float]:
        """Ping from one host to another and return statistics.

        Args:
            src_name: Source host name (e.g. ``'h_0_0_0'``).
            dst_name: Destination host name (e.g. ``'h_1_0_0'``).
            count:    Number of ICMP packets.
            timeout:  Per-packet timeout in seconds.

        Returns:
            Tuple ``(sent, received, rtt_avg_ms)``.

        Raises:
            KeyError: If either host not found.
        """
        self._require_started()
        src = self.net.get(src_name)
        dst = self.net.get(dst_name)
        if src is None:
            raise KeyError(f"Host '{src_name}' not found")
        if dst is None:
            raise KeyError(f"Host '{dst_name}' not found")

        dst_ip = self.graph.get_host_ip(dst_name)
        result = src.cmd(
            f"ping -c {count} -W {int(timeout)} {dst_ip}"
        )

        # Parse: "5 packets transmitted, 5 received, 0% packet loss, time ..."
        sent, rcvd, rtt = 0, 0, 0.0
        m = re.search(r"(\d+) packets transmitted, (\d+) received", result)
        if m:
            sent, rcvd = int(m.group(1)), int(m.group(2))
        m = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", result)
        if m:
            rtt = float(m.group(1))

        logger.info(
            "ping %s → %s: %d/%d received, avg RTT %.2fms",
            src_name, dst_name, rcvd, sent, rtt,
        )
        return sent, rcvd, rtt

    def iperf_pair(
        self,
        server_name: str,
        client_name: str,
        duration: int = 5,
        protocol: str = "tcp",
    ) -> Dict[str, float]:
        """Run an iperf3 throughput test between two hosts.

        Args:
            server_name: Host that runs iperf3 server.
            client_name: Host that runs iperf3 client.
            duration:    Test duration in seconds.
            protocol:    ``'tcp'`` or ``'udp'``.

        Returns:
            Dict with keys ``'throughput_mbps'``, ``'retransmits'``
            (TCP only), ``'jitter_ms'`` (UDP only).
        """
        self._require_started()
        server = self.net.get(server_name)
        client = self.net.get(client_name)
        if server is None or client is None:
            raise KeyError(f"Host not found: {server_name} or {client_name}")

        server_ip = self.graph.get_host_ip(server_name)

        # Start iperf3 server in background
        server.cmd(f"iperf3 -s -D --logfile /tmp/iperf3_{server_name}.log")
        time.sleep(0.5)

        # Run iperf3 client
        proto_flag = "-u" if protocol == "udp" else ""
        raw = client.cmd(
            f"iperf3 -c {server_ip} -t {duration} {proto_flag} "
            f"--json 2>/dev/null"
        )

        # Kill server
        server.cmd("pkill -f iperf3")

        result: Dict[str, float] = {
            "throughput_mbps": 0.0,
            "retransmits":     0.0,
            "jitter_ms":       0.0,
        }

        # Parse JSON output
        try:
            import json
            data = json.loads(raw)
            end  = data.get("end", {})
            if protocol == "tcp":
                sent = end.get("sum_sent", {})
                result["throughput_mbps"] = sent.get("bits_per_second", 0) / 1e6
                result["retransmits"]     = float(sent.get("retransmits", 0))
            else:
                recv = end.get("sum", {})
                result["throughput_mbps"] = recv.get("bits_per_second", 0) / 1e6
                result["jitter_ms"]       = recv.get("jitter_ms", 0.0)
        except Exception as exc:
            logger.warning("iperf3 JSON parse failed: %s", exc)

        logger.info(
            "iperf3 %s → %s (%s, %ds): %.1f Mbps",
            client_name, server_name, protocol, duration,
            result["throughput_mbps"],
        )
        return result

    # ── Link utilisation monitoring ───────────────────────────────────────────
    def collect_link_stats(self) -> List[LinkStats]:
        """Collect per-port byte/packet counters from every switch.

        Queries ``ovs-ofctl dump-ports`` on each switch and parses the output.

        Returns:
            List of :class:`LinkStats` objects, one per switch port.
        """
        self._require_started()
        stats: List[LinkStats] = []

        for sw in self.net.switches:
            raw = sw.cmd(
                f"ovs-ofctl -O OpenFlow13 dump-ports {sw.name}"
            )
            port_blocks = re.split(r"port\s+", raw)[1:]  # skip header

            for block in port_blocks:
                # Extract tx/rx bytes
                tx_m = re.search(r"tx.*?bytes=(\d+)", block)
                rx_m = re.search(r"rx.*?bytes=(\d+)", block)
                tx_bytes = int(tx_m.group(1)) if tx_m else 0
                rx_bytes = int(rx_m.group(1)) if rx_m else 0

                tx_pkt_m = re.search(r"tx.*?pkts=(\d+)", block)
                rx_pkt_m = re.search(r"rx.*?pkts=(\d+)", block)
                tx_pkts  = int(tx_pkt_m.group(1)) if tx_pkt_m else 0
                rx_pkts  = int(rx_pkt_m.group(1)) if rx_pkt_m else 0

                tx_err_m = re.search(r"tx.*?errs=(\d+)", block)
                rx_err_m = re.search(r"rx.*?errs=(\d+)", block)
                tx_errs  = int(tx_err_m.group(1)) if tx_err_m else 0
                rx_errs  = int(rx_err_m.group(1)) if rx_err_m else 0

                stats.append(LinkStats(
                    src=sw.name,
                    dst="",         # port-level — dst resolved by controller
                    tx_bytes=tx_bytes,
                    rx_bytes=rx_bytes,
                    tx_packets=tx_pkts,
                    rx_packets=rx_pkts,
                    tx_errors=tx_errs,
                    rx_errors=rx_errs,
                ))

        logger.debug("Collected link stats: %d port entries", len(stats))
        return stats

    def dump_flows(self, sw_name: str) -> str:
        """Dump OpenFlow flow table for a switch.

        Args:
            sw_name: Switch name (e.g. ``'c_0_0'``).

        Returns:
            Raw string output from ``ovs-ofctl dump-flows``.
        """
        self._require_started()
        sw = self.net.get(sw_name)
        if sw is None:
            raise KeyError(f"Switch '{sw_name}' not found")
        return sw.cmd(f"ovs-ofctl -O OpenFlow13 dump-flows {sw_name}")

    def dump_all_flows(self) -> Dict[str, str]:
        """Dump flow tables for all switches.

        Returns:
            Dict mapping switch name to its flow-table dump string.
        """
        self._require_started()
        return {
            sw.name: self.dump_flows(sw.name)
            for sw in self.net.switches
        }

    def count_flow_rules(self) -> Dict[str, int]:
        """Count installed flow rules per switch.

        Returns:
            Dict mapping switch name to the number of flow entries.
        """
        self._require_started()
        counts = {}
        for sw in self.net.switches:
            raw = self.dump_flows(sw.name)
            counts[sw.name] = raw.count("cookie=")
        return counts

    # ── Performance & scalability tests ──────────────────────────────────────
    def test_controller_reachable(self) -> bool:
        """Check TCP connectivity to the configured Ryu controller.

        Returns:
            True if the controller port is open and accepting connections.
        """
        ctrl = self.config.controller
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((ctrl.host, ctrl.port))
            sock.close()
            reachable = result == 0
            logger.info(
                "Controller %s:%d — %s",
                ctrl.host, ctrl.port,
                "reachable" if reachable else "UNREACHABLE",
            )
            return reachable
        except Exception as exc:
            logger.error("Controller reachability check failed: %s", exc)
            return False

    def test_all_switches_connected(self) -> Tuple[int, int]:
        """Check how many switches are connected to the controller.

        Returns:
            Tuple ``(connected, total)`` switch counts.
        """
        self._require_started()
        total = len(self.net.switches)
        connected = sum(
            1 for sw in self.net.switches
            if self._switch_connected(sw)
        )
        logger.info(
            "Switch connectivity: %d/%d connected", connected, total
        )
        return connected, total

    def performance_test_128hosts(
        self,
        sample_pairs: int = 10,
        iperf_duration: int = 3,
    ) -> Dict:
        """Run a basic performance test across a sample of host pairs.

        Selects ``sample_pairs`` random host pairs and measures iperf3
        throughput.  Does NOT run all 128×127 pairs (too slow for a test).

        Args:
            sample_pairs:   Number of (server, client) host pairs to test.
            iperf_duration: iperf3 duration per pair in seconds.

        Returns:
            Dict with keys:
              ``'pairs_tested'``, ``'avg_throughput_mbps'``,
              ``'min_throughput_mbps'``, ``'max_throughput_mbps'``,
              ``'total_duration_s'``.
        """
        self._require_started()
        import random

        hosts = [h.name for h in self.net.hosts]
        if len(hosts) < 2:
            raise RuntimeError("Need at least 2 hosts for performance test")

        pairs = []
        random.seed(42)
        shuffled = random.sample(hosts, min(sample_pairs * 2, len(hosts)))
        for i in range(0, min(len(shuffled) - 1, sample_pairs * 2), 2):
            pairs.append((shuffled[i], shuffled[i + 1]))

        throughputs = []
        t_start = time.time()

        for srv, cli in pairs:
            try:
                res = self.iperf_pair(srv, cli, duration=iperf_duration)
                throughputs.append(res["throughput_mbps"])
            except Exception as exc:
                logger.warning("iperf3 failed for %s → %s: %s", cli, srv, exc)
                throughputs.append(0.0)

        elapsed = time.time() - t_start

        result = {
            "pairs_tested":         len(pairs),
            "avg_throughput_mbps":  sum(throughputs) / max(len(throughputs), 1),
            "min_throughput_mbps":  min(throughputs) if throughputs else 0,
            "max_throughput_mbps":  max(throughputs) if throughputs else 0,
            "total_duration_s":     elapsed,
        }

        logger.info(
            "Performance test: %d pairs, avg=%.1f Mbps, "
            "min=%.1f Mbps, max=%.1f Mbps",
            result["pairs_tested"],
            result["avg_throughput_mbps"],
            result["min_throughput_mbps"],
            result["max_throughput_mbps"],
        )
        return result

    # ── Node accessors (delegating to Mininet) ────────────────────────────────
    def get_host(self, name: str):
        """Return the Mininet host object for the given name.

        Args:
            name: Host node name.

        Returns:
            Mininet Host object.

        Raises:
            KeyError: If host not found.
        """
        self._require_started()
        h = self.net.get(name)
        if h is None:
            raise KeyError(f"Host '{name}' not found in running network")
        return h

    def get_switch(self, name: str):
        """Return the Mininet switch object for the given name.

        Args:
            name: Switch node name.

        Returns:
            Mininet OVSSwitch object.

        Raises:
            KeyError: If switch not found.
        """
        self._require_started()
        sw = self.net.get(name)
        if sw is None:
            raise KeyError(f"Switch '{name}' not found in running network")
        return sw

    def host_cmd(self, host_name_: str, cmd: str) -> str:
        """Run a shell command on a Mininet host.

        Args:
            host_name_: Host node name.
            cmd:        Shell command string.

        Returns:
            Command stdout string.
        """
        self._require_started()
        return self.get_host(host_name_).cmd(cmd)

    # ── Internal guards ───────────────────────────────────────────────────────
    def _require_linux_root(self) -> None:
        """Raise RuntimeError if not on Linux with root privileges."""
        if not _IS_LINUX:
            raise RuntimeError(
                f"Mininet requires Linux. Current platform: {sys.platform}"
            )
        if not _IS_ROOT:
            raise RuntimeError(
                "Mininet requires root. Run with: sudo python …"
            )
        if not _MININET_AVAILABLE:
            raise RuntimeError(
                "Mininet is not installed. Run: sudo ./setup/install_mininet.sh"
            )

    def _require_started(self) -> None:
        """Raise RuntimeError if the network has not been started."""
        if not self._started or self.net is None:
            raise RuntimeError(
                "Network not started — call NetworkBuilder.start() first."
            )

    # ── Repr ─────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        status = "running" if self._started else "stopped"
        return (
            f"NetworkBuilder(k={self.config.k}, "
            f"ctrl={self.config.controller.host}:{self.config.controller.port}, "
            f"status={status})"
        )
