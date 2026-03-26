"""
LAFS Project — Fat-Tree Topology Implementation
================================================
COMP-6910 — Group 7

This module provides two complementary classes:

  FatTreeGraph  — pure-Python / NetworkX representation.
                  No Mininet dependency; safe to import on Windows.
                  Used for: path computation, topology analysis,
                  MILP solver graph input.

  FatTreeTopo   — Mininet Topo subclass.
                  Extends FatTreeGraph to drive actual Mininet nodes,
                  links, and switch configuration.

Fat-Tree Topology (k-ary):
---------------------------
  Pods                 : k
  Core switches        : (k/2)²
  Aggregation switches : k × (k/2)     (k/2 per pod)
  Edge switches        : k × (k/2)     (k/2 per pod)
  Hosts                : k³/4          (k/2 per edge switch)

For k=8 (LAFS target):
  Core       16 switches  (c_0_0 … c_3_3)
  Aggregation 32 switches  (a_0_0 … a_7_3)
  Edge        32 switches  (e_0_0 … e_7_3)
  Hosts      128           (h_0_0_0 … h_7_3_3)
  Total switches: 80

Node Naming Convention:
-----------------------
  Core        c_<row>_<col>         c_0_0, c_3_3
  Aggregation a_<pod>_<idx>         a_0_0, a_7_3
  Edge        e_<pod>_<idx>         e_0_0, e_7_3
  Host        h_<pod>_<edge>_<idx>  h_0_0_0, h_7_3_3

IP Addressing (hosts only):
---------------------------
  10.<pod>.<edge>.<host_idx + 2>
  e.g.  h_0_0_0 → 10.0.0.2
        h_7_3_3 → 10.7.3.5

MAC Addressing:
---------------
  00:00:0a:<pod>:<edge>:<host_idx + 2>
  (mirrors the IP address in hex)

Port Numbering:
---------------
  Edge switch  e_pod_i :
    Port 1 … k/2         → hosts h_pod_i_0 … h_pod_i_(k/2-1)   (downlinks)
    Port k/2+1 … k       → agg   a_pod_0 … a_pod_(k/2-1)        (uplinks)

  Aggregation switch  a_pod_i :
    Port 1 … k/2         → edge  e_pod_0 … e_pod_(k/2-1)        (downlinks)
    Port k/2+1 … k       → core  c_i_0 … c_i_(k/2-1)            (uplinks)

  Core switch  c_i_j :
    Port 1 … k           → agg   a_0_i … a_(k-1)_i  (one per pod, pod order) (downlinks)

Link Bandwidths:
----------------
  Host ↔ Edge          : bw_host  (default 1 Gbps)
  Edge ↔ Aggregation   : bw_agg   (default 10 Gbps)
  Aggregation ↔ Core   : bw_core  (default 10 Gbps)
"""

from __future__ import annotations

import logging
import math
from itertools import product
from typing import Dict, Generator, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_K        = 8
BW_HOST_GBPS     = 1.0    # Host ↔ Edge link bandwidth (Gbps)
BW_FABRIC_GBPS   = 10.0   # Switch-to-switch link bandwidth (Gbps)
LINK_DELAY_MS    = "0.1ms"
LINK_LOSS_PCT    = 0      # 0 % loss on all links


# ─────────────────────────────────────────────────────────────────────────────
#  Naming helpers
# ─────────────────────────────────────────────────────────────────────────────
def core_name(row: int, col: int) -> str:
    """Return the canonical name for a core switch.

    Args:
        row: Core switch row index, 0 ≤ row < k/2.
        col: Core switch column index, 0 ≤ col < k/2.

    Returns:
        Name string, e.g. ``'c_0_0'``.
    """
    return f"c_{row}_{col}"


def agg_name(pod: int, idx: int) -> str:
    """Return the canonical name for an aggregation switch.

    Args:
        pod: Pod number, 0 ≤ pod < k.
        idx: Aggregation switch index within the pod, 0 ≤ idx < k/2.

    Returns:
        Name string, e.g. ``'a_2_1'``.
    """
    return f"a_{pod}_{idx}"


def edge_name(pod: int, idx: int) -> str:
    """Return the canonical name for an edge switch.

    Args:
        pod: Pod number, 0 ≤ pod < k.
        idx: Edge switch index within the pod, 0 ≤ idx < k/2.

    Returns:
        Name string, e.g. ``'e_3_0'``.
    """
    return f"e_{pod}_{idx}"


def host_name(pod: int, edge: int, idx: int) -> str:
    """Return the canonical name for a host.

    Args:
        pod:  Pod number.
        edge: Edge switch index within the pod.
        idx:  Host index under that edge switch, 0 ≤ idx < k/2.

    Returns:
        Name string, e.g. ``'h_0_0_2'``.
    """
    return f"h_{pod}_{edge}_{idx}"


def host_ip(pod: int, edge: int, idx: int) -> str:
    """Compute the IP address for a host.

    Follows the convention ``10.<pod>.<edge>.<idx+2>``.
    Addresses .0 and .1 are reserved (network/gateway).

    Args:
        pod:  Pod number.
        edge: Edge switch index.
        idx:  Host index (0-based within the edge switch).

    Returns:
        Dotted-decimal IP string, e.g. ``'10.0.0.2'``.
    """
    return f"10.{pod}.{edge}.{idx + 2}"


def host_mac(pod: int, edge: int, idx: int) -> str:
    """Compute the MAC address for a host, mirroring its IP.

    Format: ``00:00:0a:<pod_hex>:<edge_hex>:<host_hex>``

    Args:
        pod:  Pod number.
        edge: Edge switch index.
        idx:  Host index.

    Returns:
        Colon-separated MAC string, e.g. ``'00:00:0a:00:00:02'``.
    """
    return f"00:00:0a:{pod:02x}:{edge:02x}:{(idx + 2):02x}"


# ─────────────────────────────────────────────────────────────────────────────
#  FatTreeGraph  (pure NetworkX — no Mininet dependency)
# ─────────────────────────────────────────────────────────────────────────────
class FatTreeGraph:
    """Pure-Python Fat-Tree graph using NetworkX.

    Suitable for path computation, topology analysis, and as input to the
    MILP optimizer without requiring Mininet to be installed.

    Attributes:
        k         (int): Number of ports per switch. Must be even.
        half_k    (int): k // 2.
        graph     (nx.Graph): Undirected graph with nodes and edges.
        n_core    (int): Number of core switches.
        n_agg     (int): Number of aggregation switches.
        n_edge    (int): Number of edge switches.
        n_hosts   (int): Number of host nodes.
        n_switches(int): Total number of switches.

    Example::

        g = FatTreeGraph(k=4)
        paths = g.get_paths("h_0_0_0", "h_1_0_0")
        print(f"{len(paths)} paths found")
    """

    def __init__(self, k: int = DEFAULT_K) -> None:
        """Initialise and build the Fat-Tree graph.

        Args:
            k: Number of ports per switch.  Must be a positive even integer.

        Raises:
            ValueError: If k is not a positive even integer.
        """
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be a positive even integer, got {k}")

        self.k      = k
        self.half_k = k // 2

        # Derived counts
        self.n_core     = self.half_k ** 2
        self.n_agg      = k * self.half_k
        self.n_edge     = k * self.half_k
        self.n_hosts    = (k ** 3) // 4
        self.n_switches = self.n_core + self.n_agg + self.n_edge

        logger.info(
            "FatTreeGraph k=%d: %d core, %d agg, %d edge, %d hosts",
            k, self.n_core, self.n_agg, self.n_edge, self.n_hosts,
        )

        self.graph: nx.Graph = nx.Graph()
        self._ip_to_node:  Dict[str, str] = {}  # IP  → node name
        self._node_to_ip:  Dict[str, str] = {}  # node name → IP
        self._node_to_mac: Dict[str, str] = {}  # node name → MAC

        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        """Construct all nodes and edges in the NetworkX graph."""
        self._add_core_nodes()
        self._add_pod_nodes()
        self._add_host_nodes()
        self._add_core_to_agg_edges()
        self._add_agg_to_edge_edges()
        self._add_edge_to_host_edges()

        logger.debug(
            "Graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _add_core_nodes(self) -> None:
        """Add (k/2)² core switch nodes."""
        for row in range(self.half_k):
            for col in range(self.half_k):
                name = core_name(row, col)
                self.graph.add_node(
                    name,
                    node_type="core",
                    row=row,
                    col=col,
                )

    def _add_pod_nodes(self) -> None:
        """Add k*(k/2) aggregation and k*(k/2) edge switch nodes."""
        for pod in range(self.k):
            for idx in range(self.half_k):
                a = agg_name(pod, idx)
                self.graph.add_node(a, node_type="agg", pod=pod, idx=idx)

                e = edge_name(pod, idx)
                self.graph.add_node(e, node_type="edge", pod=pod, idx=idx)

    def _add_host_nodes(self) -> None:
        """Add k³/4 host nodes with IP and MAC metadata."""
        for pod in range(self.k):
            for e_idx in range(self.half_k):
                for h_idx in range(self.half_k):
                    name = host_name(pod, e_idx, h_idx)
                    ip   = host_ip(pod, e_idx, h_idx)
                    mac  = host_mac(pod, e_idx, h_idx)
                    self.graph.add_node(
                        name,
                        node_type="host",
                        pod=pod,
                        edge=e_idx,
                        host_idx=h_idx,
                        ip=ip,
                        mac=mac,
                    )
                    self._ip_to_node[ip]   = name
                    self._node_to_ip[name] = ip
                    self._node_to_mac[name] = mac

    def _add_core_to_agg_edges(self) -> None:
        """Add edges between core and aggregation switches.

        Core switch c_i_j connects to aggregation switch a_pod_i in every pod.
        The port on the core side is ``pod + 1``.
        The port on the agg side is ``self.half_k + j + 1``.
        """
        for pod in range(self.k):
            for row in range(self.half_k):   # agg index within pod = row
                a = agg_name(pod, row)
                for col in range(self.half_k):
                    c = core_name(row, col)
                    # Core port: pod+1 (ports ordered by pod number)
                    # Agg port : half_k + col + 1  (uplink to col-th core)
                    self.graph.add_edge(
                        c, a,
                        link_type="core-agg",
                        port_core=pod + 1,
                        port_agg=self.half_k + col + 1,
                        bw=BW_FABRIC_GBPS,
                        capacity=BW_FABRIC_GBPS,   # Gbps, used by MILP
                    )

    def _add_agg_to_edge_edges(self) -> None:
        """Add edges between aggregation and edge switches.

        Within a pod, every aggregation switch connects to every edge switch.
        Agg port:  ``e_idx + 1``  (downlink to e_idx-th edge switch).
        Edge port: ``self.half_k + a_idx + 1``  (uplink to a_idx-th agg switch).
        """
        for pod in range(self.k):
            for a_idx in range(self.half_k):
                a = agg_name(pod, a_idx)
                for e_idx in range(self.half_k):
                    e = edge_name(pod, e_idx)
                    self.graph.add_edge(
                        a, e,
                        link_type="agg-edge",
                        port_agg=e_idx + 1,
                        port_edge=self.half_k + a_idx + 1,
                        bw=BW_FABRIC_GBPS,
                        capacity=BW_FABRIC_GBPS,
                    )

    def _add_edge_to_host_edges(self) -> None:
        """Add edges between edge switches and hosts.

        Edge port: ``h_idx + 1``  (downlink to h_idx-th host).
        Host port: ``1``           (single uplink NIC).
        """
        for pod in range(self.k):
            for e_idx in range(self.half_k):
                e = edge_name(pod, e_idx)
                for h_idx in range(self.half_k):
                    h = host_name(pod, e_idx, h_idx)
                    self.graph.add_edge(
                        e, h,
                        link_type="edge-host",
                        port_edge=h_idx + 1,
                        port_host=1,
                        bw=BW_HOST_GBPS,
                        capacity=BW_HOST_GBPS,
                    )

    # ── Path computation ──────────────────────────────────────────────────────
    def get_paths(
        self,
        src: str,
        dst: str,
        max_paths: Optional[int] = None,
    ) -> List[List[str]]:
        """Return all shortest paths between two nodes.

        Uses NetworkX ``all_shortest_paths``.  For a k=8 Fat-tree, there are
        up to ``(k/2)² = 16`` equal-cost paths between hosts in different pods.

        Args:
            src:       Source node name (e.g. ``'h_0_0_0'``) or IP address.
            dst:       Destination node name or IP address.
            max_paths: If set, return at most this many paths.

        Returns:
            List of paths; each path is a list of node names from src to dst.
            Returns an empty list if no path exists.

        Raises:
            KeyError: If src or dst IP is not in the topology.
        """
        # Allow IP address lookup
        src = self._ip_to_node.get(src, src)
        dst = self._ip_to_node.get(dst, dst)

        if src not in self.graph:
            raise KeyError(f"Source node '{src}' not in topology")
        if dst not in self.graph:
            raise KeyError(f"Destination node '{dst}' not in topology")

        if src == dst:
            return [[src]]

        try:
            paths = list(nx.all_shortest_paths(self.graph, src, dst))
        except nx.NetworkXNoPath:
            logger.warning("No path between %s and %s", src, dst)
            return []

        if max_paths is not None:
            paths = paths[:max_paths]

        return paths

    def get_k_shortest_paths(
        self,
        src: str,
        dst: str,
        k: int = 4,
    ) -> List[List[str]]:
        """Return up to k shortest simple paths using Yen's algorithm.

        Unlike ``get_paths``, this returns paths of potentially different
        lengths, ranked by total hop count.  Useful when all equal-cost
        paths are saturated and a longer path must be selected.

        Args:
            src: Source node name or IP.
            dst: Destination node name or IP.
            k:   Maximum number of paths to return.

        Returns:
            List of up to k paths, shortest first.
        """
        src = self._ip_to_node.get(src, src)
        dst = self._ip_to_node.get(dst, dst)

        try:
            gen = nx.shortest_simple_paths(self.graph, src, dst)
            return [next(gen) for _ in range(k) if True]
        except (nx.NetworkXNoPath, StopIteration):
            return []

    def get_ecmp_paths(self, src: str, dst: str) -> List[List[str]]:
        """Alias for get_paths() — returns all equal-cost shortest paths.

        Args:
            src: Source node name or IP address.
            dst: Destination node name or IP address.

        Returns:
            All shortest paths (ECMP set) between src and dst.
        """
        return self.get_paths(src, dst)

    def get_all_paths(self) -> Dict[Tuple[str, str], List[List[str]]]:
        """Compute ECMP paths for every host pair in the topology.

        This is an O(H² × P) operation where H=n_hosts, P=paths per pair.
        For k=8 (128 hosts, 16 paths/pair) expect ~260 000 paths total —
        cache the result if calling repeatedly.

        Returns:
            Dict mapping ``(src_name, dst_name)`` tuples to a list of paths.
        """
        hosts = [n for n, d in self.graph.nodes(data=True)
                 if d.get("node_type") == "host"]
        result: Dict[Tuple[str, str], List[List[str]]] = {}

        for src, dst in product(hosts, hosts):
            if src < dst:  # undirected: only compute once
                paths = self.get_paths(src, dst)
                result[(src, dst)] = paths
                result[(dst, src)] = [list(reversed(p)) for p in paths]

        logger.info(
            "Computed paths for %d host pairs", len(result) // 2
        )
        return result

    # ── Link utilities ────────────────────────────────────────────────────────
    def get_link_capacity(self, u: str, v: str) -> float:
        """Return the bandwidth capacity of the link (u, v) in Gbps.

        Args:
            u: First endpoint node name.
            v: Second endpoint node name.

        Returns:
            Link capacity in Gbps.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.graph.has_edge(u, v):
            raise KeyError(f"No edge between '{u}' and '{v}'")
        return self.graph[u][v]["capacity"]

    def get_all_links(self) -> List[Tuple[str, str, Dict]]:
        """Return all edges with their attribute dictionaries.

        Returns:
            List of ``(u, v, attrs)`` triples.
        """
        return list(self.graph.edges(data=True))

    def get_switch_links(self) -> List[Tuple[str, str, Dict]]:
        """Return only switch-to-switch links (no host-facing links).

        Returns:
            List of ``(u, v, attrs)`` triples for fabric links.
        """
        return [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if d.get("link_type") != "edge-host"
        ]

    # ── Node accessors ────────────────────────────────────────────────────────
    @property
    def core_switches(self) -> List[str]:
        """Names of all core switches, sorted."""
        return sorted(
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "core"
        )

    @property
    def agg_switches(self) -> List[str]:
        """Names of all aggregation switches, sorted."""
        return sorted(
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "agg"
        )

    @property
    def edge_switches(self) -> List[str]:
        """Names of all edge switches, sorted."""
        return sorted(
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "edge"
        )

    @property
    def all_switches(self) -> List[str]:
        """Names of all switches (core + agg + edge), sorted."""
        return sorted(
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in {"core", "agg", "edge"}
        )

    @property
    def hosts(self) -> List[str]:
        """Names of all host nodes, sorted."""
        return sorted(
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "host"
        )

    def get_host_ip(self, host_name_: str) -> str:
        """Return the IP address assigned to a host.

        Args:
            host_name_: Host node name, e.g. ``'h_0_0_0'``.

        Returns:
            IP address string.

        Raises:
            KeyError: If host_name_ not found.
        """
        return self._node_to_ip[host_name_]

    def get_host_mac(self, host_name_: str) -> str:
        """Return the MAC address assigned to a host.

        Args:
            host_name_: Host node name.

        Returns:
            MAC address string.
        """
        return self._node_to_mac[host_name_]

    def node_for_ip(self, ip: str) -> str:
        """Reverse-lookup: return the node name for a given IP.

        Args:
            ip: IP address string.

        Returns:
            Node name.

        Raises:
            KeyError: If IP not found in topology.
        """
        if ip not in self._ip_to_node:
            raise KeyError(f"IP {ip!r} not in topology")
        return self._ip_to_node[ip]

    def get_pod_of_host(self, host: str) -> int:
        """Return the pod number that contains the given host.

        Args:
            host: Host node name or IP address.

        Returns:
            Pod number (0-indexed).
        """
        host = self._ip_to_node.get(host, host)
        return self.graph.nodes[host]["pod"]

    def same_pod(self, h1: str, h2: str) -> bool:
        """Check whether two hosts reside in the same pod.

        Args:
            h1: First host name or IP.
            h2: Second host name or IP.

        Returns:
            ``True`` if both hosts share the same pod.
        """
        return self.get_pod_of_host(h1) == self.get_pod_of_host(h2)

    # ── Statistics ────────────────────────────────────────────────────────────
    def summary(self) -> str:
        """Return a human-readable summary string for this topology.

        Returns:
            Multi-line summary with node and edge counts.
        """
        return (
            f"Fat-Tree Topology (k={self.k})\n"
            f"  Core switches       : {self.n_core}\n"
            f"  Aggregation switches: {self.n_agg}\n"
            f"  Edge switches       : {self.n_edge}\n"
            f"  Total switches      : {self.n_switches}\n"
            f"  Hosts               : {self.n_hosts}\n"
            f"  Total nodes         : {self.graph.number_of_nodes()}\n"
            f"  Total links         : {self.graph.number_of_edges()}\n"
            f"  Max ECMP paths/pair : {self.half_k ** 2}\n"
        )

    def __repr__(self) -> str:
        return (
            f"FatTreeGraph(k={self.k}, "
            f"switches={self.n_switches}, hosts={self.n_hosts})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FatTreeTopo  (Mininet Topo subclass)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from mininet.topo import Topo
    from mininet.link import TCLink
    _MININET_AVAILABLE = True
except ImportError:
    # On Windows / non-Linux, provide a stub so the file can still be imported.
    class Topo:          # type: ignore[no-redef]
        """Stub Mininet Topo for non-Linux environments."""
        def __init__(self, *args, **kwargs): pass
        def addSwitch(self, name, **opts): return name
        def addHost(self, name, **opts): return name
        def addLink(self, *args, **opts): pass
        def build(self, **opts): pass

    TCLink = None
    _MININET_AVAILABLE = False
    logger.warning(
        "Mininet not found — FatTreeTopo will use stub Topo. "
        "FatTreeGraph is fully functional."
    )


class FatTreeTopo(Topo):
    """Mininet Fat-Tree topology for LAFS experiments.

    Inherits from :class:`mininet.topo.Topo` and drives Mininet node/link
    creation.  Internally delegates graph bookkeeping to
    :class:`FatTreeGraph`.

    Args:
        k:         Number of ports per switch (must be even, default 8).
        bw_host:   Host ↔ Edge link bandwidth in Gbps (default 1.0).
        bw_core:   Fabric (switch–switch) link bandwidth in Gbps (default 10.0).
        delay:     Link propagation delay string (default ``'0.1ms'``).
        loss:      Link loss percentage (default 0).

    Example::

        from mininet.net import Mininet
        from mininet.node import OVSSwitch, RemoteController

        topo = FatTreeTopo(k=8)
        net  = Mininet(topo=topo, switch=OVSSwitch,
                       controller=RemoteController)
        net.start()
        net.pingAll()
        net.stop()
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        bw_host: float = BW_HOST_GBPS,
        bw_core: float = BW_FABRIC_GBPS,
        delay: str = LINK_DELAY_MS,
        loss: int = LINK_LOSS_PCT,
        **opts,
    ) -> None:
        self.k       = k
        self.half_k  = k // 2
        self.bw_host = bw_host
        self.bw_core = bw_core
        self.delay   = delay
        self.loss    = loss

        # Validate k before calling super().__init__ which calls build()
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be a positive even integer, got {k}")

        # Build FatTreeGraph (no Mininet) for metadata and path computation
        self._graph = FatTreeGraph(k=k)

        # These track Mininet node names as returned by addSwitch/addHost
        self._mn_switches: Dict[str, str] = {}  # our name → Mininet name
        self._mn_hosts:    Dict[str, str] = {}  # our name → Mininet name

        super().__init__(**opts)

    # ── Mininet Topo.build() ──────────────────────────────────────────────────
    def build(self, **opts) -> None:  # type: ignore[override]
        """Build the Fat-Tree topology in Mininet.

        Called automatically by :class:`mininet.topo.Topo.__init__`.
        Adds all switches, hosts, and links.
        """
        logger.info("Building Mininet Fat-Tree (k=%d)…", self.k)

        self._mn_add_core_switches()
        self._mn_add_pod_switches()
        self._mn_add_hosts()
        self._mn_add_core_to_agg_links()
        self._mn_add_agg_to_edge_links()
        self._mn_add_edge_to_host_links()

        logger.info(
            "Fat-Tree built: %d switches, %d hosts",
            len(self._mn_switches),
            len(self._mn_hosts),
        )

    # ── Mininet node creation ─────────────────────────────────────────────────
    def _mn_add_core_switches(self) -> None:
        """Add core switches to Mininet with OpenFlow 1.3."""
        for row in range(self.half_k):
            for col in range(self.half_k):
                name = core_name(row, col)
                mn_name = self.addSwitch(
                    name,
                    protocols="OpenFlow13",
                    failMode="secure",
                )
                self._mn_switches[name] = mn_name
                logger.debug("Added core switch %s", name)

    def _mn_add_pod_switches(self) -> None:
        """Add aggregation and edge switches for all pods."""
        for pod in range(self.k):
            for idx in range(self.half_k):
                # Aggregation switch
                a = agg_name(pod, idx)
                mn_a = self.addSwitch(
                    a,
                    protocols="OpenFlow13",
                    failMode="secure",
                )
                self._mn_switches[a] = mn_a

                # Edge switch
                e = edge_name(pod, idx)
                mn_e = self.addSwitch(
                    e,
                    protocols="OpenFlow13",
                    failMode="secure",
                )
                self._mn_switches[e] = mn_e

        logger.debug(
            "Added %d pod switches (%d agg + %d edge)",
            2 * self.k * self.half_k,
            self.k * self.half_k,
            self.k * self.half_k,
        )

    def _mn_add_hosts(self) -> None:
        """Add host nodes with IP and MAC configuration."""
        for pod in range(self.k):
            for e_idx in range(self.half_k):
                for h_idx in range(self.half_k):
                    name = host_name(pod, e_idx, h_idx)
                    ip   = host_ip(pod, e_idx, h_idx)
                    mac  = host_mac(pod, e_idx, h_idx)

                    mn_h = self.addHost(
                        name,
                        ip=f"{ip}/8",     # /8 for 10.x.x.x network
                        mac=mac,
                        # Default route via edge switch's gateway address
                        defaultRoute=f"via 10.{pod}.{e_idx}.1",
                    )
                    self._mn_hosts[name] = mn_h

        logger.debug("Added %d hosts", len(self._mn_hosts))

    # ── Mininet link creation ─────────────────────────────────────────────────
    def _link_opts(self, bw: float) -> Dict:
        """Build TCLink keyword arguments for a given bandwidth.

        Args:
            bw: Link bandwidth in Gbps.

        Returns:
            Dict of keyword arguments for :func:`mininet.topo.Topo.addLink`.
        """
        opts: Dict = {
            "delay": self.delay,
            "loss":  self.loss,
            "use_htb": True,
        }
        if _MININET_AVAILABLE and TCLink is not None:
            opts["cls"] = TCLink
            opts["bw"]  = bw * 1000  # TCLink uses Mbps
        return opts

    def _mn_add_core_to_agg_links(self) -> None:
        """Add links between core and aggregation switches.

        Port numbering:
          Core side : ``pod + 1``
          Agg side  : ``half_k + col + 1``
        """
        for pod in range(self.k):
            for row in range(self.half_k):
                a = agg_name(pod, row)
                for col in range(self.half_k):
                    c = core_name(row, col)
                    self.addLink(
                        c, a,
                        port1=pod + 1,
                        port2=self.half_k + col + 1,
                        **self._link_opts(self.bw_core),
                    )

    def _mn_add_agg_to_edge_links(self) -> None:
        """Add links between aggregation and edge switches.

        Port numbering:
          Agg side  : ``e_idx + 1``
          Edge side : ``half_k + a_idx + 1``
        """
        for pod in range(self.k):
            for a_idx in range(self.half_k):
                a = agg_name(pod, a_idx)
                for e_idx in range(self.half_k):
                    e = edge_name(pod, e_idx)
                    self.addLink(
                        a, e,
                        port1=e_idx + 1,
                        port2=self.half_k + a_idx + 1,
                        **self._link_opts(self.bw_core),
                    )

    def _mn_add_edge_to_host_links(self) -> None:
        """Add links between edge switches and hosts.

        Port numbering:
          Edge side : ``h_idx + 1``
          Host side : ``1`` (single NIC)
        """
        for pod in range(self.k):
            for e_idx in range(self.half_k):
                e = edge_name(pod, e_idx)
                for h_idx in range(self.half_k):
                    h = host_name(pod, e_idx, h_idx)
                    self.addLink(
                        e, h,
                        port1=h_idx + 1,
                        port2=1,
                        **self._link_opts(self.bw_host),
                    )

    # ── Delegation to FatTreeGraph ────────────────────────────────────────────
    @property
    def graph(self) -> FatTreeGraph:
        """Underlying :class:`FatTreeGraph` instance for path computation."""
        return self._graph

    def get_paths(
        self,
        src: str,
        dst: str,
        max_paths: Optional[int] = None,
    ) -> List[List[str]]:
        """Delegate to :meth:`FatTreeGraph.get_paths`.

        Args:
            src:       Source node name or IP.
            dst:       Destination node name or IP.
            max_paths: Maximum number of paths to return.

        Returns:
            List of shortest paths between src and dst.
        """
        return self._graph.get_paths(src, dst, max_paths=max_paths)

    def get_ecmp_paths(self, src: str, dst: str) -> List[List[str]]:
        """Delegate to :meth:`FatTreeGraph.get_ecmp_paths`."""
        return self._graph.get_ecmp_paths(src, dst)

    def get_k_shortest_paths(
        self, src: str, dst: str, k: int = 4
    ) -> List[List[str]]:
        """Delegate to :meth:`FatTreeGraph.get_k_shortest_paths`."""
        return self._graph.get_k_shortest_paths(src, dst, k)

    def get_host_ip(self, name: str) -> str:
        """Return the IP address for a host.

        Args:
            name: Host node name.

        Returns:
            IP address string.
        """
        return self._graph.get_host_ip(name)

    def get_host_mac(self, name: str) -> str:
        """Return the MAC address for a host.

        Args:
            name: Host node name.

        Returns:
            MAC address string.
        """
        return self._graph.get_host_mac(name)

    def summary(self) -> str:
        """Return topology summary string."""
        return self._graph.summary()

    def __repr__(self) -> str:
        return (
            f"FatTreeTopo(k={self.k}, bw_host={self.bw_host}G, "
            f"bw_core={self.bw_core}G)"
        )
