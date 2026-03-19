"""
LAFS Topology Package
=====================
Exports the public API for Fat-Tree topology construction.

Classes:
    FatTreeGraph     — pure NetworkX graph (Windows-safe, no Mininet)
    FatTreeTopo      — Mininet Topo subclass (Linux only)
    NetworkBuilder   — high-level Mininet network manager (Linux only)
    NetworkConfig    — configuration dataclass for NetworkBuilder
    ControllerConfig — Ryu controller configuration
    LinkStats        — per-link statistics snapshot

Helpers:
    core_name, agg_name, edge_name, host_name
    host_ip, host_mac
"""

from src.topology.fattree import (
    FatTreeGraph,
    FatTreeTopo,
    core_name,
    agg_name,
    edge_name,
    host_name,
    host_ip,
    host_mac,
    DEFAULT_K,
    BW_HOST_GBPS,
    BW_FABRIC_GBPS,
    LINK_DELAY_MS,
)

from src.topology.network_builder import (
    NetworkBuilder,
    NetworkConfig,
    ControllerConfig,
    LinkStats,
)

__all__ = [
    # Graph / Topology
    "FatTreeGraph",
    "FatTreeTopo",
    # Builder
    "NetworkBuilder",
    "NetworkConfig",
    "ControllerConfig",
    "LinkStats",
    # Naming helpers
    "core_name",
    "agg_name",
    "edge_name",
    "host_name",
    "host_ip",
    "host_mac",
    # Constants
    "DEFAULT_K",
    "BW_HOST_GBPS",
    "BW_FABRIC_GBPS",
    "LINK_DELAY_MS",
]
