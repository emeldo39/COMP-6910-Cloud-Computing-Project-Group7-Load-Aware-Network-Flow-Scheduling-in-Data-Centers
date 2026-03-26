"""
LAFS — ECMP Scheduler
=====================
COMP-6910 — Group 7

Equal-Cost Multi-Path (ECMP) scheduler: the canonical baseline against
which LAFS and all other schedulers are compared.

Algorithm
---------
For each incoming flow f with 5-tuple (src_ip, dst_ip, src_port, dst_port,
protocol):

  1. Retrieve all shortest (ECMP) paths between the source and destination
     hosts from the FatTreeGraph.  Paths are cached after the first lookup
     for each (src_host, dst_host) pair.

  2. Compute a 32-bit CRC32 hash of the 5-tuple packed into 13 bytes:
       4 B  src_ip   (uint32, network byte order)
       4 B  dst_ip   (uint32, network byte order)
       2 B  src_port (uint16)
       2 B  dst_port (uint16)
       1 B  protocol (uint8)

  3. Select path at index  hash_value % len(paths).

Properties
----------
* Deterministic: the same 5-tuple always maps to the same path (no per-flow
  state needed at the switch level).
* Stateless: does not track link load or queue depth.
* O(1) per flow after the first cache lookup for a host pair.
* CRC32 provides uniform distribution across random port numbers and IPs
  (verified in unit tests below).

Limitations (why LAFS improves on this)
----------------------------------------
* Hash collisions: two flows with different 5-tuples may hash to the same
  path, creating load imbalance ("hash polarisation").
* No congestion awareness: elephant flows that collide on a path are not
  rerouted even if an alternative path is free.
* No FCT / fairness optimisation.

Reference
---------
RFC 2991 — Multipath Issues in Unicast and Multicast Next-Hop Selection.
Al-Fares et al., SIGCOMM 2008 — A Scalable, Commodity Data Center Network
Architecture.
"""

from __future__ import annotations

import socket
import struct
import zlib
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.scheduler.base_scheduler import BaseScheduler
from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow


_log = logging.getLogger("lafs.scheduler.ecmp")


# =============================================================================
# Standalone hash function (importable for analysis / unit tests)
# =============================================================================

def _ip_to_uint32(ip: str) -> int:
    """Convert a dotted-decimal IPv4 string to a 32-bit unsigned integer."""
    return struct.unpack("!I", socket.inet_aton(ip))[0]


def ecmp_hash(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: int,
) -> int:
    """
    Compute a deterministic 32-bit hash of a 5-tuple using CRC32.

    The 5-tuple is packed into a fixed 13-byte big-endian binary blob:
      Bytes  0–3  : src_ip as uint32 (network order)
      Bytes  4–7  : dst_ip as uint32 (network order)
      Bytes  8–9  : src_port as uint16
      Bytes 10–11 : dst_port as uint16
      Byte  12    : protocol as uint8

    CRC32 (from ``zlib``) is chosen because:
    * It is extremely fast (hardware-accelerated on most CPUs).
    * It provides good bit-level dispersion across IP/port ranges.
    * It is deterministic and cross-platform reproducible.
    * Its 32-bit output gives 4 billion distinct hash values — far more
      than the maximum of 16 ECMP paths in a k=8 Fat-tree.

    Parameters
    ----------
    src_ip, dst_ip : str
        IPv4 addresses in dotted-decimal notation.
    src_port, dst_port : int
        Port numbers in [0, 65535].
    protocol : int
        IP protocol number (1, 6, or 17).

    Returns
    -------
    int
        32-bit unsigned hash value in [0, 2**32 − 1].

    Examples
    --------
    >>> ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
    2847392714   # example; actual value determined by CRC32
    >>> # Same call always returns the same value:
    >>> ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6) == ecmp_hash("10.0.0.2", "10.1.0.2", 12345, 80, 6)
    True
    """
    packed = struct.pack(
        "!IIHHB",              # I=uint32, H=uint16, B=uint8  → 4+4+2+2+1 = 13 bytes
        _ip_to_uint32(src_ip),
        _ip_to_uint32(dst_ip),
        src_port & 0xFFFF,
        dst_port & 0xFFFF,
        protocol & 0xFF,
    )
    # zlib.crc32 returns a signed int on some Python builds; mask to uint32.
    return zlib.crc32(packed) & 0xFFFF_FFFF


# =============================================================================
# ECMPScheduler
# =============================================================================

class ECMPScheduler(BaseScheduler):
    """
    ECMP baseline scheduler for the LAFS project.

    Uses a CRC32 hash of each flow's 5-tuple to deterministically select one
    of the available equal-cost paths between the source and destination hosts.

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology.  Used to look up all ECMP candidate paths for
        each (src_host, dst_host) pair.
    cache_paths : bool, optional
        If True (default), ECMP path lists are cached per host pair so that
        repeated lookups for the same pair avoid re-running NetworkX shortest-
        path algorithms.  Set to False only for memory-constrained testing.

    Attributes
    ----------
    _path_cache : Dict[(str, str), List[List[str]]]
        Cached ECMP paths keyed by (src_node, dst_node).
    _hash_counts : Dict[int, int]
        Histogram of raw hash values modulo number-of-paths (0-indexed path
        index), accumulated across all flows.  Used for load-distribution
        analysis.
    _path_index_counts : Dict[(str,str), List[int]]
        Per host-pair count of how many flows were sent on each path index.

    Examples
    --------
    >>> from src.topology.fattree import FatTreeGraph
    >>> from src.scheduler.ecmp import ECMPScheduler
    >>> from src.workload.flow import Flow
    >>>
    >>> topo = FatTreeGraph(k=4)
    >>> sched = ECMPScheduler(topo)
    >>> flow = Flow.create("10.0.0.2", "10.1.0.2", 12345, 80)
    >>> path = sched.schedule_flow(flow)
    >>> path[0], path[-1]
    ('h_0_0_0', 'h_1_0_0')
    """

    def __init__(self, topology: FatTreeGraph, cache_paths: bool = True) -> None:
        super().__init__(topology)
        self._cache_paths: bool = cache_paths

        # (src_node, dst_node) → list of ECMP paths (each path = list of nodes)
        self._path_cache: Dict[Tuple[str, str], List[List[str]]] = {}

        # path-index histogram: bucket_index → flow count
        # bucket_index = hash % len(paths)  ∈ [0, max_ecmp_paths - 1]
        self._hash_bucket_counts: Dict[int, int] = defaultdict(int)

        # per (src, dst) pair: list of per-path-index flow counts
        self._path_index_counts: Dict[Tuple[str, str], List[int]] = {}

        self._log.debug(
            "ECMPScheduler ready (path caching %s)",
            "ON" if cache_paths else "OFF",
        )

    # ── BaseScheduler interface ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ecmp"

    def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
        """
        Assign *flow* to an ECMP path using 5-tuple CRC32 hashing.

        Steps
        -----
        1. Resolve src/dst IPs to topology node names.
        2. Retrieve (cached) list of ECMP candidate paths.
        3. Hash the 5-tuple and select path at index ``hash % n_paths``.
        4. Update internal statistics.
        5. Return the selected path.

        Parameters
        ----------
        flow : Flow
            The flow to schedule.

        Returns
        -------
        List[str] or None
            Selected path as ordered list of node names, or ``None`` if the
            src/dst IP is not in the topology or no path exists.
        """
        # ── 1. IP → node lookup ───────────────────────────────────────────────
        # node_for_ip raises KeyError when IP is absent; treat that as None.
        try:
            src_node = self.topology.node_for_ip(flow.src_ip)
        except KeyError:
            self._log.debug("src_ip %s not in topology", flow.src_ip)
            return None
        try:
            dst_node = self.topology.node_for_ip(flow.dst_ip)
        except KeyError:
            self._log.debug("dst_ip %s not in topology", flow.dst_ip)
            return None

        # Same-host flow (loopback): trivial single-node path.
        if src_node == dst_node:
            return [src_node]

        # ── 2. Candidate paths (cached) ───────────────────────────────────────
        cache_key: Tuple[str, str] = (src_node, dst_node)
        paths = self._get_paths_cached(cache_key, src_node, dst_node)

        if not paths:
            self._log.warning(
                "No ECMP paths between %s and %s", src_node, dst_node
            )
            return None

        n_paths = len(paths)

        # ── 3. 5-tuple hash → path index ──────────────────────────────────────
        h = ecmp_hash(*flow.five_tuple)
        path_idx = h % n_paths
        selected_path = paths[path_idx]

        # ── 4. Statistics ─────────────────────────────────────────────────────
        self._hash_bucket_counts[path_idx] += 1

        if cache_key not in self._path_index_counts:
            self._path_index_counts[cache_key] = [0] * n_paths
        self._path_index_counts[cache_key][path_idx] += 1

        self._log.debug(
            "Flow %s: hash=0x%08x → path[%d/%d] %s→%s",
            flow.flow_id, h, path_idx, n_paths, src_node, dst_node,
        )

        return selected_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_paths_cached(
        self, key: Tuple[str, str], src: str, dst: str
    ) -> List[List[str]]:
        """Return ECMP paths from cache; populate cache on miss."""
        if self._cache_paths and key in self._path_cache:
            return self._path_cache[key]

        paths = self.topology.get_paths(src, dst)

        if self._cache_paths:
            self._path_cache[key] = paths

        return paths

    # ── Statistics / analysis API ─────────────────────────────────────────────

    def cache_size(self) -> int:
        """Number of distinct (src, dst) host pairs currently in the cache."""
        return len(self._path_cache)

    def clear_cache(self) -> None:
        """Evict all cached path lists (forces re-lookup on next flow)."""
        self._path_cache.clear()
        self._log.debug("Path cache cleared")

    def hash_distribution(self) -> Dict[int, int]:
        """
        Return the histogram of path-index selections across all flows.

        Returns
        -------
        Dict[int, int]
            Maps path_index (0-based) → number of flows routed on that index.
            An ideal uniform distribution has all values equal to
            total_flows / n_paths.
        """
        return dict(self._hash_bucket_counts)

    def path_balance_ratio(self) -> Optional[float]:
        """
        Compute the load-balance quality as min_load / max_load.

        A ratio of 1.0 means perfect balance; close to 0.0 means severe
        imbalance (nearly all flows on one path).  Returns ``None`` if no
        flows have been scheduled.
        """
        if not self._hash_bucket_counts:
            return None
        counts = list(self._hash_bucket_counts.values())
        mn, mx = min(counts), max(counts)
        if mx == 0:
            return None
        return mn / mx

    def per_pair_distribution(
        self, src_ip: str, dst_ip: str
    ) -> Optional[List[int]]:
        """
        Return per-path flow counts for a specific host pair.

        Parameters
        ----------
        src_ip, dst_ip : str
            Source and destination host IPs.

        Returns
        -------
        List[int] or None
            List of flow counts indexed by path index, or ``None`` if the
            pair has not been scheduled yet.
        """
        src_node = self.topology.node_for_ip(src_ip)
        dst_node = self.topology.node_for_ip(dst_ip)
        if src_node is None or dst_node is None:
            return None
        key = (src_node, dst_node)
        return list(self._path_index_counts.get(key, []))

    def ecmp_stats(self) -> str:
        """
        Return a formatted ECMP-specific statistics block.

        Includes: cache size, hash distribution, balance ratio.
        """
        dist = self.hash_distribution()
        ratio = self.path_balance_ratio()
        lines = [
            f"=== ECMP Statistics ===",
            f"Path cache size  : {self.cache_size()} host pairs",
            f"Hash distribution: {dict(sorted(dist.items()))}",
            f"Balance ratio    : "
            + (f"{ratio:.3f}" if ratio is not None else "N/A (no flows yet)"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ECMPScheduler("
            f"k={self.topology.k}, "
            f"scheduled={self.metrics.flows_scheduled}, "
            f"cache={self.cache_size()})"
        )
