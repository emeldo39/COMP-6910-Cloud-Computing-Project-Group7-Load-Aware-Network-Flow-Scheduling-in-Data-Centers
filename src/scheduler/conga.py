"""
LAFS — CONGA Scheduler (Simulated)
====================================
COMP 6910 — Group 7

Simulates the core CONGA (Congestion-aware Load Balancing at the Granularity
of Flowlets for Data Center Networks) algorithm.

Reference
---------
Alizadeh, M., Edsall, T., Dharmapurikar, S., Vaidyanathan, R., Chu, K.,
Fingerhut, A., … Vahdat, A. (2014). CONGA: Distributed Congestion-Aware
Load Balancing for Data Centers. ACM SIGCOMM 2014.

CONGA Algorithm Summary
------------------------
CONGA operates at leaf (edge) switches in a leaf-spine or fat-tree fabric.
Each leaf maintains two tables:

  1. **Congestion Table** (CT):
     ``CT[(src_leaf, dst_leaf, path_idx)] = dre_metric``
     Updated by congestion feedback carried in packet headers from remote
     leaves back to the source leaf.

  2. **Flowlet Table** (FT):
     ``FT[5-tuple] = (path_idx, last_seen_time)``
     Flowlets are bursts of packets with inter-packet gaps smaller than
     ``FLOWLET_GAP``.  A new flowlet (gap > threshold) may be rerouted.

Path Selection per Flowlet
--------------------------
  1. Look up the flowlet in FT.
  2. If the flowlet is new or the gap since the last packet exceeds the
     flowlet gap:
       a. Query CT for all available paths to the destination leaf.
       b. Select the path with the minimum DRE congestion metric.
       c. Update FT with the selected path and current time.
  3. If the flowlet is continuing, use the path already recorded in FT.

Congestion Metric (DRE — Distributed Rate Estimation)
------------------------------------------------------
In real CONGA, each switch computes a local DRE value from its transmit
byte counter using EWMA:

    DRE_new = (1 - alpha) * DRE_old + alpha * (bytes_in_window / capacity)

where alpha controls the decay rate.

Simulation Model
----------------
Since we do not have real switch packet counters, this simulation:

  * Assigns a simulated DRE value to each path based on the total bytes
    assigned to that path divided by a notional capacity.
  * Updates DRE using EWMA after each flow assignment.
  * Applies a time-based exponential decay to DRE to model traffic draining.
  * Simulates flowlet detection by comparing ``flow.arrival_time`` with the
    last-seen time stored in the flowlet table.

Flowlet Gap (FLOWLET_GAP_S)
---------------------------
Default: 500 µs (0.0005 s). Flows arriving at the same scheduler within
this window are treated as part of the same flowlet and sent on the same
path.  A gap larger than this allows rerouting.

This matches production CONGA deployments where FLOWLET_GAP ≈ 500 µs
(roughly 2 × maximum RTT within a fat-tree pod).
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.scheduler.base_scheduler import BaseScheduler
from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow

_log = logging.getLogger("lafs.scheduler.conga")


# ── Constants ─────────────────────────────────────────────────────────────────
FLOWLET_GAP_S: float = 0.0005          # 500 µs inter-flowlet gap
DRE_ALPHA: float = 0.2                  # EWMA smoothing factor for DRE update
DRE_DECAY_RATE: float = 0.95            # per-second exponential decay factor
PATH_CAPACITY_GBPS: float = 1.0         # bottleneck link capacity (Gbps)
PATH_CAPACITY_BYTES_S: float = PATH_CAPACITY_GBPS * 1e9 / 8  # bytes/s


# =============================================================================
# CongestionTable
# =============================================================================
class CongestionTable:
    """
    Per-leaf congestion feedback table.

    Stores DRE (Distributed Rate Estimation) metrics for each
    (src_leaf, dst_leaf, path_index) triple.  Values are in [0.0, 1.0]
    where 0.0 = no congestion and 1.0 = full saturation.

    In real CONGA, the table is stored in TCAM on the leaf switch ASIC and
    updated by feedback carried in the ECN bits of returning packets.  Here
    we maintain it as a Python dict and update it after each simulated flow
    assignment.

    Parameters
    ----------
    alpha : float
        EWMA smoothing factor.  Larger values react faster but are noisier.
    decay_rate : float
        Fraction of DRE remaining after one second with no new traffic.
        E.g. 0.95 means DRE decays to 5% of its value per second of silence.
    """

    def __init__(
        self,
        alpha: float = DRE_ALPHA,
        decay_rate: float = DRE_DECAY_RATE,
    ) -> None:
        self._alpha = alpha
        self._decay_rate = decay_rate
        # (src_leaf, dst_leaf, path_idx) → DRE value [0.0, 1.0]
        self._table: Dict[Tuple[str, str, int], float] = defaultdict(float)
        # (src_leaf, dst_leaf, path_idx) → wall-clock time of last update
        self._last_update: Dict[Tuple[str, str, int], float] = {}

    def get(self, src_leaf: str, dst_leaf: str, path_idx: int) -> float:
        """Return the current DRE metric, applying time-based decay first."""
        key = (src_leaf, dst_leaf, path_idx)
        self._apply_decay(key)
        return self._table[key]

    def update(
        self,
        src_leaf: str,
        dst_leaf: str,
        path_idx: int,
        added_bytes: int,
        window_s: float = 0.001,
    ) -> float:
        """
        Update DRE for a (src_leaf, dst_leaf, path_idx) triple using EWMA.

        The new utilisation sample is computed as::

            sample = added_bytes / (PATH_CAPACITY_BYTES_S * window_s)

        Then merged into the running DRE::

            DRE_new = (1 - alpha) * DRE_decayed + alpha * sample

        Parameters
        ----------
        src_leaf, dst_leaf : str
            Leaf switch names (edge switches in fat-tree).
        path_idx : int
            Index into the ECMP path list for this (src, dst) pair.
        added_bytes : int
            Bytes added to this path in the current scheduling decision.
        window_s : float
            Assumed measurement window in seconds (default 1 ms).

        Returns
        -------
        float
            Updated DRE value (clamped to [0.0, 1.0]).
        """
        key = (src_leaf, dst_leaf, path_idx)
        self._apply_decay(key)

        capacity = PATH_CAPACITY_BYTES_S * window_s
        sample = min(1.0, added_bytes / capacity if capacity > 0 else 1.0)

        old = self._table[key]
        new = (1.0 - self._alpha) * old + self._alpha * sample
        new = max(0.0, min(1.0, new))  # clamp to [0, 1]
        self._table[key] = new
        self._last_update[key] = time.monotonic()
        return new

    def best_path_idx(
        self, src_leaf: str, dst_leaf: str, n_paths: int
    ) -> Tuple[int, float]:
        """
        Return the path index with the minimum DRE metric.

        Ties are broken by path index (deterministic).

        Parameters
        ----------
        src_leaf, dst_leaf : str
            Leaf switch identifiers.
        n_paths : int
            Number of ECMP paths available (indexes 0 … n_paths-1).

        Returns
        -------
        (path_idx, dre_value) : (int, float)
        """
        best_idx, best_dre = 0, float("inf")
        for idx in range(n_paths):
            dre = self.get(src_leaf, dst_leaf, idx)
            if dre < best_dre:
                best_idx, best_dre = idx, dre
        return best_idx, best_dre

    def _apply_decay(self, key: Tuple[str, str, int]) -> None:
        """Decay DRE based on elapsed time since last update."""
        if key not in self._last_update:
            return
        elapsed = time.monotonic() - self._last_update[key]
        if elapsed <= 0:
            return
        # Exponential decay: DRE *= decay_rate ** elapsed_seconds
        decay = self._decay_rate ** elapsed
        self._table[key] *= decay
        self._last_update[key] = time.monotonic()

    def reset(self) -> None:
        """Clear all DRE state."""
        self._table.clear()
        self._last_update.clear()

    def snapshot(self) -> Dict[Tuple[str, str, int], float]:
        """Return a copy of all current DRE values (after decay)."""
        for key in list(self._table.keys()):
            self._apply_decay(key)
        return dict(self._table)


# =============================================================================
# FlowletTable
# =============================================================================
class FlowletTable:
    """
    Per-leaf flowlet detection table.

    Maps each active 5-tuple to its current path index and the wall-clock
    time of the last packet seen on that flowlet.  When the gap since the
    last packet exceeds ``flowlet_gap_s``, the next packet starts a new
    flowlet and may be rerouted.

    Parameters
    ----------
    flowlet_gap_s : float
        Inter-packet gap threshold in seconds that separates flowlets.
    """

    def __init__(self, flowlet_gap_s: float = FLOWLET_GAP_S) -> None:
        self._gap = flowlet_gap_s
        # 5-tuple → (path_idx, last_seen_time)
        self._table: Dict[tuple, Tuple[int, float]] = {}
        self._new_flowlets: int = 0      # total new flowlets detected
        self._continuing: int = 0        # total continuing flowlets

    def lookup(
        self, five_tuple: tuple, now: float
    ) -> Optional[int]:
        """
        Look up *five_tuple* in the flowlet table.

        Returns the existing path index if the flowlet is continuing
        (i.e. ``now - last_seen < gap``), or ``None`` if this is a new
        flowlet (or the entry has expired).

        Parameters
        ----------
        five_tuple : tuple
            (src_ip, dst_ip, src_port, dst_port, protocol)
        now : float
            Current time in seconds (monotonic).

        Returns
        -------
        int or None
            Existing path index, or None for a new flowlet.
        """
        if five_tuple not in self._table:
            return None
        path_idx, last_seen = self._table[five_tuple]
        if (now - last_seen) >= self._gap:
            # Gap exceeded → new flowlet, may be rerouted.
            return None
        return path_idx

    def record(self, five_tuple: tuple, path_idx: int, now: float) -> None:
        """
        Record that *five_tuple* was forwarded on *path_idx* at time *now*.
        """
        is_new = (five_tuple not in self._table) or (
            (now - self._table[five_tuple][1]) >= self._gap
        )
        if is_new:
            self._new_flowlets += 1
        else:
            self._continuing += 1
        self._table[five_tuple] = (path_idx, now)

    def evict_expired(self, now: float) -> int:
        """
        Remove entries older than ``gap`` seconds.  Returns count evicted.

        Call periodically to prevent unbounded table growth.
        """
        expired = [k for k, (_, t) in self._table.items() if now - t >= self._gap]
        for k in expired:
            del self._table[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._table)

    @property
    def new_flowlets(self) -> int:
        return self._new_flowlets

    @property
    def continuing_flowlets(self) -> int:
        return self._continuing

    def reset(self) -> None:
        self._table.clear()
        self._new_flowlets = 0
        self._continuing = 0


# =============================================================================
# CONGAScheduler
# =============================================================================
class CONGAScheduler(BaseScheduler):
    """
    Simulated CONGA load balancer for the LAFS Fat-tree topology.

    Combines flowlet detection (FlowletTable) with congestion-aware path
    selection (CongestionTable) to distribute traffic more evenly than
    static ECMP, especially under asymmetric load.

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology used for ECMP path lookup.
    flowlet_gap_s : float, optional
        Inter-packet gap threshold for flowlet detection (default 500 µs).
    dre_alpha : float, optional
        EWMA alpha for DRE updates (default 0.2).
    dre_decay_rate : float, optional
        Per-second DRE decay factor (default 0.95).

    Attributes
    ----------
    congestion_table : CongestionTable
        Tracks per-path DRE metrics.
    flowlet_table : FlowletTable
        Tracks active flowlets and their assigned paths.

    Examples
    --------
    >>> topo = FatTreeGraph(k=4)
    >>> sched = CONGAScheduler(topo)
    >>> flow = Flow.create("10.0.0.2", "10.1.0.2", 12345, 80)
    >>> path = sched.schedule_flow(flow)
    >>> path[0], path[-1]
    ('h_0_0_0', 'h_1_0_0')
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        flowlet_gap_s: float = FLOWLET_GAP_S,
        dre_alpha: float = DRE_ALPHA,
        dre_decay_rate: float = DRE_DECAY_RATE,
    ) -> None:
        super().__init__(topology)
        self.flowlet_gap_s = flowlet_gap_s
        self.congestion_table = CongestionTable(
            alpha=dre_alpha, decay_rate=dre_decay_rate
        )
        self.flowlet_table = FlowletTable(flowlet_gap_s=flowlet_gap_s)

        # Path cache: (src_node, dst_node) → list of paths
        self._path_cache: Dict[Tuple[str, str], List[List[str]]] = {}

        self._log.debug(
            "CONGAScheduler ready: flowlet_gap=%.3f ms, dre_alpha=%.2f",
            flowlet_gap_s * 1000, dre_alpha,
        )

    # ── BaseScheduler interface ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "conga"

    def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
        """
        Route one flow using CONGA's flowlet-aware, congestion-driven logic.

        Steps
        -----
        1. Resolve src/dst IPs to edge (leaf) switch names.
        2. Retrieve cached ECMP candidate paths.
        3. Look up the flowlet table:
           * If the 5-tuple is a **continuing flowlet**: reuse the existing
             path to preserve in-order delivery.
           * If this is a **new flowlet**: query the congestion table and
             select the path with the minimum DRE metric.
        4. Update the flowlet table and DRE for the selected path.
        5. Return the selected path.

        Parameters
        ----------
        flow : Flow
            The flow (flowlet) to schedule.

        Returns
        -------
        List[str] or None
            Selected path, or None if src/dst IP not in topology.
        """
        now = time.monotonic()

        # ── 1. IP → node ──────────────────────────────────────────────────────
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

        if src_node == dst_node:
            return [src_node]

        # ── 2. Candidate paths ────────────────────────────────────────────────
        cache_key = (src_node, dst_node)
        paths = self._get_paths_cached(cache_key, src_node, dst_node)
        if not paths:
            return None
        n_paths = len(paths)

        # ── 3. Leaf switch identification ─────────────────────────────────────
        # In fat-tree: the first switch after the host is its edge (leaf) switch.
        src_leaf = paths[0][1] if len(paths[0]) > 1 else src_node
        dst_leaf = paths[0][-2] if len(paths[0]) > 1 else dst_node

        # ── 4. Flowlet lookup ─────────────────────────────────────────────────
        existing_idx = self.flowlet_table.lookup(flow.five_tuple, now)

        if existing_idx is not None and existing_idx < n_paths:
            # Continuing flowlet — preserve path for in-order delivery.
            path_idx = existing_idx
            self._log.debug(
                "Flow %s: continuing flowlet → path[%d]", flow.flow_id, path_idx
            )
        else:
            # New flowlet — select least-congested path.
            path_idx, dre = self.congestion_table.best_path_idx(
                src_leaf, dst_leaf, n_paths
            )
            self._log.debug(
                "Flow %s: new flowlet → path[%d] (DRE=%.4f)",
                flow.flow_id, path_idx, dre,
            )

        # ── 5. Update state ───────────────────────────────────────────────────
        selected_path = paths[path_idx]
        self.flowlet_table.record(flow.five_tuple, path_idx, now)
        self.congestion_table.update(
            src_leaf, dst_leaf, path_idx, flow.size_bytes
        )

        return selected_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_paths_cached(
        self, key: Tuple[str, str], src: str, dst: str
    ) -> List[List[str]]:
        if key not in self._path_cache:
            self._path_cache[key] = self.topology.get_paths(src, dst)
        return self._path_cache[key]

    # ── Control plane API ─────────────────────────────────────────────────────

    def evict_expired_flowlets(self) -> int:
        """
        Purge stale entries from the flowlet table.

        Returns the number of entries evicted.  Call this periodically
        (e.g. once per scheduling epoch) to bound memory usage.
        """
        evicted = self.flowlet_table.evict_expired(time.monotonic())
        if evicted:
            self._log.debug("Evicted %d expired flowlet entries", evicted)
        return evicted

    def inject_congestion(
        self,
        src_leaf: str,
        dst_leaf: str,
        path_idx: int,
        dre_value: float,
    ) -> None:
        """
        Directly set a DRE value in the congestion table.

        Use this in tests or to simulate feedback from a remote leaf switch
        that has detected congestion on a specific path.

        Parameters
        ----------
        src_leaf, dst_leaf : str
            Edge switch pair identifying the path set.
        path_idx : int
            Index of the congested path.
        dre_value : float
            DRE congestion level in [0.0, 1.0].
        """
        key = (src_leaf, dst_leaf, path_idx)
        self.congestion_table._table[key] = max(0.0, min(1.0, dre_value))
        self.congestion_table._last_update[key] = time.monotonic()
        self._log.debug(
            "Injected congestion: %s→%s path[%d] DRE=%.3f",
            src_leaf, dst_leaf, path_idx, dre_value,
        )

    # ── Statistics / analysis ─────────────────────────────────────────────────

    def flowlet_stats(self) -> Dict[str, int]:
        """Return flowlet detection statistics."""
        return {
            "table_size": self.flowlet_table.size,
            "new_flowlets": self.flowlet_table.new_flowlets,
            "continuing_flowlets": self.flowlet_table.continuing_flowlets,
        }

    def congestion_snapshot(self) -> Dict[Tuple[str, str, int], float]:
        """Return all current DRE values (after decay)."""
        return self.congestion_table.snapshot()

    def conga_stats(self) -> str:
        """Return a formatted CONGA-specific statistics block."""
        ft = self.flowlet_stats()
        ct = self.congestion_snapshot()
        max_dre = max(ct.values()) if ct else 0.0
        lines = [
            "=== CONGA Statistics ===",
            f"Flowlet table size  : {ft['table_size']}",
            f"New flowlets        : {ft['new_flowlets']}",
            f"Continuing flowlets : {ft['continuing_flowlets']}",
            f"Congestion entries  : {len(ct)}",
            f"Max DRE observed    : {max_dre:.4f}",
            f"Path cache size     : {len(self._path_cache)} pairs",
        ]
        return "\n".join(lines)

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.congestion_table.reset()
        self.flowlet_table.reset()
        self._path_cache.clear()

    def __repr__(self) -> str:
        return (
            f"CONGAScheduler("
            f"k={self.topology.k}, "
            f"scheduled={self.metrics.flows_scheduled}, "
            f"flowlets={self.flowlet_table.size})"
        )
