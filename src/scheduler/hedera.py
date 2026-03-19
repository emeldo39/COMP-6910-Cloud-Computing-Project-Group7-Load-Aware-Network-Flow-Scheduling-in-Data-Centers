"""
LAFS — Hedera Scheduler
=======================
COMP 6910 — Group 7

Simulates the Hedera global traffic management system for fat-tree networks.

Reference
---------
Al-Fares, M., Radhakrishnan, S., Raghavan, B., Huang, N., & Vahdat, A. (2010).
Hedera: Dynamic Flow Scheduling for Data Center Networks. NSDI 2010.

Algorithm Overview
------------------
Hedera separates flows into two classes:

  Mice flows (< elephant_threshold_bytes):
    Routed using standard ECMP 5-tuple hashing — no intervention.

  Elephant flows (≥ elephant_threshold_bytes):
    Managed by a centralised global scheduler using the **Global First Fit
    (GFF)** algorithm:

      1. Collect all active elephant-flow demands (size_bytes as bandwidth
         proxy in this simulation; in production Hedera uses switch-reported
         byte counters).
      2. Sort elephant flows by demand descending (largest first).
      3. For each elephant flow, try all ECMP candidate paths. Pick the path
         with the most remaining capacity (i.e. lowest current load).
      4. Assign the flow to that path and update the load accounting.

Load Accounting
---------------
``PathLoadTracker`` maintains a byte-count per unique path (keyed by the
tuple of node names).  For this simulation:

  * Flows are added when scheduled; they are never automatically expired
    (call ``release_flow()`` to remove a completed flow).
  * Path capacity is 1 Gbps × flow_duration (infinite if duration unknown),
    so the scheduler always picks the least-loaded path regardless of whether
    it is technically over-subscribed.

Compared to ECMP
----------------
ECMP can cause hash collisions where multiple elephant flows land on the same
path while parallel paths sit idle.  Hedera's GFF reduces this by actively
placing large flows on underutilised paths.

Simulation Limitations
----------------------
In a real Hedera deployment:
  * Demand is estimated from per-flow byte counters polled from switches.
  * Re-scheduling is triggered periodically (every ~300 ms).
  * Flows may be live-migrated by installing new flow rules on OVS.
This simulation models the path-selection logic faithfully but does not poll
real switch counters or install OpenFlow rules.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.scheduler.base_scheduler import BaseScheduler
from src.scheduler.ecmp import ECMPScheduler, ecmp_hash
from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow, ELEPHANT_THRESHOLD_BYTES

_log = logging.getLogger("lafs.scheduler.hedera")


# =============================================================================
# PathLoadTracker
# =============================================================================
class PathLoadTracker:
    """
    Tracks cumulative byte load on each network path.

    A path is identified by the tuple of its node names, e.g.::

        ("h_0_0_0", "e_0_0", "a_0_1", "c_1_0", "a_1_0", "e_1_0", "h_1_0_0")

    Bytes are additive: assigning two 1 MB flows to the same path records
    2 MB of load on that path.  Call ``release_flow()`` when a flow completes
    to subtract its bytes.

    Parameters
    ----------
    path_capacity_gbps : float
        Notional bottleneck link capacity per path in Gbps.
        Used to compute utilisation fractions (for reporting only —
        the scheduler always picks the least-loaded path regardless).
    """

    def __init__(self, path_capacity_gbps: float = 1.0) -> None:
        self._capacity_bits: float = path_capacity_gbps * 1e9
        # path_tuple → total bytes assigned
        self._load_bytes: Dict[tuple, int] = defaultdict(int)
        # flow_id → (path_tuple, bytes) so we can release later
        self._flow_registry: Dict[str, Tuple[tuple, int]] = {}

    def assign(self, flow: Flow, path: List[str]) -> None:
        """Add flow's bytes to the given path's load."""
        key = tuple(path)
        self._load_bytes[key] += flow.size_bytes
        self._flow_registry[flow.flow_id] = (key, flow.size_bytes)

    def release(self, flow_id: str) -> bool:
        """
        Remove a completed flow's bytes from its path.

        Returns True if the flow was found and released, False otherwise.
        """
        if flow_id not in self._flow_registry:
            return False
        key, nbytes = self._flow_registry.pop(flow_id)
        self._load_bytes[key] = max(0, self._load_bytes[key] - nbytes)
        if self._load_bytes[key] == 0:
            del self._load_bytes[key]
        return True

    def get_load_bytes(self, path: List[str]) -> int:
        """Return the total bytes assigned to *path* (0 if never used)."""
        return self._load_bytes.get(tuple(path), 0)

    def get_utilisation(self, path: List[str]) -> float:
        """
        Return the fraction of capacity consumed (0.0–∞).

        Values > 1.0 indicate over-subscription in the simulation model.
        In practice, a real scheduler would cap assignment to ≤ capacity.
        """
        if self._capacity_bits == 0:
            return 0.0
        # Treat bytes as if they were bits/s for a 1-second window.
        return (self._load_bytes.get(tuple(path), 0) * 8) / self._capacity_bits

    def least_loaded_path(self, paths: List[List[str]]) -> Optional[List[str]]:
        """Return the path in *paths* with the fewest bytes currently assigned."""
        if not paths:
            return None
        return min(paths, key=lambda p: self._load_bytes.get(tuple(p), 0))

    def all_loads(self) -> Dict[tuple, int]:
        """Return a copy of the full load table (path_tuple → bytes)."""
        return dict(self._load_bytes)

    def reset(self) -> None:
        """Clear all load and registry state."""
        self._load_bytes.clear()
        self._flow_registry.clear()


# =============================================================================
# HederaScheduler
# =============================================================================
class HederaScheduler(BaseScheduler):
    """
    Hedera-style Global First Fit scheduler.

    Mice flows are handled by an internal ECMPScheduler (no per-flow state).
    Elephant flows are assigned via GFF to the least-loaded available path.

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology.
    elephant_threshold_bytes : int, optional
        Minimum flow size to be treated as an elephant.
        Defaults to ``ELEPHANT_THRESHOLD_BYTES`` (1 MB).
    path_capacity_gbps : float, optional
        Per-path bottleneck capacity in Gbps for utilisation reporting.
        Default 1.0 Gbps (host-link rate in the LAFS Fat-tree).

    Attributes
    ----------
    _ecmp : ECMPScheduler
        Internal ECMP scheduler for mice flows.
    _load : PathLoadTracker
        Byte-load accounting for elephant paths.
    elephant_threshold : int
        Size threshold (bytes) separating mice from elephants.

    Examples
    --------
    >>> topo = FatTreeGraph(k=4)
    >>> sched = HederaScheduler(topo)
    >>> mice  = Flow.create("10.0.0.2", "10.1.0.2", 1000, 80, size_bytes=50_000)
    >>> giant = Flow.create("10.0.0.2", "10.1.0.2", 2000, 80, size_bytes=5_000_000)
    >>> sched.schedule_flow(mice)   # routed via ECMP hash
    [...]
    >>> sched.schedule_flow(giant)  # routed via GFF (least-loaded path)
    [...]
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        elephant_threshold_bytes: int = ELEPHANT_THRESHOLD_BYTES,
        path_capacity_gbps: float = 1.0,
    ) -> None:
        super().__init__(topology)
        self.elephant_threshold: int = elephant_threshold_bytes
        self._ecmp = ECMPScheduler(topology, cache_paths=True)
        self._load = PathLoadTracker(path_capacity_gbps=path_capacity_gbps)

        # Statistics
        self._mice_count: int = 0
        self._elephant_count: int = 0
        self._reschedule_count: int = 0   # flows moved during reschedule_elephants()

        self._log.debug(
            "HederaScheduler ready: elephant_threshold=%d bytes, "
            "path_capacity=%.1f Gbps",
            elephant_threshold_bytes, path_capacity_gbps,
        )

    # ── BaseScheduler interface ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "hedera"

    def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
        """
        Route one flow.

        * Mice (size < ``elephant_threshold``): delegate to internal ECMP.
        * Elephant (size ≥ ``elephant_threshold``): apply Global First Fit —
          choose the candidate path with the smallest current byte-load.

        Parameters
        ----------
        flow : Flow
            The flow to schedule.

        Returns
        -------
        List[str] or None
            Assigned path, or None if src/dst not in topology.
        """
        if flow.size_bytes < self.elephant_threshold:
            # ── Mice: standard ECMP ───────────────────────────────────────────
            path = self._ecmp.schedule_flow(flow)
            if path:
                self._mice_count += 1
            return path

        # ── Elephant: Global First Fit ────────────────────────────────────────
        candidates = self.get_candidate_paths(flow.src_ip, flow.dst_ip)
        if not candidates:
            return None

        path = self._global_first_fit(flow, candidates)
        if path:
            self._load.assign(flow, path)
            self._elephant_count += 1
        return path

    # ── GFF core ─────────────────────────────────────────────────────────────

    def _global_first_fit(
        self, flow: Flow, candidates: List[List[str]]
    ) -> Optional[List[str]]:
        """
        Select the path with the smallest current byte-load.

        This implements the core of Hedera's Global First Fit algorithm.
        In the original paper, demands are sorted globally before assignment;
        here we select the best path for a single flow independently (the
        sorting happens in ``reschedule_elephants()`` for batch operations).

        Parameters
        ----------
        flow : Flow
            The elephant flow requesting a path.
        candidates : List[List[str]]
            ECMP candidate paths (all have equal length / hop count).

        Returns
        -------
        List[str]
            The least-loaded candidate path.
        """
        best_path = self._load.least_loaded_path(candidates)
        self._log.debug(
            "GFF: flow %s (%d B) → path[load=%d B] %s→%s",
            flow.flow_id, flow.size_bytes,
            self._load.get_load_bytes(best_path) if best_path else -1,
            flow.src_ip, flow.dst_ip,
        )
        return best_path

    # ── Batch global rescheduling ─────────────────────────────────────────────

    def reschedule_elephants(self, elephant_flows: List[Flow]) -> Dict[str, Optional[List[str]]]:
        """
        Re-run Global First Fit across a batch of elephant flows.

        This simulates Hedera's periodic global rescheduling step:

          1. Release all current elephant path assignments (reset load state).
          2. Sort flows by demand (size_bytes) descending — largest first.
          3. Assign each flow to the least-loaded path via GFF.

        This should be called by the controller every ~300 ms (or whenever
        congestion is detected) with the current active elephant-flow set.

        Parameters
        ----------
        elephant_flows : List[Flow]
            All currently active elephant flows to reschedule.

        Returns
        -------
        Dict[str, Optional[List[str]]]
            Maps flow_id → newly assigned path (or None on failure).
        """
        # 1. Release all current elephant assignments.
        for f in elephant_flows:
            self._load.release(f.flow_id)

        # 2. Sort by demand descending (largest flows first — they are
        #    hardest to place and benefit most from uncongested paths).
        sorted_flows = sorted(elephant_flows, key=lambda f: f.size_bytes, reverse=True)

        # 3. GFF assignment.
        results: Dict[str, Optional[List[str]]] = {}
        for flow in sorted_flows:
            candidates = self.get_candidate_paths(flow.src_ip, flow.dst_ip)
            if not candidates:
                results[flow.flow_id] = None
                continue
            path = self._global_first_fit(flow, candidates)
            if path:
                self._load.assign(flow, path)
                flow.assigned_path = path
                self._reschedule_count += 1
            results[flow.flow_id] = path

        self._log.info(
            "reschedule_elephants: %d/%d flows placed",
            sum(1 for p in results.values() if p), len(elephant_flows),
        )
        return results

    # ── Flow lifecycle ────────────────────────────────────────────────────────

    def release_flow(self, flow_id: str) -> bool:
        """
        Signal that a flow has completed; remove its load contribution.

        Parameters
        ----------
        flow_id : str
            ID of the completed flow.

        Returns
        -------
        bool
            True if the flow was found and released.
        """
        released = self._load.release(flow_id)
        if released:
            self._log.debug("Flow %s released from load tracker", flow_id)
        return released

    # ── Statistics / analysis ─────────────────────────────────────────────────

    @property
    def mice_count(self) -> int:
        """Number of mice flows scheduled since last reset."""
        return self._mice_count

    @property
    def elephant_count(self) -> int:
        """Number of elephant flows scheduled since last reset."""
        return self._elephant_count

    @property
    def reschedule_count(self) -> int:
        """Total number of flows re-placed by reschedule_elephants()."""
        return self._reschedule_count

    def path_loads(self) -> Dict[tuple, int]:
        """Return current byte-load per path tuple."""
        return self._load.all_loads()

    def path_utilisation(self, path: List[str]) -> float:
        """Return the utilisation fraction (0.0–∞) for a specific path."""
        return self._load.get_utilisation(path)

    def load_balance_ratio(self) -> Optional[float]:
        """
        min_load / max_load across all currently loaded paths.

        1.0 = perfect balance, near 0.0 = severe imbalance.
        Returns None if no elephant flows are active.
        """
        loads = self._load.all_loads()
        if not loads:
            return None
        values = list(loads.values())
        mn, mx = min(values), max(values)
        if mx == 0:
            return None
        return mn / mx

    def hedera_stats(self) -> str:
        """Return a formatted Hedera-specific statistics block."""
        loads = self.path_loads()
        ratio = self.load_balance_ratio()
        lines = [
            "=== Hedera Statistics ===",
            f"Mice flows (ECMP)   : {self._mice_count}",
            f"Elephant flows (GFF): {self._elephant_count}",
            f"Reschedule events   : {self._reschedule_count}",
            f"Active elephant paths: {len(loads)}",
            f"Load balance ratio  : "
            + (f"{ratio:.3f}" if ratio is not None else "N/A"),
        ]
        return "\n".join(lines)

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self._mice_count = 0
        self._elephant_count = 0
        self._reschedule_count = 0
        self._load.reset()
        self._ecmp.reset_metrics()

    def __repr__(self) -> str:
        return (
            f"HederaScheduler("
            f"k={self.topology.k}, "
            f"mice={self._mice_count}, "
            f"elephants={self._elephant_count})"
        )
