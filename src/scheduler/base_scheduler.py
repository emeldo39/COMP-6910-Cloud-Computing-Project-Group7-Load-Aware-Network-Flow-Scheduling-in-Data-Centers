"""
LAFS — BaseScheduler
====================
COMP 6910 — Group 7

Abstract base class that every LAFS scheduler must extend.

Responsibilities
----------------
* Provide a uniform schedule_flow(flow) / schedule_flows(flows) API so that
  ECMP, Hedera, CONGA, and the final LAFS scheduler are all interchangeable.
* Collect per-scheduler statistics (flows, bytes, per-path counts, latency).
* Expose logging via the Python standard library so output verbosity is
  controlled by the caller without touching scheduler code.
* Validate topology at construction time.

Subclass contract
-----------------
    class MyScheduler(BaseScheduler):
        @property
        def name(self) -> str: return "my_scheduler"

        def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
            # Return an ordered list of node names [src_host, ..., dst_host]
            # or None if no feasible path exists.
            ...
"""

from __future__ import annotations

import abc
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

# Forward-declared to avoid a circular import at module load time.
# The actual import happens at runtime when BaseScheduler.__init__ runs.
from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow


# ── Module-level logger (scheduler subclasses get child loggers) ──────────────
_log = logging.getLogger("lafs.scheduler")


# =============================================================================
# SchedulerMetrics
# =============================================================================
class SchedulerMetrics:
    """
    Lightweight statistics container populated by BaseScheduler during operation.

    All public attributes are safe to read at any time; they are updated
    atomically within the GIL (no explicit locking needed for CPython
    single-thread use as in our experiments).

    Attributes
    ----------
    flows_scheduled : int
        Total number of flows that received a valid path assignment.
    flows_failed : int
        Total number of flows for which no path could be found.
    total_bytes_scheduled : int
        Sum of size_bytes across all successfully scheduled flows.
    mice_flows, medium_flows, elephant_flows : int
        Breakdown by flow type (<100 KB / 100 KB–1 MB / ≥1 MB).
    path_usage : Dict[tuple, int]
        Maps (node, …, node) path tuples to the number of flows sent on them.
    _schedule_latencies : List[float]
        Raw per-flow scheduling decision latencies in seconds (for percentile
        computation).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Zero all counters and restart the uptime clock."""
        self.flows_scheduled: int = 0
        self.flows_failed: int = 0
        self.total_bytes_scheduled: int = 0
        self.mice_flows: int = 0
        self.medium_flows: int = 0
        self.elephant_flows: int = 0
        # Key: tuple of node names along the path; Value: number of flows.
        self.path_usage: Dict[tuple, int] = defaultdict(int)
        self._schedule_latencies: List[float] = []
        self._start_time: float = time.monotonic()

    # ── Recording helpers (called by BaseScheduler.schedule_flows) ────────────

    def record_scheduled(self, flow: Flow, path: List[str], latency_s: float) -> None:
        """Record a successfully scheduled flow."""
        self.flows_scheduled += 1
        self.total_bytes_scheduled += flow.size_bytes
        self._schedule_latencies.append(latency_s)
        if path:
            self.path_usage[tuple(path)] += 1
        if flow.is_mice:
            self.mice_flows += 1
        elif flow.is_elephant:
            self.elephant_flows += 1
        else:
            self.medium_flows += 1

    def record_failure(self) -> None:
        """Record a scheduling failure (no path available)."""
        self.flows_failed += 1

    # ── Derived statistics ────────────────────────────────────────────────────

    @property
    def avg_latency_us(self) -> float:
        """Mean scheduling decision time in microseconds."""
        if not self._schedule_latencies:
            return 0.0
        return (sum(self._schedule_latencies) / len(self._schedule_latencies)) * 1e6

    @property
    def p99_latency_us(self) -> float:
        """99th-percentile scheduling decision time in microseconds."""
        if not self._schedule_latencies:
            return 0.0
        sorted_lat = sorted(self._schedule_latencies)
        idx = max(0, int(len(sorted_lat) * 0.99) - 1)
        return sorted_lat[idx] * 1e6

    @property
    def unique_paths_used(self) -> int:
        """Number of distinct paths that have had at least one flow assigned."""
        return len(self.path_usage)

    @property
    def uptime_s(self) -> float:
        """Seconds since this metrics instance was created or last reset."""
        return time.monotonic() - self._start_time

    def path_distribution(self) -> Dict[tuple, float]:
        """Return per-path fraction of total scheduled flows (0.0–1.0)."""
        total = self.flows_scheduled
        if total == 0:
            return {}
        return {path: count / total for path, count in self.path_usage.items()}

    def summary(self) -> str:
        """Return a human-readable multi-line statistics report."""
        lines = [
            f"Flows scheduled : {self.flows_scheduled}",
            f"  mice          : {self.mice_flows}",
            f"  medium        : {self.medium_flows}",
            f"  elephant      : {self.elephant_flows}",
            f"Flows failed    : {self.flows_failed}",
            f"Total bytes     : {self.total_bytes_scheduled / 1e6:.2f} MB",
            f"Avg latency     : {self.avg_latency_us:.2f} µs",
            f"P99 latency     : {self.p99_latency_us:.2f} µs",
            f"Unique paths    : {self.unique_paths_used}",
            f"Uptime          : {self.uptime_s:.1f} s",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SchedulerMetrics(scheduled={self.flows_scheduled}, "
            f"failed={self.flows_failed}, "
            f"paths={self.unique_paths_used})"
        )


# =============================================================================
# BaseScheduler
# =============================================================================
class BaseScheduler(abc.ABC):
    """
    Abstract base class for all LAFS flow schedulers.

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology used for candidate-path lookup.  Must be a
        FatTreeGraph so that node naming, IP lookups, and ECMP paths are
        available to every subclass.

    Subclassing
    -----------
    Override the two abstract members::

        @property
        def name(self) -> str:
            return "my_scheduler"   # used in log messages and reports

        def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
            # Your path-selection logic here.
            # Return a list of node names (src → … → dst) or None.
            ...

    Then call super().__init__(topology) from your __init__ to get:
      * self.topology   — the FatTreeGraph
      * self.metrics    — SchedulerMetrics (auto-populated by schedule_flows)
      * self._log       — a Logger named "lafs.scheduler.<name>"
    """

    def __init__(self, topology: FatTreeGraph) -> None:
        if not isinstance(topology, FatTreeGraph):
            raise TypeError(
                f"topology must be a FatTreeGraph, got {type(topology).__name__}"
            )
        self.topology = topology
        self.metrics = SchedulerMetrics()
        # Child logger: "lafs.scheduler.ecmp", "lafs.scheduler.hedera", etc.
        self._log = logging.getLogger(f"lafs.scheduler.{self.name}")
        self._log.debug(
            "Scheduler '%s' initialised on k=%d Fat-tree "
            "(%d hosts, %d switches)",
            self.name, topology.k, topology.n_hosts, topology.n_switches,
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, lowercase scheduler identifier (used in logs and filenames)."""

    @abc.abstractmethod
    def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
        """
        Assign *flow* to a network path and return it.

        Parameters
        ----------
        flow : Flow
            The flow requesting path assignment.

        Returns
        -------
        List[str] or None
            Ordered list of node names forming the path from source host to
            destination host (inclusive of both endpoints), e.g.::

                ["h_0_0_0", "e_0_0", "a_0_1", "c_1_2", "a_2_1", "e_2_0", "h_2_0_0"]

            Return ``None`` when no feasible path exists (e.g. src/dst IP not
            in topology).
        """

    # ── Concrete helpers (available to all subclasses) ────────────────────────

    def schedule_flows(self, flows: List[Flow]) -> Dict[str, Optional[List[str]]]:
        """
        Schedule a batch of flows, updating metrics after each decision.

        For each flow the method:
          1. Calls ``schedule_flow(flow)`` (implemented by subclass).
          2. On success: stores path on ``flow.assigned_path``, stamps
             ``flow.schedule_time``, updates metrics.
          3. On failure / exception: records a failure; path is ``None``.

        Parameters
        ----------
        flows : List[Flow]
            Flows to schedule (order preserved in result).

        Returns
        -------
        Dict[str, Optional[List[str]]]
            Maps each ``flow.flow_id`` to its assigned path or ``None``.
        """
        results: Dict[str, Optional[List[str]]] = {}

        for flow in flows:
            t0 = time.perf_counter()
            try:
                path = self.schedule_flow(flow)
            except Exception as exc:
                self._log.warning(
                    "schedule_flow raised for flow %s: %s", flow.flow_id, exc
                )
                path = None
            latency = time.perf_counter() - t0

            if path is not None:
                self.metrics.record_scheduled(flow, path, latency)
                flow.assigned_path = path
                flow.schedule_time = time.time()
            else:
                self.metrics.record_failure()
                self._log.debug(
                    "No path for flow %s (%s → %s)",
                    flow.flow_id, flow.src_ip, flow.dst_ip,
                )

            results[flow.flow_id] = path

        n_ok = sum(1 for p in results.values() if p is not None)
        self._log.info(
            "Batch complete: %d/%d flows scheduled (%.1f%%)",
            n_ok, len(flows), 100.0 * n_ok / max(len(flows), 1),
        )
        return results

    def get_candidate_paths(self, src_ip: str, dst_ip: str) -> List[List[str]]:
        """
        Return all shortest (ECMP) paths between two host IPs.

        Translates IPs to node names via the topology's IP→node map, then
        delegates to ``FatTreeGraph.get_paths()``.

        Parameters
        ----------
        src_ip, dst_ip : str
            Source and destination IP addresses (dotted-decimal).

        Returns
        -------
        List[List[str]]
            Possibly empty list of paths; each path is a list of node names.
        """
        try:
            src_node = self.topology.node_for_ip(src_ip)
        except KeyError:
            self._log.debug("src IP %s not found in topology", src_ip)
            return []
        try:
            dst_node = self.topology.node_for_ip(dst_ip)
        except KeyError:
            self._log.debug("dst IP %s not found in topology", dst_ip)
            return []
        return self.topology.get_paths(src_node, dst_node)

    def reset_metrics(self) -> None:
        """Clear all collected statistics and restart the uptime clock."""
        self.metrics.reset()
        self._log.debug("Metrics reset")

    def report(self) -> str:
        """Return a formatted statistics report string."""
        return f"=== {self.name.upper()} Scheduler Report ===\n{self.metrics.summary()}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"k={self.topology.k}, "
            f"scheduled={self.metrics.flows_scheduled})"
        )
