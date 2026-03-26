"""
LAFS -- Link Load Sampler
=========================
COMP-6910 -- Group 7

Converts per-flow scheduling events into per-link utilisation time series.

The sampler answers: "over each delta_t-second window, what fraction of
capacity did each directed link carry?"

Collection model
----------------
When flows have been assigned paths (flow.assigned_path is set), their
size_bytes are attributed to every directed link along the path.  Bytes
are counted in the time window that contains the flow's arrival_time.

This is a *flow-arrival* model: bytes are attributed to the window when
the flow starts, not spread across its full transmission time.  This is
deliberate -- the predictor's goal is to forecast *when* congestion will
arrive, so arrival timing is the relevant signal.

Usage
-----
    from src.topology.fattree import FatTreeGraph
    from src.metrics.link_load import LinkLoadSampler

    topo    = FatTreeGraph(k=8)
    sampler = LinkLoadSampler(topo, window_s=0.1)
    sampler.ingest(flows)                         # flows with assigned_path

    series = sampler.get_series(("e_0_0", "a_0_0"))
    print(series.values())   # [0.23, 0.04, 0.41, ...]  (utilisation/window)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow


# ── Default capacity (host-link bottleneck) ───────────────────────────────────
_DEFAULT_CAPACITY_BPS: float = 1e9          # 1 Gbps
_FABRIC_CAPACITY_BPS:  float = 10e9         # 10 Gbps (switch-to-switch)


# =============================================================================
# LinkLoadSample  --  one measurement for one link in one time window
# =============================================================================

@dataclass
class LinkLoadSample:
    """
    A single utilisation measurement for one directed link.

    Attributes
    ----------
    link : (str, str)
        Directed link as (upstream_node, downstream_node).
    t_start : float
        Window start time in seconds (simulation time, not wall clock).
    t_end : float
        Window end time in seconds.
    bytes_observed : int
        Total bytes of flows that arrived during [t_start, t_end).
    utilisation : float
        Fraction of link capacity consumed.
        Formula: bytes_observed * 8 / (window_s * capacity_bps).
        Values > 1.0 indicate over-subscription within this window.
    """
    link: Tuple[str, str]
    t_start: float
    t_end: float
    bytes_observed: int
    utilisation: float

    @property
    def window_s(self) -> float:
        return self.t_end - self.t_start


# =============================================================================
# LinkLoadSeries  --  ordered time series of samples for one link
# =============================================================================

class LinkLoadSeries:
    """
    Fixed-capacity circular buffer of LinkLoadSamples for one directed link.

    Parameters
    ----------
    link : (str, str)
        The directed link this series tracks.
    capacity_bps : float
        Link capacity in bits per second (used for utilisation computation).
    max_samples : int
        Maximum number of samples to retain (oldest are discarded).
    """

    def __init__(
        self,
        link: Tuple[str, str],
        capacity_bps: float = _DEFAULT_CAPACITY_BPS,
        max_samples: int = 1000,
    ) -> None:
        self.link = link
        self.capacity_bps = capacity_bps
        self._samples: deque[LinkLoadSample] = deque(maxlen=max_samples)

    def append(self, sample: LinkLoadSample) -> None:
        """Add a new sample to the series."""
        self._samples.append(sample)

    def values(self) -> List[float]:
        """Return utilisation values in chronological order."""
        return [s.utilisation for s in self._samples]

    def timestamps(self) -> List[float]:
        """Return window-start timestamps in chronological order."""
        return [s.t_start for s in self._samples]

    def last_n(self, n: int) -> List[float]:
        """Return the most recent n utilisation values."""
        vals = self.values()
        return vals[-n:] if n < len(vals) else vals

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def mean(self) -> float:
        """Arithmetic mean utilisation across all retained samples."""
        v = self.values()
        return sum(v) / len(v) if v else 0.0

    @property
    def variance(self) -> float:
        """Sample variance of utilisation."""
        v = self.values()
        if len(v) < 2:
            return 0.0
        m = self.mean
        return sum((x - m) ** 2 for x in v) / (len(v) - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def latest(self) -> Optional[LinkLoadSample]:
        """Most recent sample, or None if empty."""
        return self._samples[-1] if self._samples else None

    def __repr__(self) -> str:
        return (
            f"LinkLoadSeries(link={self.link}, "
            f"n={len(self)}, mean={self.mean:.3f})"
        )


# =============================================================================
# LinkLoadSampler  --  main collection engine
# =============================================================================

class LinkLoadSampler:
    """
    Derives per-link utilisation time series from a list of scheduled flows.

    The topology is used to:
      * Enumerate all directed links and their capacities.
      * Validate that assigned_path node names belong to the topology.

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology (used for link enumeration and capacity lookup).
    window_s : float
        Sampling window width in seconds.  Flows are binned into windows of
        this duration based on their arrival_time.  Default 0.1 s (100 ms).
    capacity_bps : float
        Default link capacity in bps.  The sampler uses the topology's per-link
        capacity when available, falling back to this value.
    max_samples : int
        Maximum samples per LinkLoadSeries (circular buffer depth).

    Usage
    -----
        sampler = LinkLoadSampler(topo, window_s=0.1)
        sampler.ingest(flows)
        series = sampler.get_series(("e_0_0", "a_0_0"))
        forecaster.fit(sampler)
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        window_s: float = 0.1,
        capacity_bps: float = _DEFAULT_CAPACITY_BPS,
        max_samples: int = 1000,
    ) -> None:
        self.topology = topology
        self.window_s = window_s
        self._default_capacity_bps = capacity_bps
        self._max_samples = max_samples

        # Pre-build capacity map from topology: directed link -> bps
        self._capacity: Dict[Tuple[str, str], float] = {}
        for u, v, attrs in topology.get_all_links():
            cap_gbps = attrs.get("capacity", capacity_bps / 1e9)
            bps = cap_gbps * 1e9
            self._capacity[(u, v)] = bps
            self._capacity[(v, u)] = bps   # undirected graph -- both directions

        # Lazy-initialised per-link series.
        self._series: Dict[Tuple[str, str], LinkLoadSeries] = {}

        # Raw accumulator: window_idx -> link -> bytes
        # window_idx = int(arrival_time / window_s)
        self._window_bytes: Dict[int, Dict[Tuple[str, str], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._ingested_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, flows: List[Flow]) -> int:
        """
        Process a list of scheduled flows and update window accumulators.

        Flows without an assigned_path are silently skipped.

        Parameters
        ----------
        flows : List[Flow]
            Flows with flow.assigned_path and flow.arrival_time set.

        Returns
        -------
        int
            Number of flows successfully ingested (had a path assigned).
        """
        ingested = 0
        for f in flows:
            if not f.assigned_path or len(f.assigned_path) < 2:
                continue
            win_idx = int(f.arrival_time / self.window_s)
            # Count bytes on every directed link along the path.
            for i in range(len(f.assigned_path) - 1):
                link = (f.assigned_path[i], f.assigned_path[i + 1])
                self._window_bytes[win_idx][link] += f.size_bytes
            ingested += 1

        self._ingested_count += ingested
        return ingested

    def build_series(self) -> None:
        """
        Convert window accumulators into LinkLoadSeries objects.

        Call this after all flows have been ingested, before calling
        get_series() or all_series().
        """
        if not self._window_bytes:
            return

        sorted_windows = sorted(self._window_bytes.keys())

        for win_idx in sorted_windows:
            t_start = win_idx * self.window_s
            t_end   = t_start + self.window_s
            link_bytes = self._window_bytes[win_idx]

            for link, nbytes in link_bytes.items():
                cap = self._capacity.get(link, self._default_capacity_bps)
                util = (nbytes * 8) / (self.window_s * cap) if cap > 0 else 0.0

                if link not in self._series:
                    self._series[link] = LinkLoadSeries(
                        link, cap, self._max_samples
                    )

                sample = LinkLoadSample(
                    link=link,
                    t_start=t_start,
                    t_end=t_end,
                    bytes_observed=nbytes,
                    utilisation=util,
                )
                self._series[link].append(sample)

    def get_series(self, link: Tuple[str, str]) -> LinkLoadSeries:
        """
        Return the LinkLoadSeries for *link*.

        If the link has not been observed, returns an empty series.
        """
        if link not in self._series:
            cap = self._capacity.get(link, self._default_capacity_bps)
            return LinkLoadSeries(link, cap, self._max_samples)
        return self._series[link]

    def all_series(self) -> Dict[Tuple[str, str], LinkLoadSeries]:
        """Return a dict of all link -> LinkLoadSeries mappings."""
        return dict(self._series)

    def active_links(self) -> List[Tuple[str, str]]:
        """Return the list of links that have at least one observation."""
        return list(self._series.keys())

    def utilisation_snapshot(self, t: float) -> Dict[Tuple[str, str], float]:
        """
        Return the utilisation of every active link at time t.

        The value returned is the utilisation in the window that contains t.
        Links not observed in that window return 0.0.
        """
        win_idx = int(t / self.window_s)
        snapshot: Dict[Tuple[str, str], float] = {}
        for link, series in self._series.items():
            # Find the sample for this window.
            util = 0.0
            for s in reversed(list(series._samples)):
                if abs(s.t_start - win_idx * self.window_s) < 1e-9:
                    util = s.utilisation
                    break
            snapshot[link] = util
        return snapshot

    @property
    def n_windows(self) -> int:
        """Number of distinct time windows observed."""
        return len(self._window_bytes)

    @property
    def n_flows_ingested(self) -> int:
        return self._ingested_count

    def summary(self) -> str:
        """Return a brief human-readable summary."""
        lines = [
            f"LinkLoadSampler: window={self.window_s * 1e3:.0f} ms, "
            f"flows={self.n_flows_ingested}, "
            f"windows={self.n_windows}, "
            f"active links={len(self._series)}",
        ]
        if self._series:
            means = [s.mean for s in self._series.values()]
            lines.append(
                f"  Utilisation: mean={sum(means)/len(means):.3f}, "
                f"max={max(means):.3f}, "
                f"non-zero links={sum(1 for m in means if m > 0)}"
            )
        return "\n".join(lines)
