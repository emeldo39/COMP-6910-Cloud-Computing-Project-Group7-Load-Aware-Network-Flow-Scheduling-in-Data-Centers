"""
LAFS — Workload Runner & Statistics
=====================================
COMP 6910 — Group 7

Unified entry point for workload generation and flow statistics.

WorkloadRunner orchestrates one or more workload generators and:
  * Merges their outputs into a single time-ordered flow trace.
  * Applies a global load cap by optionally sub-sampling flows.
  * Computes WorkloadStats (size distribution, arrival rate, mix).

WorkloadStats provides the summary statistics needed for the evaluation
section of the LAFS report (FCT distribution inputs, load fractions,
tenant breakdown).

Usage
-----
    from src.topology.fattree import FatTreeGraph
    from src.workload.runner import WorkloadRunner, WorkloadConfig

    topo = FatTreeGraph(k=8)
    cfg  = WorkloadConfig(
        workload_types=["facebook", "allreduce"],
        n_flows=5_000,
        load_fraction=0.7,
        n_tenants=4,
        seed=42,
    )
    runner = WorkloadRunner(topo, cfg)
    flows  = runner.generate()
    stats  = runner.stats(flows)
    print(stats.summary())
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow, MICE_THRESHOLD_BYTES, ELEPHANT_THRESHOLD_BYTES
from src.workload.facebook_websearch import (
    FacebookWebSearchGenerator,
    FacebookWebSearchConfig,
)
from src.workload.allreduce import AllReduceGenerator, AllReduceConfig
from src.workload.microservice import MicroserviceRPCGenerator, MicroserviceConfig


# ── Valid workload type tokens ────────────────────────────────────────────────
WORKLOAD_TYPES = frozenset({"facebook", "allreduce", "microservice", "mixed"})


# =============================================================================
# WorkloadConfig
# =============================================================================
@dataclass
class WorkloadConfig:
    """
    Top-level configuration for WorkloadRunner.

    Parameters
    ----------
    workload_types : List[str]
        One or more of: 'facebook', 'allreduce', 'microservice'.
        Use 'mixed' as a shorthand for all three combined.
    n_flows : int
        Target total flow count across all generators.
        Each generator receives a proportional share.
    load_fraction : float
        Target fraction of host-link capacity (0.3 – 0.9).
        Passed to each sub-generator; also used by the runner to
        sub-sample the merged trace if necessary.
    n_tenants : int
        Number of tenants (4–16 per proposal §4).
    duration_s : float
        Simulation window length in seconds.
    seed : int
        Master RNG seed. Sub-generators receive seed + offset to ensure
        their outputs are independent yet reproducible.
    facebook_weight : float
        Fraction of n_flows to allocate to Facebook web-search (0–1).
    allreduce_weight : float
        Fraction of n_flows to allocate to AllReduce.
    microservice_weight : float
        Fraction of n_flows to allocate to microservice RPC chains.
        Weights are normalised to sum to 1.0 automatically.
    n_workers : int
        Number of AllReduce workers (passed to AllReduceConfig).
    model_preset : str
        AllReduce model size preset (passed to AllReduceConfig).
    ms_graph_type : str
        Microservice graph topology: 'chain', 'fan_out', or 'mixed'.
    ms_fan_out : int
        Fan-out degree for microservice generator.
    """
    workload_types: List[str] = field(default_factory=lambda: ["facebook"])
    n_flows: int = 1_000
    load_fraction: float = 0.5
    n_tenants: int = 1
    duration_s: float = 10.0
    seed: int = 42
    # Per-generator weights (normalised automatically).
    facebook_weight: float = 1.0
    allreduce_weight: float = 1.0
    microservice_weight: float = 1.0
    # Sub-generator specific params.
    n_workers: int = 4
    model_preset: str = "resnet50"
    ms_graph_type: str = "mixed"
    ms_fan_out: int = 4

    def __post_init__(self) -> None:
        # Expand 'mixed' shorthand.
        expanded: List[str] = []
        for wt in self.workload_types:
            if wt == "mixed":
                expanded.extend(["facebook", "allreduce", "microservice"])
            elif wt in WORKLOAD_TYPES - {"mixed"}:
                expanded.append(wt)
            else:
                raise ValueError(
                    f"Unknown workload_type {wt!r}. "
                    f"Choose from {sorted(WORKLOAD_TYPES)}"
                )
        self.workload_types = list(dict.fromkeys(expanded))  # deduplicate

        if not (0.0 < self.load_fraction <= 1.0):
            raise ValueError(f"load_fraction must be in (0, 1], got {self.load_fraction}")
        if self.n_flows < 1:
            raise ValueError("n_flows must be >= 1")
        if self.n_tenants < 1:
            raise ValueError("n_tenants must be >= 1")

    def _flow_counts(self) -> Dict[str, int]:
        """
        Allocate n_flows proportionally across active workload types.
        Returns a dict {workload_type: n_flows}.
        """
        weights: Dict[str, float] = {
            "facebook":    self.facebook_weight,
            "allreduce":   self.allreduce_weight,
            "microservice": self.microservice_weight,
        }
        active = {wt: weights[wt] for wt in self.workload_types}
        total_w = sum(active.values())
        if total_w == 0:
            total_w = 1.0

        counts: Dict[str, int] = {}
        remaining = self.n_flows
        types = list(active.keys())
        for i, wt in enumerate(types):
            if i == len(types) - 1:
                counts[wt] = remaining
            else:
                c = max(1, round(self.n_flows * active[wt] / total_w))
                counts[wt] = c
                remaining -= c
        return counts


# =============================================================================
# WorkloadStats
# =============================================================================
@dataclass
class WorkloadStats:
    """
    Summary statistics over a generated flow trace.

    Attributes
    ----------
    n_flows : int
        Total number of flows.
    mice_count, medium_count, elephant_count : int
        Flow counts by size class (<100 KB / 100 KB–10 MB / >10 MB).
    mean_size_bytes : float
        Arithmetic mean flow size.
    median_size_bytes : float
        Median flow size (50th percentile).
    p90_size_bytes, p99_size_bytes : float
        90th and 99th percentile flow sizes.
    arrival_rate : float
        Mean flow arrival rate (flows/second).
    duration_s : float
        Total trace duration (last_arrival - first_arrival).
    workload_mix : Dict[str, int]
        Counts by workload source (prefix of flow_id: 'fb', 'ar', 'rpc').
    tenant_counts : Dict[int, int]
        Flow counts per tenant (extracted from flow_id prefix 't{i}_').
    size_bytes_list : List[int]
        Raw list of all flow sizes (for histogram / CDF plotting).
    """
    n_flows: int = 0
    mice_count: int = 0
    medium_count: int = 0
    elephant_count: int = 0
    mean_size_bytes: float = 0.0
    median_size_bytes: float = 0.0
    p90_size_bytes: float = 0.0
    p99_size_bytes: float = 0.0
    arrival_rate: float = 0.0
    duration_s: float = 0.0
    workload_mix: Dict[str, int] = field(default_factory=dict)
    tenant_counts: Dict[int, int] = field(default_factory=dict)
    size_bytes_list: List[int] = field(default_factory=list, repr=False)

    # Proposal metrics.
    mice_fraction: float = 0.0
    elephant_fraction: float = 0.0

    # Jain's fairness index across tenants.
    jains_index: float = 0.0

    def summary(self) -> str:
        """Return a multi-line human-readable statistics report."""
        lines = [
            "=== Workload Statistics ===",
            f"Total flows       : {self.n_flows:,}",
            f"Mice (<100KB)     : {self.mice_count:,} ({self.mice_fraction*100:.1f}%)",
            f"Medium            : {self.medium_count:,}",
            f"Elephant (>10MB)  : {self.elephant_count:,} ({self.elephant_fraction*100:.1f}%)",
            f"Mean size         : {self.mean_size_bytes/1e3:.1f} KB",
            f"Median size       : {self.median_size_bytes/1e3:.1f} KB",
            f"P90 size          : {self.p90_size_bytes/1e3:.1f} KB",
            f"P99 size          : {self.p99_size_bytes/1e6:.2f} MB",
            f"Arrival rate      : {self.arrival_rate:.1f} flows/s",
            f"Duration          : {self.duration_s:.2f} s",
            f"Workload mix      : {dict(sorted(self.workload_mix.items()))}",
            f"Tenants           : {len(self.tenant_counts)}",
            f"Jain's index      : {self.jains_index:.4f}",
        ]
        return "\n".join(lines)


def _jains_fairness(counts: Dict) -> float:
    """Compute Jain's fairness index: (Σx)² / (n · Σx²)."""
    values = list(counts.values())
    if not values or len(values) == 1:
        return 1.0
    n = len(values)
    s = sum(values)
    sq = sum(v * v for v in values)
    if sq == 0:
        return 1.0
    return (s * s) / (n * sq)


# =============================================================================
# WorkloadRunner
# =============================================================================
class WorkloadRunner:
    """
    Orchestrates multiple workload generators and merges their outputs.

    Each sub-generator produces flows independently with its own seed.
    The runner merges them into a single chronologically ordered trace.

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology.
    config : WorkloadConfig
        Generation parameters.

    Examples
    --------
    >>> topo = FatTreeGraph(k=4)
    >>> cfg  = WorkloadConfig(workload_types=['facebook', 'allreduce'],
    ...                       n_flows=200, load_fraction=0.6, seed=7)
    >>> runner = WorkloadRunner(topo, cfg)
    >>> flows  = runner.generate()
    >>> len(flows)
    200
    >>> stats = runner.compute_stats(flows)
    >>> stats.n_flows
    200
    """

    def __init__(self, topology: FatTreeGraph, config: WorkloadConfig) -> None:
        self.topology = topology
        self.config = config
        self._rng = random.Random(config.seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> List[Flow]:
        """
        Run all configured sub-generators and return merged, time-sorted flows.

        If the merged count exceeds ``config.n_flows``, the trace is randomly
        sub-sampled to exactly that count while preserving time ordering.

        Returns
        -------
        List[Flow]
            Chronologically ordered flow list of length <= ``config.n_flows``.
        """
        cfg = self.config
        flow_counts = cfg._flow_counts()
        all_flows: List[Flow] = []

        for wt in cfg.workload_types:
            n = flow_counts[wt]
            sub_seed = cfg.seed + hash(wt) % 100_000
            gen_flows = self._run_generator(wt, n, sub_seed)
            all_flows.extend(gen_flows)

        # Sort by arrival time.
        all_flows.sort(key=lambda f: f.arrival_time)

        # Sub-sample if over-produced.
        if len(all_flows) > cfg.n_flows:
            # Reservoir-sample preserving time order.
            rng = random.Random(cfg.seed + 1)
            indices = sorted(rng.sample(range(len(all_flows)), cfg.n_flows))
            all_flows = [all_flows[i] for i in indices]

        return all_flows

    def compute_stats(self, flows: List[Flow]) -> WorkloadStats:
        """
        Compute WorkloadStats over a list of flows.

        Parameters
        ----------
        flows : List[Flow]
            The flow trace to analyse.

        Returns
        -------
        WorkloadStats
        """
        if not flows:
            return WorkloadStats()

        sizes = sorted(f.size_bytes for f in flows)
        n = len(sizes)

        # Size class counts (proposal thresholds: mice <100KB, elephant >10MB).
        mice_count = sum(1 for s in sizes if s < MICE_THRESHOLD_BYTES)
        elephant_count = sum(1 for s in sizes if s >= 10_000_000)
        medium_count = n - mice_count - elephant_count

        # Percentiles.
        def pct(p: float) -> float:
            idx = min(n - 1, int(n * p))
            return float(sizes[idx])

        # Arrival rate / duration.
        arrivals = sorted(f.arrival_time for f in flows)
        duration = max(0.0, arrivals[-1] - arrivals[0])
        rate = n / duration if duration > 0 else float(n)

        # Workload mix (by flow_id prefix).
        mix: Dict[str, int] = {}
        for f in flows:
            parts = f.flow_id.split("_")
            # Identify workload: 'fb', 'ar', 'rpc', 't{n}' patterns.
            key = "unknown"
            if "fb" in parts:
                key = "facebook"
            elif "ar" in parts or "ring" in parts or "ps" in parts or "pipe" in parts:
                key = "allreduce"
            elif "rpc" in parts:
                key = "microservice"
            mix[key] = mix.get(key, 0) + 1

        # Tenant counts.
        tenant_counts: Dict[int, int] = {}
        for f in flows:
            parts = f.flow_id.split("_")
            if parts[0].startswith("t") and parts[0][1:].isdigit():
                tid = int(parts[0][1:])
                tenant_counts[tid] = tenant_counts.get(tid, 0) + 1
            else:
                tenant_counts[-1] = tenant_counts.get(-1, 0) + 1

        jains = _jains_fairness(tenant_counts)

        return WorkloadStats(
            n_flows=n,
            mice_count=mice_count,
            medium_count=medium_count,
            elephant_count=elephant_count,
            mean_size_bytes=sum(sizes) / n,
            median_size_bytes=pct(0.50),
            p90_size_bytes=pct(0.90),
            p99_size_bytes=pct(0.99),
            arrival_rate=rate,
            duration_s=duration,
            workload_mix=mix,
            tenant_counts=tenant_counts,
            size_bytes_list=sizes,
            mice_fraction=mice_count / n,
            elephant_fraction=elephant_count / n,
            jains_index=jains,
        )

    # ── Sub-generator dispatch ────────────────────────────────────────────────

    def _run_generator(self, wt: str, n_flows: int, seed: int) -> List[Flow]:
        """Instantiate and run a single sub-generator."""
        cfg = self.config
        topo = self.topology

        if wt == "facebook":
            sub_cfg = FacebookWebSearchConfig(
                n_flows=n_flows,
                load_fraction=cfg.load_fraction,
                n_tenants=cfg.n_tenants,
                duration_s=cfg.duration_s,
                seed=seed,
            )
            return FacebookWebSearchGenerator(topo, sub_cfg).generate()

        elif wt == "allreduce":
            n_workers = min(cfg.n_workers, len(list(topo.hosts)))
            n_workers = max(2, n_workers)
            sub_cfg = AllReduceConfig(
                n_workers=n_workers,
                model_preset=cfg.model_preset,
                n_iterations=max(1, n_flows // n_workers),
                seed=seed,
            )
            return AllReduceGenerator(topo, sub_cfg).generate()

        elif wt == "microservice":
            sub_cfg = MicroserviceConfig(
                n_requests=n_flows,
                graph_type=cfg.ms_graph_type,
                fan_out=cfg.ms_fan_out,
                n_tenants=cfg.n_tenants,
                seed=seed,
            )
            gen = MicroserviceRPCGenerator(topo, sub_cfg)
            # n_requests × flows_per_request may overshoot; runner sub-samples.
            raw = gen.generate()
            return raw

        else:
            raise ValueError(f"Unknown workload type: {wt!r}")
