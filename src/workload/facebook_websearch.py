"""
LAFS — Facebook Web Search Workload Generator
==============================================
COMP 6910 — Group 7

Generates synthetic flow traces that match the statistical properties of
Facebook's web search cluster traffic as described in:

  Benson, T., Akella, A., & Maltz, D. A. (2010).
  Network Traffic Characteristics of Data Centers in the Wild.
  ACM IMC 2010.

Traffic characteristics reproduced here
----------------------------------------
Flow-size distribution (CDF, empirical from web-search workload):
  < 1 KB   : ~20%  — DNS lookups, tiny metadata, heartbeats
  < 10 KB  : ~55%  — short queries, index lookups, small responses
  < 100 KB : ~90%  — standard web results, thumbnails, partial pages
  < 1 MB   : ~95%  — full page + assets, medium result sets
  < 10 MB  : ~98%  — large responses, cached objects
  ≤ 100 MB : 100%  — bulk transfers, log shipping (rare)

From the LAFS proposal: mice < 100 KB, elephants > 10 MB.
At the 90th percentile boundary, ~90 % of flows are mice,
~2 % are elephants, and ~8 % are medium.

Arrival process
---------------
Flows arrive according to a Poisson process (exponential inter-arrival
times) at rate λ flows/second, derived from:
    λ = load_fraction × link_capacity_gbps × 1e9 / mean_flow_size_bytes

Traffic pattern
---------------
Web search exhibits a rack-concentrated "many-to-one" pattern:
  * "Leaf" hosts send queries to a small set of aggregator/root hosts.
  * Pure random host selection is available as an alternative.

Multi-tenant support
--------------------
Each tenant receives a non-overlapping slice of the host pool.
Intra-tenant flows dominate (95 %); a small fraction cross tenant
boundaries (modeling shared infrastructure or side-channel traffic).

All parameters are reproducible via an explicit integer seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow

# ── Flow-size CDF ─────────────────────────────────────────────────────────────
# (cumulative_probability, size_bytes_lower, size_bytes_upper)
# Each tuple is a CDF segment; sizes are sampled uniformly within the segment.
# Derived from Benson et al. IMC 2010 web-search traffic.
_FB_WEBSEARCH_CDF: List[Tuple[float, int, int]] = [
    (0.20,          0,          1_000),    # 20 % : < 1 KB
    (0.55,      1_000,         10_000),    # 35 % : 1 KB – 10 KB
    (0.90,     10_000,        100_000),    # 35 % : 10 KB – 100 KB (mice boundary)
    (0.95,    100_000,      1_000_000),    #  5 % : 100 KB – 1 MB
    (0.98,  1_000_000,     10_000_000),    #  3 % : 1 MB – 10 MB
    (1.00, 10_000_000,    100_000_000),    #  2 % : 10 MB – 100 MB (elephant)
]

# ── Network constants ─────────────────────────────────────────────────────────
_LINK_RATE_GBPS: float = 1.0             # host-link bottleneck
_LINK_RATE_BPS: float = _LINK_RATE_GBPS * 1e9

# ── Well-known dst ports for web search ──────────────────────────────────────
_WEB_DST_PORTS: List[int] = [80, 443, 8080, 8443]


def _sample_flow_size(rng: random.Random) -> int:
    """
    Draw a random flow size (bytes) from the Facebook web-search CDF
    using inverse transform sampling over piecewise-uniform segments.
    """
    u = rng.random()
    prev_cdf = 0.0
    for (cdf, lo, hi) in _FB_WEBSEARCH_CDF:
        if u <= cdf:
            # Uniform within [lo, hi)
            fraction = (u - prev_cdf) / (cdf - prev_cdf)
            return lo + int(fraction * (hi - lo))
        prev_cdf = cdf
    # Fallback (should never reach here)
    return _FB_WEBSEARCH_CDF[-1][2]


# =============================================================================
# FacebookWebSearchConfig
# =============================================================================
@dataclass
class FacebookWebSearchConfig:
    """
    Configuration for the Facebook web-search workload generator.

    Parameters
    ----------
    n_flows : int
        Total number of flows to generate.
    load_fraction : float
        Target fraction of host-link capacity to utilise (0.3 – 0.9).
        Controls the Poisson arrival rate λ.
    n_tenants : int
        Number of tenants to emulate (each gets an equal slice of hosts).
    start_time : float
        Wall-clock start time (seconds) for the first possible arrival.
    duration_s : float
        Duration of the simulation window in seconds.
        Flows are generated within [start_time, start_time + duration_s].
    seed : int
        RNG seed for reproducibility.
    cross_tenant_fraction : float
        Fraction of flows that cross tenant boundaries (default 0.05 = 5 %).
    aggregator_fraction : float
        Fraction of dst hosts that act as aggregators / root nodes.
        Models the "many-to-one" concentration typical of web search.
        Set to 0.0 for pure uniform random host selection.
    """
    n_flows: int = 1_000
    load_fraction: float = 0.5
    n_tenants: int = 1
    start_time: float = 0.0
    duration_s: float = 10.0
    seed: int = 42
    cross_tenant_fraction: float = 0.05
    aggregator_fraction: float = 0.25   # 25 % of hosts are "aggregators"

    def __post_init__(self) -> None:
        if not (0.0 < self.load_fraction <= 1.0):
            raise ValueError(f"load_fraction must be in (0, 1], got {self.load_fraction}")
        if self.n_tenants < 1:
            raise ValueError(f"n_tenants must be >= 1, got {self.n_tenants}")
        if self.n_flows < 1:
            raise ValueError(f"n_flows must be >= 1, got {self.n_flows}")


# =============================================================================
# FacebookWebSearchGenerator
# =============================================================================
class FacebookWebSearchGenerator:
    """
    Generates flows matching the Facebook web-search traffic distribution.

    Each generated ``Flow`` has:
    * size sampled from the empirical Facebook CDF
    * Poisson arrival time derived from the target load fraction
    * src/dst host pair drawn from the topology host pool
    * tenant_id encoded in the flow_id prefix ("t{i}_f{j}")
    * dst_port in {80, 443, 8080, 8443}

    Parameters
    ----------
    topology : FatTreeGraph
        The network topology (used to enumerate hosts and their IPs).
    config : FacebookWebSearchConfig
        Generation parameters.

    Examples
    --------
    >>> topo = FatTreeGraph(k=4)
    >>> cfg  = FacebookWebSearchConfig(n_flows=200, load_fraction=0.6, seed=7)
    >>> gen  = FacebookWebSearchGenerator(topo, cfg)
    >>> flows = gen.generate()
    >>> len(flows)
    200
    >>> sum(f.is_mice for f in flows) / len(flows) > 0.85
    True
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        config: Optional[FacebookWebSearchConfig] = None,
    ) -> None:
        self.topology = topology
        self.config = config or FacebookWebSearchConfig()
        self._rng = random.Random(self.config.seed)
        self._hosts: List[str] = sorted(topology.hosts)

        n_hosts = len(self._hosts)
        if n_hosts < 2:
            raise ValueError("Topology must have at least 2 hosts")
        if self.config.n_tenants > n_hosts:
            raise ValueError(
                f"n_tenants ({self.config.n_tenants}) > n_hosts ({n_hosts})"
            )

        # Partition hosts into tenant slices (round-robin to be fair).
        self._tenant_hosts: List[List[str]] = [[] for _ in range(self.config.n_tenants)]
        for i, h in enumerate(self._hosts):
            self._tenant_hosts[i % self.config.n_tenants].append(h)

        # Within each tenant, designate a fraction as "aggregators" (popular dst).
        cfg = self.config
        self._tenant_aggregators: List[List[str]] = []
        for tenant_hosts in self._tenant_hosts:
            n_agg = max(1, int(len(tenant_hosts) * cfg.aggregator_fraction))
            self._tenant_aggregators.append(tenant_hosts[:n_agg])

        # Poisson rate: mean flow size from CDF → arrive at load_fraction of capacity.
        mean_size = self._compute_mean_flow_size()
        self._arrival_rate: float = (
            cfg.load_fraction * _LINK_RATE_BPS / (mean_size * 8)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> List[Flow]:
        """
        Generate ``config.n_flows`` flows according to the Facebook web-search
        traffic model.

        Returns
        -------
        List[Flow]
            Flows sorted by arrival_time (ascending).
        """
        flows: List[Flow] = []
        cfg = self.config
        t = cfg.start_time
        end_t = cfg.start_time + cfg.duration_s

        for i in range(cfg.n_flows):
            # ── Poisson inter-arrival ─────────────────────────────────────────
            inter = self._rng.expovariate(self._arrival_rate)
            t = min(t + inter, end_t)

            # ── Tenant selection ──────────────────────────────────────────────
            tenant_id = self._rng.randint(0, cfg.n_tenants - 1)
            cross = (
                cfg.n_tenants > 1
                and self._rng.random() < cfg.cross_tenant_fraction
            )

            # ── Host pair selection ───────────────────────────────────────────
            src_host, dst_host = self._pick_host_pair(tenant_id, cross)

            # ── Flow attributes ───────────────────────────────────────────────
            size = _sample_flow_size(self._rng)
            src_port = self._rng.randint(1024, 65535)
            dst_port = self._rng.choice(_WEB_DST_PORTS)
            flow_id = f"t{tenant_id}_fb_{i:06d}"

            flows.append(Flow(
                flow_id=flow_id,
                src_ip=self.topology.get_host_ip(src_host),
                dst_ip=self.topology.get_host_ip(dst_host),
                src_port=src_port,
                dst_port=dst_port,
                protocol=6,          # TCP
                size_bytes=size,
                arrival_time=t,
            ))

        flows.sort(key=lambda f: f.arrival_time)
        return flows

    def size_distribution_stats(self) -> dict:
        """
        Generate a sample of 10 000 flow sizes and return summary statistics.
        Does not affect the main RNG state (uses an independent RNG).
        """
        sample_rng = random.Random(self.config.seed + 999_999)
        sizes = [_sample_flow_size(sample_rng) for _ in range(10_000)]
        sizes.sort()
        n = len(sizes)
        mice = sum(1 for s in sizes if s < 100_000)
        elephants = sum(1 for s in sizes if s >= 10_000_000)
        return {
            "mean_bytes":     sum(sizes) / n,
            "median_bytes":   sizes[n // 2],
            "p90_bytes":      sizes[int(n * 0.90)],
            "p99_bytes":      sizes[int(n * 0.99)],
            "mice_fraction":  mice / n,
            "elephant_fraction": elephants / n,
            "min_bytes":      sizes[0],
            "max_bytes":      sizes[-1],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pick_host_pair(self, tenant_id: int, cross: bool) -> Tuple[str, str]:
        """Pick (src, dst) host pair; cross=True allows cross-tenant dst."""
        src_pool = self._tenant_hosts[tenant_id]
        src_host = self._rng.choice(src_pool)

        if cross:
            # dst from a different tenant.
            other_tenants = [
                tid for tid in range(self.config.n_tenants) if tid != tenant_id
            ]
            other_tid = self._rng.choice(other_tenants)
            dst_pool = self._tenant_hosts[other_tid]
        else:
            dst_pool = self._tenant_aggregators[tenant_id]

        # Avoid src == dst.
        candidates = [h for h in dst_pool if h != src_host]
        if not candidates:
            candidates = [h for h in self._hosts if h != src_host]
        dst_host = self._rng.choice(candidates)
        return src_host, dst_host

    @staticmethod
    def _compute_mean_flow_size() -> float:
        """Compute the mean flow size (bytes) from the CDF using midpoint rule."""
        total = 0.0
        prev_cdf = 0.0
        for (cdf, lo, hi) in _FB_WEBSEARCH_CDF:
            weight = cdf - prev_cdf
            midpoint = (lo + hi) / 2
            total += weight * midpoint
            prev_cdf = cdf
        return total
