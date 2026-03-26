"""
LAFS -- Scheduler Comparison Experiment
=========================================
COMP-6910 -- Group 7

Compares ECMP, Hedera, CONGA, and LAFS on identical Facebook web-search
workloads across four load levels (30 %, 50 %, 70 %, 90 %).

Pipeline per run
----------------
  FatTreeGraph(k)
      -> FacebookWebSearchGenerator  (n_flows, load_fraction, seed)
      -> <Scheduler>.schedule_flows()
      -> PathFIFOSimulator            (per-path work-conserving FIFO)
      -> MetricsCollector             (FCT, fairness, link utilisation)

Ablation (at 50 % load)
-----------------------
  LAFS                -- full system (MILP + zero predicted utils)
  LAFS-pred           -- MILP + LoadForecaster fitted on ECMP link trace
  LAFS-no-mice        -- MILP, mice_hop_weight=0 (no hop-count penalty)
  ECMP                -- hash-based baseline

Outputs
-------
  results/metrics/comparison_<timestamp>.json   -- full nested results
  results/metrics/comparison_<timestamp>.csv    -- flat CSV for pandas
  results/figures/fct_p99_vs_load.png
  results/figures/fct_cdf_50pct.png
  results/figures/link_util_vs_load.png
  results/figures/jains_fairness_vs_load.png
  results/figures/ablation_50pct.png

Usage
-----
  python experiments/run_comparison.py
  python experiments/run_comparison.py --k 4 --n-flows 200 --no-plots
  python experiments/run_comparison.py --loads 0.5 0.7 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# -- project root on sys.path -------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow
from src.workload.facebook_websearch import (
    FacebookWebSearchGenerator,
    FacebookWebSearchConfig,
)
from src.scheduler.ecmp import ECMPScheduler
from src.scheduler.hedera import HederaScheduler
from src.scheduler.conga import CONGAScheduler
from src.optimizer.lafs_scheduler import LAFSScheduler
from src.optimizer.milp_solver import MILPConfig
from src.metrics.link_load import LinkLoadSampler
from src.prediction.forecaster import LoadForecaster

# =============================================================================
# Simulation helpers
# =============================================================================

_HOST_LINK_BPS: float = 1e9    # 1 Gbps host links
_FABRIC_LINK_BPS: float = 10e9  # 10 Gbps fabric links
_PROP_DELAY_S: float = 5e-6    # 5 us per hop


def _link_cap(u: str, v: str) -> float:
    """Return capacity in bps for directed link (u, v)."""
    if u.startswith("h_") or v.startswith("h_"):
        return _HOST_LINK_BPS
    return _FABRIC_LINK_BPS


def _ideal_fct(flow: Flow) -> float:
    return (flow.size_bytes * 8) / _HOST_LINK_BPS


def _is_host_link(u: str, v: str) -> bool:
    return u.startswith("h_") or v.startswith("h_")


def _compute_link_utils(
    flows: List[Flow],
    sim_duration: float = 10.0,
    fabric_only: bool = False,
) -> Dict[Tuple[str, str], float]:
    """
    Steady-state per-link utilisation:
        util[l] = total_bytes_through_l * 8 / sim_duration / cap[l]

    fabric_only=True excludes host-edge links (which are fixed by src/dst
    and cannot be influenced by path selection — only fabric links differ
    between schedulers).
    """
    link_bytes: Dict[Tuple[str, str], float] = defaultdict(float)
    for f in flows:
        if not f.assigned_path:
            continue
        path = f.assigned_path
        for h in range(len(path) - 1):
            u, v = path[h], path[h + 1]
            if fabric_only and _is_host_link(u, v):
                continue
            link_bytes[(u, v)] += f.size_bytes

    return {
        link: total * 8 / sim_duration / _link_cap(link[0], link[1])
        for link, total in link_bytes.items()
    }


def _simulate_fct(flows: List[Flow], sim_duration: float = 10.0) -> Dict[str, float]:
    """
    Congestion-aware FCT using M/G/1 approximation on FABRIC links.

    For each flow f:
        rho_f   = max FABRIC link utilisation on path(f)  [fabric only]
        tx_f    = size_f * 8 / HOST_LINK_BPS
        FCT_f   = tx_f / (1 - min(rho_f, 0.99)) + prop_delay

    Uses fabric-only utilisation because:
    * Host-link rho is identical across all schedulers (path-independent).
    * Fabric links are where scheduler decisions have impact — different path
      choices lead to different fabric utilisation and thus different FCT.
    """
    # Fabric-only utilisation: captures scheduler impact
    fabric_utils = _compute_link_utils(flows, sim_duration, fabric_only=True)

    fcts: Dict[str, float] = {}
    for f in flows:
        if not f.assigned_path:
            continue
        path = f.assigned_path
        n_hops = len(path) - 1
        prop = n_hops * _PROP_DELAY_S
        rho = max(
            (fabric_utils.get((path[h], path[h + 1]), 0.0)
             for h in range(n_hops)
             if not _is_host_link(path[h], path[h + 1])),
            default=0.0,
        )
        rho = min(rho, 0.99)
        tx = f.size_bytes * 8 / _HOST_LINK_BPS
        fcts[f.flow_id] = tx / max(1.0 - rho, 0.01) + prop
    return fcts


def _compute_link_imbalance(
    flows: List[Flow], sim_duration: float = 10.0
) -> Tuple[float, float, float, int]:
    """
    Return (fabric_util_mean, fabric_util_p95, fabric_util_max, n_hot_links)
    using FABRIC-ONLY links (excludes host-edge links which are path-independent).
    n_hot_links = fabric links with utilisation > 0.5.
    """
    utils = list(
        _compute_link_utils(flows, sim_duration, fabric_only=True).values()
    )
    active = [u for u in utils if u > 0]
    if not active:
        return 0.0, 0.0, 0.0, 0
    return (
        sum(active) / len(active),
        _percentile(active, 95),
        max(active),
        sum(1 for u in active if u > 0.5),
    )


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = max(0, int(math.ceil(len(sv) * p / 100.0)) - 1)
    return sv[min(idx, len(sv) - 1)]


def _jains_fairness(values: List[float]) -> float:
    """Jain's fairness index on a list of values (slowdowns)."""
    if not values:
        return 0.0
    n = len(values)
    s = sum(values)
    sq = sum(v * v for v in values)
    return (s * s) / (n * sq) if sq > 0 else 1.0


# =============================================================================
# SchedulerResult
# =============================================================================

@dataclass
class SchedulerResult:
    """All metrics for one (scheduler, load) combination."""
    scheduler_name: str
    load_fraction: float
    n_flows: int
    n_scheduled: int

    # FCT -- all flows (ms)
    fct_p50_ms: float = 0.0
    fct_p95_ms: float = 0.0
    fct_p99_ms: float = 0.0
    fct_mean_ms: float = 0.0

    # FCT -- mice only (ms)
    mice_fct_p50_ms: float = 0.0
    mice_fct_p95_ms: float = 0.0
    mice_fct_p99_ms: float = 0.0

    # FCT -- elephant only (ms)
    elephant_fct_p50_ms: float = 0.0
    elephant_fct_p99_ms: float = 0.0

    # Slowdown (FCT / ideal FCT)
    slowdown_mean: float = 0.0
    slowdown_p99: float = 0.0

    # Fairness
    jains_fairness: float = 0.0

    # Link utilisation (steady-state: bytes/duration/cap)
    link_util_mean: float = 0.0
    link_util_p95: float = 0.0
    link_util_max: float = 0.0
    link_n_hot: int = 0         # links with utilisation > 80 %

    # Scheduling overhead
    balance_ratio: float = 0.0
    unique_paths: int = 0
    solve_time_ms: float = 0.0   # LAFS only; 0 for others

    # Counts
    n_mice: int = 0
    n_elephant: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# ExperimentRunner
# =============================================================================

class ExperimentRunner:
    """
    Orchestrates the full comparison experiment.

    Parameters
    ----------
    k : int
        Fat-tree parameter (k=4 → 16 hosts, k=8 → 128 hosts).
    n_flows : int
        Flows to generate per run.
    seed : int
        RNG seed (same for all runs -- fair comparison).
    milp_time_limit : float
        Max seconds for LAFS MILP solver per run.
    verbose : bool
        Print per-run progress to stdout.
    """

    def __init__(
        self,
        k: int = 8,
        n_flows: int = 1000,
        seed: int = 42,
        milp_time_limit: float = 10.0,
        verbose: bool = True,
    ) -> None:
        self.k = k
        self.n_flows = n_flows
        self.seed = seed
        self.milp_time_limit = milp_time_limit
        self.verbose = verbose

        if self.verbose:
            print(f"[setup] Building FatTreeGraph(k={k}) ...")
        self.topology = FatTreeGraph(k=k)
        if self.verbose:
            print(
                f"[setup] Topology: {self.topology.n_hosts} hosts, "
                f"{self.topology.n_switches} switches, "
                f"{self.topology.graph.number_of_edges()} directed links"
            )

    # ------------------------------------------------------------------
    # Flow generation
    # ------------------------------------------------------------------

    def _generate_flows(self, load_fraction: float) -> List[Flow]:
        # aggregator_fraction=0.05: 5% of hosts are "hot" aggregators receiving
        # ~95% of traffic.  With k=8 (128 hosts), this means ~6 aggregators
        # each absorbing ~158 flows on average.  This creates realistic fabric
        # congestion that differs between schedulers.
        cfg = FacebookWebSearchConfig(
            n_flows=self.n_flows,
            load_fraction=load_fraction,
            seed=self.seed,
            aggregator_fraction=0.05,
        )
        gen = FacebookWebSearchGenerator(self.topology, cfg)
        return gen.generate()

    # ------------------------------------------------------------------
    # Scheduler factories (fresh instances to avoid state pollution)
    # ------------------------------------------------------------------

    def _make_ecmp(self) -> ECMPScheduler:
        return ECMPScheduler(self.topology)

    def _make_hedera(self) -> HederaScheduler:
        return HederaScheduler(self.topology)

    def _make_conga(self) -> CONGAScheduler:
        return CONGAScheduler(self.topology)

    def _make_lafs(
        self,
        mice_hop_weight: float = 1e-3,
        window_s: float = 10.0,
    ) -> LAFSScheduler:
        # window_s = simulation duration so per-flow demand = size/duration,
        # giving utilisation values in [0, 1] for typical workloads.
        cfg = MILPConfig(
            solver="pulp",
            time_limit_s=self.milp_time_limit,
            mip_gap=0.01,
            mice_hop_weight=mice_hop_weight,
            verbose=False,
        )
        return LAFSScheduler(
            self.topology, milp_config=cfg, forecaster=None, window_s=window_s
        )

    def _make_lafs_with_forecaster(
        self, sampler: LinkLoadSampler
    ) -> LAFSScheduler:
        """LAFS variant with a fitted LoadForecaster (ablation)."""
        forecaster = LoadForecaster(
            self.topology, method="hybrid", horizon_s=0.1, window_s=0.1
        )
        forecaster.fit(sampler)
        cfg = MILPConfig(
            solver="pulp",
            time_limit_s=self.milp_time_limit,
            mip_gap=0.01,
            verbose=False,
        )
        return LAFSScheduler(
            self.topology, milp_config=cfg, forecaster=forecaster, window_s=10.0
        )

    # ------------------------------------------------------------------
    # Core run: one scheduler × one load level
    # ------------------------------------------------------------------

    def _run_one(
        self,
        scheduler_name: str,
        scheduler,
        flows: List[Flow],
        load_fraction: float,
        milp_result=None,
    ) -> SchedulerResult:
        """Schedule *flows*, simulate FCT, collect all metrics."""

        # -- FCT simulation (congestion-aware M/G/1 model) --
        scheduled_flows = [f for f in flows if f.assigned_path]
        # Compute simulation duration from actual flow arrival times
        if scheduled_flows:
            t_min = min(f.arrival_time for f in scheduled_flows)
            t_max = max(f.arrival_time for f in scheduled_flows)
            sim_dur = max(t_max - t_min, 1.0)   # at least 1 s
        else:
            sim_dur = 10.0

        fcts = _simulate_fct(scheduled_flows, sim_duration=sim_dur)

        all_fcts = [fcts[f.flow_id] * 1e3 for f in scheduled_flows if f.flow_id in fcts]
        mice_fcts = [
            fcts[f.flow_id] * 1e3
            for f in scheduled_flows
            if f.is_mice and f.flow_id in fcts
        ]
        eleph_fcts = [
            fcts[f.flow_id] * 1e3
            for f in scheduled_flows
            if f.is_elephant and f.flow_id in fcts
        ]
        slowdowns = [
            fcts[f.flow_id] / _ideal_fct(f)
            for f in scheduled_flows
            if f.flow_id in fcts and _ideal_fct(f) > 0
        ]

        # -- Steady-state link utilisation --
        util_mean, util_p95, util_max, n_hot = _compute_link_imbalance(
            scheduled_flows, sim_duration=sim_dur
        )

        # -- Metrics from scheduler --
        m = scheduler.metrics
        balance = m.path_distribution()
        balance_ratio = (
            min(balance.values()) / max(balance.values())
            if len(balance) > 1 and max(balance.values()) > 0
            else 1.0
        )

        solve_ms = (milp_result.solve_time_s * 1e3) if milp_result else 0.0

        return SchedulerResult(
            scheduler_name=scheduler_name,
            load_fraction=load_fraction,
            n_flows=len(flows),
            n_scheduled=len(scheduled_flows),
            fct_p50_ms=_percentile(all_fcts, 50),
            fct_p95_ms=_percentile(all_fcts, 95),
            fct_p99_ms=_percentile(all_fcts, 99),
            fct_mean_ms=sum(all_fcts) / len(all_fcts) if all_fcts else 0.0,
            mice_fct_p50_ms=_percentile(mice_fcts, 50),
            mice_fct_p95_ms=_percentile(mice_fcts, 95),
            mice_fct_p99_ms=_percentile(mice_fcts, 99),
            elephant_fct_p50_ms=_percentile(eleph_fcts, 50),
            elephant_fct_p99_ms=_percentile(eleph_fcts, 99),
            slowdown_mean=sum(slowdowns) / len(slowdowns) if slowdowns else 0.0,
            slowdown_p99=_percentile(slowdowns, 99),
            jains_fairness=_jains_fairness(slowdowns),
            link_util_mean=util_mean,
            link_util_p95=util_p95,
            link_util_max=util_max,
            link_n_hot=n_hot,
            balance_ratio=balance_ratio,
            unique_paths=m.unique_paths_used,
            solve_time_ms=solve_ms,
            n_mice=m.mice_flows,
            n_elephant=m.elephant_flows,
        )

    def _schedule_and_run(
        self,
        scheduler_name: str,
        scheduler,
        flows: List[Flow],
        load_fraction: float,
    ) -> SchedulerResult:
        """Run scheduling + simulation for non-LAFS schedulers."""
        if self.verbose:
            print(f"  [{scheduler_name}] scheduling {len(flows)} flows ...", end="", flush=True)
        t0 = time.perf_counter()

        if scheduler_name == "Hedera":
            scheduler.schedule_flows(flows)
            elephants = [f for f in flows if f.is_elephant]
            if elephants:
                scheduler.reschedule_elephants(elephants)
        else:
            scheduler.schedule_flows(flows)

        elapsed = time.perf_counter() - t0
        if self.verbose:
            n_ok = sum(1 for f in flows if f.assigned_path)
            print(f" {n_ok}/{len(flows)} scheduled in {elapsed*1e3:.1f}ms")

        return self._run_one(scheduler_name, scheduler, flows, load_fraction)

    def _schedule_lafs(
        self,
        scheduler_name: str,
        scheduler: LAFSScheduler,
        flows: List[Flow],
        load_fraction: float,
    ) -> SchedulerResult:
        """Run LAFS MILP batch scheduling."""
        if self.verbose:
            print(f"  [{scheduler_name}] MILP solving {len(flows)} flows ...", end="", flush=True)

        milp_result = scheduler.schedule_flows_milp(flows)

        if self.verbose:
            print(
                f" {len(milp_result.assignments)}/{len(flows)} assigned | "
                f"status={milp_result.status} "
                f"max_util={milp_result.max_utilisation:.3f} "
                f"time={milp_result.solve_time_s*1e3:.0f}ms"
            )

        return self._run_one(scheduler_name, scheduler, flows, load_fraction, milp_result)

    # ------------------------------------------------------------------
    # Load sweep
    # ------------------------------------------------------------------

    def run_load_sweep(
        self,
        loads: List[float] = None,
    ) -> Dict[float, Dict[str, SchedulerResult]]:
        """
        Run all four schedulers at each load level.

        Returns
        -------
        dict: load_fraction -> dict: scheduler_name -> SchedulerResult
        """
        if loads is None:
            loads = [0.3, 0.5, 0.7, 0.9]

        results: Dict[float, Dict[str, SchedulerResult]] = {}

        for load in loads:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Load = {load*100:.0f}%  (generating {self.n_flows} flows ...)")
                print(f"{'='*60}")

            results[load] = {}

            # -- ECMP --
            flows = self._generate_flows(load)
            sched = self._make_ecmp()
            results[load]["ECMP"] = self._schedule_and_run("ECMP", sched, flows, load)

            # -- Hedera --
            flows = self._generate_flows(load)
            sched = self._make_hedera()
            results[load]["Hedera"] = self._schedule_and_run("Hedera", sched, flows, load)

            # -- CONGA --
            flows = self._generate_flows(load)
            sched = self._make_conga()
            results[load]["CONGA"] = self._schedule_and_run("CONGA", sched, flows, load)

            # -- LAFS --
            flows = self._generate_flows(load)
            sched = self._make_lafs()
            results[load]["LAFS"] = self._schedule_lafs("LAFS", sched, flows, load)

            if self.verbose:
                _print_load_summary(results[load])

        return results

    # ------------------------------------------------------------------
    # Ablation study (at a single load level)
    # ------------------------------------------------------------------

    def run_ablation(self, load: float = 0.5) -> Dict[str, SchedulerResult]:
        """
        Compare LAFS variants at one load level.

        Variants
        --------
        ECMP             -- hash-based baseline
        LAFS             -- MILP + zero predicted utils
        LAFS-pred        -- MILP + LoadForecaster fitted on ECMP link trace
        LAFS-no-mice     -- MILP, mice_hop_weight=0 (no hop-count tie-breaker)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Ablation study at load={load*100:.0f}%")
            print(f"{'='*60}")

        ablation: Dict[str, SchedulerResult] = {}

        # ECMP baseline
        flows = self._generate_flows(load)
        sched = self._make_ecmp()
        ablation["ECMP"] = self._schedule_and_run("ECMP", sched, flows, load)

        # LAFS (no prediction)
        flows = self._generate_flows(load)
        sched = self._make_lafs()
        ablation["LAFS"] = self._schedule_lafs("LAFS", sched, flows, load)

        # LAFS with prediction: fit forecaster on ECMP link trace
        if self.verbose:
            print("  [LAFS-pred] fitting LoadForecaster on ECMP link trace ...")
        ecmp_flows = self._generate_flows(load)
        ecmp_sched = self._make_ecmp()
        ecmp_sched.schedule_flows(ecmp_flows)
        sampler = LinkLoadSampler(self.topology, window_s=0.1)
        sampler.ingest(ecmp_flows)
        sampler.build_series()

        flows = self._generate_flows(load)
        sched = self._make_lafs_with_forecaster(sampler)
        ablation["LAFS-pred"] = self._schedule_lafs("LAFS-pred", sched, flows, load)

        # LAFS without mice hop-weight
        flows = self._generate_flows(load)
        sched = self._make_lafs(mice_hop_weight=0.0)
        ablation["LAFS-no-mice"] = self._schedule_lafs("LAFS-no-mice", sched, flows, load)

        if self.verbose:
            _print_load_summary(ablation)

        return ablation


# =============================================================================
# Console helpers
# =============================================================================

def _print_load_summary(results: Dict[str, SchedulerResult]) -> None:
    header = (
        f"  {'Scheduler':<14} {'Sched':>6} {'P50ms':>7} "
        f"{'P99ms':>8} {'miceP99':>8} {'utilMax':>8} "
        f"{'hotLnk':>7} {'Jain':>6} {'SolvMs':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, r in results.items():
        print(
            f"  {name:<14} {r.n_scheduled:>6} {r.fct_p50_ms:>7.3f} "
            f"{r.fct_p99_ms:>8.2f} {r.mice_fct_p99_ms:>8.3f} "
            f"{r.link_util_max:>8.3f} {r.link_n_hot:>7} "
            f"{r.jains_fairness:>6.4f} {r.solve_time_ms:>7.1f}"
        )


# =============================================================================
# Save results
# =============================================================================

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_results(
    sweep: Dict[float, Dict[str, SchedulerResult]],
    ablation: Dict[str, SchedulerResult],
    output_dir: str,
) -> Tuple[str, str]:
    """Save JSON + CSV.  Returns (json_path, csv_path)."""
    os.makedirs(output_dir, exist_ok=True)
    ts = _timestamp()

    # -- JSON --
    payload = {
        "sweep": {
            str(load): {name: r.to_dict() for name, r in sched_map.items()}
            for load, sched_map in sweep.items()
        },
        "ablation": {name: r.to_dict() for name, r in ablation.items()},
    }
    json_path = os.path.join(output_dir, f"comparison_{ts}.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # -- CSV (flat rows) --
    csv_path = os.path.join(output_dir, f"comparison_{ts}.csv")
    rows = []
    for load, sched_map in sweep.items():
        for name, r in sched_map.items():
            rows.append({"experiment": "sweep", **r.to_dict()})
    for name, r in ablation.items():
        rows.append({"experiment": "ablation", **r.to_dict()})

    if rows:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


# =============================================================================
# Plotting
# =============================================================================

def plot_results(
    sweep: Dict[float, Dict[str, SchedulerResult]],
    ablation: Dict[str, SchedulerResult],
    figures_dir: str,
    flows_for_cdf: Optional[Dict[str, List[Flow]]] = None,
) -> List[str]:
    """
    Generate five matplotlib figures.  Returns list of saved file paths.
    Imports matplotlib lazily so the module remains importable without it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (works headlessly)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[plot] matplotlib not available -- skipping plots")
        return []

    os.makedirs(figures_dir, exist_ok=True)
    saved = []
    loads = sorted(sweep.keys())
    scheduler_names = ["ECMP", "Hedera", "CONGA", "LAFS"]
    colors = {"ECMP": "#2196F3", "Hedera": "#FF9800", "CONGA": "#4CAF50", "LAFS": "#F44336"}
    markers = {"ECMP": "o", "Hedera": "s", "CONGA": "^", "LAFS": "D"}

    # ── 1. FCT P99 vs Load ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in scheduler_names:
        all_p99 = [sweep[l][name].fct_p99_ms for l in loads if name in sweep[l]]
        mice_p99 = [sweep[l][name].mice_fct_p99_ms for l in loads if name in sweep[l]]
        x = [l * 100 for l in loads if name in sweep[l]]
        axes[0].plot(x, all_p99, marker=markers[name], color=colors[name], label=name)
        axes[1].plot(x, mice_p99, marker=markers[name], color=colors[name], label=name)

    for ax, title, ylabel in zip(
        axes,
        ["All flows -- P99 FCT vs Load", "Mice flows -- P99 FCT vs Load"],
        ["P99 FCT (ms)", "Mice P99 FCT (ms)"],
    ):
        ax.set_xlabel("Load (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([int(l * 100) for l in loads])

    plt.tight_layout()
    path = os.path.join(figures_dir, "fct_p99_vs_load.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(path)

    # ── 2. Link utilisation vs Load ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [l * 100 for l in loads]
    for name in scheduler_names:
        y_max = [sweep[l][name].link_util_max for l in loads if name in sweep[l]]
        y_mean = [sweep[l][name].link_util_mean for l in loads if name in sweep[l]]
        ax.plot(x, y_max, marker=markers[name], color=colors[name],
                linestyle="-", label=f"{name} (max)")
        ax.plot(x, y_mean, marker=markers[name], color=colors[name],
                linestyle="--", alpha=0.5)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="Capacity limit")
    ax.set_xlabel("Load (%)")
    ax.set_ylabel("Link Utilisation")
    ax.set_title("Max link utilisation vs Load")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([int(l * 100) for l in loads])
    plt.tight_layout()
    path = os.path.join(figures_dir, "link_util_vs_load.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(path)

    # ── 3. Jain's fairness vs Load ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in scheduler_names:
        y = [sweep[l][name].jains_fairness for l in loads if name in sweep[l]]
        ax.plot(x, y, marker=markers[name], color=colors[name], label=name)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Load (%)")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Fairness (FCT slowdown) vs Load")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([int(l * 100) for l in loads])
    plt.tight_layout()
    path = os.path.join(figures_dir, "jains_fairness_vs_load.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(path)

    # ── 4. Ablation bar chart at 50 % load ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    abl_names = list(ablation.keys())
    abl_colors = ["#2196F3", "#F44336", "#9C27B0", "#FF5722"]
    x_pos = range(len(abl_names))

    metrics_abl = [
        ("fct_p99_ms", "P99 FCT (ms)", "All-flow P99 FCT"),
        ("mice_fct_p99_ms", "Mice P99 FCT (ms)", "Mice P99 FCT"),
        ("jains_fairness", "Jain's Index", "Fairness Index"),
    ]
    for ax, (attr, ylabel, title) in zip(axes, metrics_abl):
        vals = [getattr(ablation[n], attr) for n in abl_names]
        bars = ax.bar(x_pos, vals, color=abl_colors[:len(abl_names)])
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(abl_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.suptitle("Ablation study (load=50%)", y=1.02)
    plt.tight_layout()
    path = os.path.join(figures_dir, "ablation_50pct.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(path)

    # ── 5. FCT CDF at 50 % load (all schedulers) ────────────────────────────
    # Reconstruct FCT distribution from percentile points (approximation)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pcts = [10, 25, 50, 75, 90, 95, 99]

    if 0.5 in sweep:
        for ax_idx, (flow_type, p50_attr, p95_attr, p99_attr, title) in enumerate([
            ("All", "fct_p50_ms", "fct_p95_ms", "fct_p99_ms", "All flows FCT CDF (50% load)"),
            ("Mice", "mice_fct_p50_ms", "mice_fct_p95_ms", "mice_fct_p99_ms",
             "Mice FCT CDF (50% load)"),
        ]):
            ax = axes[ax_idx]
            for name in scheduler_names:
                if name not in sweep[0.5]:
                    continue
                r = sweep[0.5][name]
                pts = [
                    getattr(r, p50_attr),
                    getattr(r, p95_attr),
                    getattr(r, p99_attr),
                ]
                # Plot 3 known points as CDF markers
                cdf_pcts = [50, 95, 99]
                ax.plot(
                    pts, [p / 100 for p in cdf_pcts],
                    marker=markers[name], color=colors[name], label=name,
                )
            ax.set_xlabel("FCT (ms)")
            ax.set_ylabel("CDF")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(figures_dir, "fct_cdf_50pct.png")
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(path)

    return saved


# =============================================================================
# Main
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LAFS scheduler comparison experiment"
    )
    p.add_argument("--k", type=int, default=8, help="Fat-tree k (default 8)")
    p.add_argument("--n-flows", type=int, default=1000,
                   help="Flows per run (default 1000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    p.add_argument("--loads", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
                   help="Load fractions to sweep (default: 0.3 0.5 0.7 0.9)")
    p.add_argument("--ablation-load", type=float, default=0.5,
                   help="Load for ablation study (default 0.5)")
    p.add_argument("--milp-time-limit", type=float, default=10.0,
                   help="MILP solver time limit in seconds (default 10)")
    p.add_argument("--output-dir", default="results/metrics",
                   help="Directory for JSON/CSV output")
    p.add_argument("--figures-dir", default="results/figures",
                   help="Directory for plot files")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib plotting")
    p.add_argument("--no-ablation", action="store_true",
                   help="Skip ablation study")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    runner = ExperimentRunner(
        k=args.k,
        n_flows=args.n_flows,
        seed=args.seed,
        milp_time_limit=args.milp_time_limit,
        verbose=not args.quiet,
    )

    t_start = time.perf_counter()

    # -- Load sweep --
    sweep = runner.run_load_sweep(loads=sorted(args.loads))

    # -- Ablation --
    ablation_load = args.ablation_load
    if ablation_load not in sweep:
        ablation_load = args.loads[0]

    if args.no_ablation:
        ablation: Dict = {}
    else:
        ablation = runner.run_ablation(load=ablation_load)

    total_s = time.perf_counter() - t_start
    print(f"\n[done] Total experiment time: {total_s:.1f}s")

    # -- Save --
    json_path, csv_path = save_results(sweep, ablation, args.output_dir)
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")

    # -- Plot --
    if not args.no_plots:
        saved_figs = plot_results(sweep, ablation, args.figures_dir)
        for p in saved_figs:
            print(f"[figure] {p}")

    # -- Final summary table --
    print("\n" + "=" * 72)
    print("FINAL SUMMARY -- P99 FCT (ms) at each load level")
    print("=" * 72)
    loads_sorted = sorted(sweep.keys())
    header = f"  {'Scheduler':<12}" + "".join(f"  {int(l*100)}%" for l in loads_sorted)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in ["ECMP", "Hedera", "CONGA", "LAFS"]:
        row = f"  {name:<12}"
        for load in loads_sorted:
            r = sweep[load].get(name)
            row += f"  {r.fct_p99_ms:>5.1f}" if r else "     N/A"
        print(row)

    print("\nJain's Fairness Index")
    print("  " + "-" * (len(header) - 2))
    for name in ["ECMP", "Hedera", "CONGA", "LAFS"]:
        row = f"  {name:<12}"
        for load in loads_sorted:
            r = sweep[load].get(name)
            row += f"  {r.jains_fairness:.4f}" if r else "     N/A"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
