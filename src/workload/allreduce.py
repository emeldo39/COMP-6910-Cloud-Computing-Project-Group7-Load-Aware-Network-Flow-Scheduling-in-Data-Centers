"""
LAFS — Synthetic AllReduce Workload Generator
=============================================
COMP-6910 — Group 7

Generates synthetic traffic traces that model distributed deep-learning
training workloads, specifically the AllReduce collective operation used by
frameworks such as PyTorch DDP, Horovod, and NCCL.

Background
----------
During back-propagation, each worker GPU computes gradients for its local
mini-batch.  AllReduce combines (sum-reduces) these gradients across all
workers so every worker ends up with identical averaged gradients before
taking the next optimiser step.

The dominant production algorithm is **Ring AllReduce** (NCCL default):

  Phase 1 — Reduce-Scatter (n-1 steps):
    Worker i sends G/n bytes to worker (i+1) % n,
    each step propagating partial sums around the ring.

  Phase 2 — All-Gather (n-1 steps):
    Worker i sends its fully-reduced shard to the ring,
    reconstructing the full gradient tensor on each worker.

Total bytes per worker per iteration:
    2 × G × (n-1)/n  ≈  2G   for large n

This generator models a training job as a sequence of **epochs**, each
containing ``steps_per_epoch`` iterations.  Each iteration has three phases:

  1. Forward pass  — small activations between pipeline stages
  2. Backward pass — gradient computation (local, no network traffic)
  3. AllReduce     — gradient synchronisation (the network-intensive phase)

Traffic structure per AllReduce iteration (ring, n workers):
  * n ring flows:  worker_i → worker_{(i+1) % n},  size = gradient_bytes / n
  * All n flows start simultaneously (synchronised burst)
  * All must complete before the next iteration starts (barrier semantics)

Additionally, a parameter-server variant is provided:
  * n flows: worker_i → PS (parameter server)
  * n flows: PS → worker_i
  * Total 2n flows per iteration

Reference
---------
Qian, K., et al. (2024). Alibaba HPN: A Data Center Network for Large
Language Model Training. ACM SIGCOMM 2024.

From the LAFS proposal: "AllReduce operations generate synchronized bursts
accounting for 50–75% of traffic".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow


# ── Model size presets (bytes of gradient per iteration) ─────────────────────
MODEL_SIZES = {
    "resnet50":   25_000_000,     #  25 MB  (ResNet-50 ~25 M params × 4 bytes)
    "bert_base":  110_000_000,    # 110 MB  (BERT-base ~110 M params)
    "gpt2":       548_000_000,    # 548 MB  (GPT-2 small)
    "llama_7b":  14_000_000_000,  #  14 GB  (LLaMA-7B, impractical on k=4 — use for k=8)
    "custom":     0,              # user-specified via gradient_bytes
}

# ── AllReduce port (NCCL default) ─────────────────────────────────────────────
_NCCL_PORT: int = 29500


# =============================================================================
# AllReduceConfig
# =============================================================================
@dataclass
class AllReduceConfig:
    """
    Configuration for the AllReduce workload generator.

    Parameters
    ----------
    n_workers : int
        Number of workers participating in AllReduce. Must be ≥ 2.
        Workers are assigned to hosts from the topology (first n_workers hosts
        sorted alphabetically, i.e. h_0_0_0, h_0_0_1, …).
    gradient_bytes : int
        Total gradient tensor size in bytes per iteration.
        Overrides ``model_preset`` if both are specified and > 0.
    model_preset : str
        One of 'resnet50', 'bert_base', 'gpt2', 'llama_7b', 'custom'.
        If 'custom', ``gradient_bytes`` must be set.
    n_iterations : int
        Number of AllReduce iterations (training steps) to simulate.
    steps_per_epoch : int
        Training steps per epoch (used for epoch boundary logging only).
    start_time : float
        Simulation start time in seconds.
    iteration_gap_s : float
        Gap between the END of one AllReduce and the START of the next,
        modelling computation time (forward + backward pass).
        A Gaussian jitter of ±10 % is applied for realism.
    mode : str
        'ring'  — ring AllReduce (default, NCCL-style)
        'ps'    — parameter-server AllReduce (workers ↔ one PS node)
    seed : int
        RNG seed.
    pipeline_stages : int
        If > 1, model pipeline parallelism: workers are split into ``stages``
        groups and adjacent groups exchange activations during the forward pass.
    activation_bytes : int
        Bytes of activations transferred per pipeline boundary per step.
        Only used when pipeline_stages > 1.
    """
    n_workers: int = 8
    gradient_bytes: int = 0
    model_preset: str = "bert_base"
    n_iterations: int = 100
    steps_per_epoch: int = 100
    start_time: float = 0.0
    iteration_gap_s: float = 0.050       # 50 ms compute + backward time
    mode: str = "ring"                   # 'ring' or 'ps'
    seed: int = 42
    pipeline_stages: int = 1
    activation_bytes: int = 10_000_000  # 10 MB activations per pipeline boundary

    def __post_init__(self) -> None:
        if self.n_workers < 2:
            raise ValueError(f"n_workers must be >= 2, got {self.n_workers}")
        if self.mode not in ("ring", "ps"):
            raise ValueError(f"mode must be 'ring' or 'ps', got {self.mode}")
        if self.pipeline_stages < 1:
            raise ValueError(f"pipeline_stages must be >= 1, got {self.pipeline_stages}")
        # Resolve gradient size.
        if self.gradient_bytes == 0:
            gb = MODEL_SIZES.get(self.model_preset, 0)
            if gb == 0:
                raise ValueError(
                    f"model_preset '{self.model_preset}' unknown and gradient_bytes=0"
                )
            self.gradient_bytes = gb

    @property
    def shard_bytes(self) -> int:
        """Bytes per ring step per worker (gradient / n_workers)."""
        return max(1, self.gradient_bytes // self.n_workers)

    @property
    def total_bytes_per_iteration(self) -> int:
        """
        Total bytes injected into the network per AllReduce iteration.
        Ring: 2 × gradient_bytes × (n-1)/n  ≈  2G
        PS:   2 × gradient_bytes (n workers upload + PS broadcasts back)
        """
        n = self.n_workers
        if self.mode == "ring":
            return 2 * self.gradient_bytes * (n - 1) // n
        else:
            return 2 * self.gradient_bytes


# =============================================================================
# AllReduceGenerator
# =============================================================================
class AllReduceGenerator:
    """
    Generates synthetic AllReduce flow traces for ML training workloads.

    Each generated iteration produces a **synchronised burst** of flows
    that all share the same arrival_time (the barrier start time).  The
    barrier end time is not modelled here — the scheduler experiment layer
    must track flow completions to determine when the next iteration starts.

    Parameters
    ----------
    topology : FatTreeGraph
        Host pool to draw workers from.
    config : AllReduceConfig
        Generation parameters.

    Examples
    --------
    >>> topo = FatTreeGraph(k=4)
    >>> cfg  = AllReduceConfig(n_workers=4, model_preset='resnet50', n_iterations=10)
    >>> gen  = AllReduceGenerator(topo, cfg)
    >>> flows = gen.generate()
    >>> len(flows)   # n_workers flows × n_iterations (ring mode)
    40
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        config: Optional[AllReduceConfig] = None,
    ) -> None:
        self.topology = topology
        self.config = config or AllReduceConfig()
        self._rng = random.Random(self.config.seed)

        all_hosts = sorted(topology.hosts)
        if len(all_hosts) < self.config.n_workers:
            raise ValueError(
                f"Topology has {len(all_hosts)} hosts but n_workers="
                f"{self.config.n_workers}"
            )

        # Assign first n_workers hosts as the worker pool.
        self._workers: List[str] = all_hosts[: self.config.n_workers]
        self._worker_ips: List[str] = [
            topology.get_host_ip(w) for w in self._workers
        ]

        # PS mode: the first worker acts as the parameter server.
        self._ps_host: str = self._workers[0]
        self._ps_ip: str = self._worker_ips[0]

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> List[Flow]:
        """
        Generate all AllReduce flows for the configured number of iterations.

        Returns
        -------
        List[Flow]
            Flows sorted by arrival_time then flow_id.
            All flows within a single iteration share the same arrival_time
            (synchronised burst semantics).
        """
        all_flows: List[Flow] = []
        cfg = self.config
        t = cfg.start_time

        for iteration in range(cfg.n_iterations):
            # Compute time with ±10 % jitter on the iteration gap.
            jitter = 1.0 + self._rng.gauss(0, 0.1)
            gap = max(0.0, cfg.iteration_gap_s * jitter)

            if cfg.mode == "ring":
                iter_flows = self._ring_allreduce_flows(iteration, t)
            else:
                iter_flows = self._ps_allreduce_flows(iteration, t)

            # Pipeline activation flows (if pipeline parallelism enabled).
            if cfg.pipeline_stages > 1:
                iter_flows += self._pipeline_flows(iteration, t)

            all_flows.extend(iter_flows)

            # Advance time: gap (compute) + estimated AllReduce duration.
            # Estimate: shard_bytes × 8 / link_rate × 2 rounds
            link_rate_bps = 1e9
            allreduce_est_s = (cfg.shard_bytes * 8 / link_rate_bps) * 2
            t += gap + allreduce_est_s

        all_flows.sort(key=lambda f: (f.arrival_time, f.flow_id))
        return all_flows

    def worker_hosts(self) -> List[str]:
        """Return the list of worker node names used by this generator."""
        return list(self._workers)

    def worker_ips(self) -> List[str]:
        """Return worker IP addresses."""
        return list(self._worker_ips)

    def iteration_flow_count(self) -> int:
        """Number of flows generated per AllReduce iteration."""
        cfg = self.config
        if cfg.mode == "ring":
            base = cfg.n_workers          # one ring flow per worker
        else:
            base = 2 * cfg.n_workers      # upload + download per worker

        pipeline_extra = 0
        if cfg.pipeline_stages > 1:
            n_boundaries = cfg.pipeline_stages - 1
            pipeline_extra = n_boundaries * 2   # forward + backward per boundary

        return base + pipeline_extra

    # ── Ring AllReduce ────────────────────────────────────────────────────────

    def _ring_allreduce_flows(self, iteration: int, t: float) -> List[Flow]:
        """
        Generate one ring AllReduce iteration:
        worker_i → worker_{(i+1) % n}  for each i in [0, n-1].

        Total flows: n_workers.
        Each flow carries: shard_bytes bytes (gradient / n_workers).
        """
        cfg = self.config
        n = cfg.n_workers
        flows: List[Flow] = []

        for i in range(n):
            src_ip = self._worker_ips[i]
            dst_ip = self._worker_ips[(i + 1) % n]
            flow_id = f"ar_ring_it{iteration:04d}_w{i}"
            flows.append(Flow(
                flow_id=flow_id,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=self._rng.randint(32768, 60999),
                dst_port=_NCCL_PORT,
                protocol=6,          # TCP
                size_bytes=cfg.shard_bytes,
                arrival_time=t,
            ))
        return flows

    # ── Parameter-Server AllReduce ────────────────────────────────────────────

    def _ps_allreduce_flows(self, iteration: int, t: float) -> List[Flow]:
        """
        Generate one PS AllReduce iteration:
          Phase 1: each worker uploads gradient shard to PS (n flows).
          Phase 2: PS broadcasts updated parameters to each worker (n flows).

        Total flows: 2 × n_workers.
        """
        cfg = self.config
        n = cfg.n_workers
        flows: List[Flow] = []
        phase2_t = t + 0.001  # PS phase starts 1 ms after uploads

        for i, (w_ip) in enumerate(self._worker_ips):
            if w_ip == self._ps_ip:
                continue   # PS doesn't upload to itself

            # Phase 1: worker → PS
            flows.append(Flow(
                flow_id=f"ar_ps_it{iteration:04d}_w{i}_up",
                src_ip=w_ip,
                dst_ip=self._ps_ip,
                src_port=self._rng.randint(32768, 60999),
                dst_port=_NCCL_PORT,
                protocol=6,
                size_bytes=cfg.shard_bytes,
                arrival_time=t,
            ))

            # Phase 2: PS → worker
            flows.append(Flow(
                flow_id=f"ar_ps_it{iteration:04d}_w{i}_dn",
                src_ip=self._ps_ip,
                dst_ip=w_ip,
                src_port=_NCCL_PORT,
                dst_port=self._rng.randint(32768, 60999),
                protocol=6,
                size_bytes=cfg.shard_bytes,
                arrival_time=phase2_t,
            ))
        return flows

    # ── Pipeline parallelism ──────────────────────────────────────────────────

    def _pipeline_flows(self, iteration: int, t: float) -> List[Flow]:
        """
        Generate pipeline-parallelism activation transfers between stages.

        Workers are split into ``pipeline_stages`` equal groups.
        Each adjacent pair of groups exchanges activations.
        """
        cfg = self.config
        n = cfg.n_workers
        stage_size = max(1, n // cfg.pipeline_stages)
        flows: List[Flow] = []

        for stage in range(cfg.pipeline_stages - 1):
            # Last worker in stage → first worker in stage+1 (forward).
            src_idx = min((stage + 1) * stage_size - 1, n - 1)
            dst_idx = min((stage + 1) * stage_size, n - 1)
            if src_idx == dst_idx:
                continue

            flows.append(Flow(
                flow_id=f"ar_pipe_it{iteration:04d}_s{stage}_fwd",
                src_ip=self._worker_ips[src_idx],
                dst_ip=self._worker_ips[dst_idx],
                src_port=self._rng.randint(32768, 60999),
                dst_port=_NCCL_PORT + 1,
                protocol=6,
                size_bytes=cfg.activation_bytes,
                arrival_time=t,
            ))

            # Backward: stage+1 → stage (gradient).
            flows.append(Flow(
                flow_id=f"ar_pipe_it{iteration:04d}_s{stage}_bwd",
                src_ip=self._worker_ips[dst_idx],
                dst_ip=self._worker_ips[src_idx],
                src_port=self._rng.randint(32768, 60999),
                dst_port=_NCCL_PORT + 1,
                protocol=6,
                size_bytes=cfg.activation_bytes,
                arrival_time=t + 0.0005,   # slight offset
            ))

        return flows
