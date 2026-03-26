"""
LAFS -- Load-Aware Flow Scheduler
====================================
COMP 6910 -- Group 7

LAFSScheduler integrates three components into one scheduler:

    Prediction  (LoadForecaster / NetworkLoadForecast)
          |
          v
    MILP solver (LAFSMILPSolver)
          |
          v
    Path assignment on Flow objects

The scheduler implements BaseScheduler so it is drop-in compatible with
ECMPScheduler, HederaScheduler, and CONGAScheduler.

Single-flow API (schedule_flow):
    Falls back to the first ECMP candidate path.  The MILP is designed for
    batch scheduling windows, not one-at-a-time requests.

Batch API (schedule_flows / schedule_flows_milp):
    1. Collect candidate paths for every flow via get_candidate_paths().
    2. Extract link capacities from the topology graph.
    3. Get predicted utilisation from the attached LoadForecaster (if any).
    4. Call LAFSMILPSolver.solve() to get optimal assignments.
    5. Stamp assigned_path and schedule_time on each Flow object.
    6. Update BaseScheduler metrics.
    7. Return a MILPResult with solve metadata.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from src.scheduler.base_scheduler import BaseScheduler
from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow
from src.optimizer.milp_solver import LAFSMILPSolver, MILPConfig, MILPResult

_log = logging.getLogger("lafs.scheduler.lafs")

# Topology link-capacity constants (must match fattree.py)
_BW_HOST_BPS = 1e9    # 1 Gbps  -- host-to-edge links
_BW_FABRIC_BPS = 10e9  # 10 Gbps -- all fabric (agg/core) links


class LAFSScheduler(BaseScheduler):
    """
    Load-Aware Flow Scheduling using MILP-based path placement.

    Parameters
    ----------
    topology : FatTreeGraph
        Fat-tree topology shared with other schedulers.
    milp_config : MILPConfig or None
        MILP solver configuration.  If None, default config is used
        (reads LAFS_SOLVER environment variable).
    forecaster : LoadForecaster or None
        Pre-fitted load forecaster.  When provided, predicted link
        utilisations are passed to the MILP solver.  When None, all
        predicted utilisations are treated as zero (pure load-balancing
        without forecast).
    window_s : float
        Scheduling window length in seconds (default 0.1 s = 100 ms).
        Controls bandwidth demand = size_bytes * 8 / window_s.
    n_paths_limit : int or None
        If set, truncate each flow's candidate path list to this many
        paths.  Reduces MILP variable count at the cost of coverage.
        None means use all available paths.

    Examples
    --------
    ::

        topo = FatTreeGraph(k=4)
        sched = LAFSScheduler(topo)
        result = sched.schedule_flows_milp(flows)
        print(result.summary())
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        milp_config: Optional[MILPConfig] = None,
        forecaster=None,     # Optional[LoadForecaster] -- avoid circular import
        window_s: float = 0.1,
        n_paths_limit: Optional[int] = None,
    ) -> None:
        super().__init__(topology)
        self._solver = LAFSMILPSolver(milp_config)
        self._forecaster = forecaster
        self._window_s = window_s
        self._n_paths_limit = n_paths_limit
        # Cache topology link capacities once at construction
        self._link_capacities: Dict[Tuple[str, str], float] = (
            self._build_link_capacities()
        )
        _log.debug(
            "LAFSScheduler ready: solver=%s window=%.3fs links=%d",
            self._solver.config.solver,
            self._window_s,
            len(self._link_capacities),
        )

    # ------------------------------------------------------------------
    # BaseScheduler interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "lafs"

    def schedule_flow(self, flow: Flow) -> Optional[List[str]]:
        """
        Schedule a single flow using ECMP candidate path (first available).

        The MILP is designed for batch scheduling; this method provides
        single-flow compatibility with the BaseScheduler API.
        """
        paths = self.get_candidate_paths(flow.src_ip, flow.dst_ip)
        if not paths:
            return None
        return paths[0]

    def schedule_flows(
        self, flows: List[Flow]
    ) -> Dict[str, Optional[List[str]]]:
        """
        Override BaseScheduler.schedule_flows to use batch MILP placement.

        Returns a plain dict[flow_id -> path] for API compatibility; for
        richer solve metadata use ``schedule_flows_milp`` directly.
        """
        result = self.schedule_flows_milp(flows)
        return {
            fid: path for fid, path in result.assignments.items()
        }

    # ------------------------------------------------------------------
    # Primary batch API
    # ------------------------------------------------------------------

    def schedule_flows_milp(self, flows: List[Flow]) -> MILPResult:
        """
        Schedule *flows* via the MILP solver and update Flow objects in-place.

        Steps
        -----
        1. Build per-flow candidate path lists.
        2. Obtain predicted link utilisations from the forecaster (or zeros).
        3. Invoke ``LAFSMILPSolver.solve()``.
        4. Stamp ``assigned_path`` and ``schedule_time`` on each flow.
        5. Update ``self.metrics`` (SchedulerMetrics).

        Parameters
        ----------
        flows : List[Flow]
            Flows to schedule.  Each flow must have a valid src_ip/dst_ip
            that exists in the topology.

        Returns
        -------
        MILPResult
            Solve metadata including status, max_utilisation, solve_time_s,
            and the full assignment dict.
        """
        t_batch_start = time.perf_counter()

        # -- Step 1: candidate paths per flow --
        candidate_paths: Dict[str, List[List[str]]] = {}
        flows_with_paths: List[Flow] = []
        flows_no_path: List[Flow] = []

        for flow in flows:
            paths = self.get_candidate_paths(flow.src_ip, flow.dst_ip)
            if self._n_paths_limit and len(paths) > self._n_paths_limit:
                paths = paths[: self._n_paths_limit]
            if paths:
                candidate_paths[flow.flow_id] = paths
                flows_with_paths.append(flow)
            else:
                flows_no_path.append(flow)
                self.metrics.record_failure()
                _log.debug(
                    "No paths for flow %s (%s -> %s)",
                    flow.flow_id, flow.src_ip, flow.dst_ip,
                )

        # -- Step 2: predicted link utilisation --
        predicted_utils = self._get_predicted_utils()

        # -- Step 3: solve --
        milp_result = self._solver.solve(
            flows=flows_with_paths,
            candidate_paths=candidate_paths,
            predicted_utils=predicted_utils,
            link_capacities=self._link_capacities,
            window_s=self._window_s,
        )

        # -- Step 4: stamp Flow objects --
        now = time.time()
        latency_per_flow = milp_result.solve_time_s / max(len(flows_with_paths), 1)

        for flow in flows_with_paths:
            path = milp_result.assignments.get(flow.flow_id)
            if path is not None:
                flow.assigned_path = path
                flow.schedule_time = now
                self.metrics.record_scheduled(flow, path, latency_per_flow)
            else:
                self.metrics.record_failure()

        _log.info(
            "MILP batch: %d flows, %d scheduled, %d no-path | %s",
            len(flows),
            len(milp_result.assignments),
            len(flows_no_path),
            milp_result.summary(),
        )

        return milp_result

    # ------------------------------------------------------------------
    # Forecaster integration
    # ------------------------------------------------------------------

    def attach_forecaster(self, forecaster) -> None:
        """Attach or replace the LoadForecaster used for predicted utils."""
        self._forecaster = forecaster

    def update_forecast(self, new_flows: List[Flow], t_now: Optional[float] = None):
        """
        Update the internal forecaster with newly-arrived flows and get an
        updated ``NetworkLoadForecast``.

        No-op if no forecaster is attached.
        """
        if self._forecaster is not None:
            return self._forecaster.update(new_flows, t_now)
        return None

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def _build_link_capacities(self) -> Dict[Tuple[str, str], float]:
        """
        Build a dict of (u, v) -> bps from the topology graph edge attributes.

        Host-edge links: 1 Gbps.  All fabric links: 10 Gbps.
        """
        caps: Dict[Tuple[str, str], float] = {}
        for u, v, data in self.topology.graph.edges(data=True):
            bw = data.get("bw", None)
            if bw is not None:
                bps = float(bw) * 1e9  # bw stored in Gbps
            else:
                # Infer from node type: host nodes contain no '_' after prefix
                u_is_host = u.startswith("h_")
                v_is_host = v.startswith("h_")
                bps = _BW_HOST_BPS if (u_is_host or v_is_host) else _BW_FABRIC_BPS
            # Store both directions (undirected topology)
            caps[(u, v)] = bps
            caps[(v, u)] = bps
        return caps

    def _get_predicted_utils(self) -> Dict[Tuple[str, str], float]:
        """
        Return predicted utilisation per directed link.

        Uses the attached forecaster's latest forecast, or returns an empty
        dict (all links treated as 0% utilised) if no forecaster is present.
        """
        if self._forecaster is None:
            return {}
        try:
            forecast = self._forecaster.predict()
            return {
                link: lf.predicted_utilisation
                for link, lf in forecast.forecasts.items()
            }
        except Exception as exc:
            _log.warning("Forecaster.predict() failed, using zero utils: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def milp_config_summary(self) -> str:
        """Return a one-line summary of the active solver configuration."""
        cfg = self._solver.config
        return (
            f"solver={cfg.solver} time_limit={cfg.time_limit_s}s "
            f"mip_gap={cfg.mip_gap} window={self._window_s}s"
        )
