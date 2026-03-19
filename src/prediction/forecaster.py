"""
LAFS -- Load Forecaster
========================
COMP 6910 -- Group 7

Main prediction entry point: fits per-link predictors to time-series data
from LinkLoadSampler and produces NetworkLoadForecast objects that the
scheduler and MILP optimizer consume.

Architecture
------------
                    LinkLoadSampler
                          |
                  (per-link LinkLoadSeries)
                          |
                    LoadForecaster
                    /    |     \\
              EWMA   ARIMA   Hybrid   (one predictor per active link)
                    \\   |    /
                  NetworkLoadForecast
                    /         \\
          Scheduler             MILP Optimizer
      (path_max_utilisation)   (capacity constraints)

Usage
-----
    sampler = LinkLoadSampler(topo, window_s=0.1)
    sampler.ingest(flows)
    sampler.build_series()

    forecaster = LoadForecaster(topo, method="hybrid", horizon_s=0.3)
    forecaster.fit(sampler)

    forecast = forecaster.predict()
    print(forecast.congested_links(threshold=0.7))
    print(forecast.path_max_utilisation(["h_0_0_0", "e_0_0", "a_0_1"]))
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.topology.fattree import FatTreeGraph
from src.metrics.link_load import LinkLoadSampler, LinkLoadSeries
from src.prediction.ewma import EWMAPredictor
from src.prediction.arima import ARIMAPredictor
from src.prediction.hybrid import HybridPredictor


# ── Valid prediction methods ──────────────────────────────────────────────────
VALID_METHODS = frozenset({"ewma", "arima", "hybrid"})

# ── Minimum observations before a predictor produces non-trivial forecasts ───
_MIN_OBS_EWMA:  int = 3
_MIN_OBS_ARIMA: int = 20
_MIN_OBS_HYBRID: int = 5


# =============================================================================
# LinkLoadForecast  --  single-link prediction result
# =============================================================================

@dataclass
class LinkLoadForecast:
    """
    Predicted utilisation for one directed link at a specific future time.

    Attributes
    ----------
    link : (str, str)
        Directed link as (node_a, node_b).
    t_predict : float
        Wall-clock (or simulation) time when the prediction was made.
    horizon_s : float
        How far ahead the forecast looks, in seconds.
    predicted_utilisation : float
        Point estimate of link utilisation at t_predict + horizon_s.
        A value of 1.0 means the link is predicted to be fully saturated.
    confidence_lo : float
        Lower bound of the 90% prediction interval (>= 0).
    confidence_hi : float
        Upper bound of the 90% prediction interval.
    method : str
        Prediction method used: 'ewma', 'arima', or 'hybrid'.
    n_samples : int
        Number of historical samples used to fit this predictor.
    """
    link: Tuple[str, str]
    t_predict: float
    horizon_s: float
    predicted_utilisation: float
    confidence_lo: float
    confidence_hi: float
    method: str
    n_samples: int = 0

    @property
    def is_congested(self) -> bool:
        """True if the point estimate exceeds 0.7 (default LAFS threshold)."""
        return self.predicted_utilisation >= 0.7

    @property
    def uncertainty(self) -> float:
        """Half-width of the 90% confidence interval."""
        return (self.confidence_hi - self.confidence_lo) / 2.0

    def __repr__(self) -> str:
        return (
            f"LinkLoadForecast({self.link[0]}->{self.link[1]}, "
            f"util={self.predicted_utilisation:.3f} "
            f"[{self.confidence_lo:.3f},{self.confidence_hi:.3f}], "
            f"h={self.horizon_s*1e3:.0f}ms)"
        )


# =============================================================================
# NetworkLoadForecast  --  full-network prediction snapshot
# =============================================================================

@dataclass
class NetworkLoadForecast:
    """
    Predicted utilisation for all active links in the network.

    Produced by LoadForecaster.predict() and consumed by the scheduler
    (for path selection) and the MILP optimizer (for capacity constraints).

    Attributes
    ----------
    t_predict : float
        Prediction timestamp.
    horizon_s : float
        Forecast horizon in seconds.
    forecasts : Dict[(str,str), LinkLoadForecast]
        Per-link forecasts keyed by directed link tuple.
    method : str
        Prediction method used globally.
    """
    t_predict: float
    horizon_s: float
    forecasts: Dict[Tuple[str, str], LinkLoadForecast] = field(default_factory=dict)
    method: str = "hybrid"

    # ── Query API (used by scheduler / MILP) ─────────────────────────────────

    def utilisation(self, node_a: str, node_b: str) -> float:
        """
        Return the predicted utilisation for the directed link (a, b).

        Returns 0.0 if the link was not observed / has no forecast.
        """
        fc = self.forecasts.get((node_a, node_b))
        return fc.predicted_utilisation if fc is not None else 0.0

    def confidence_hi(self, node_a: str, node_b: str) -> float:
        """Upper bound of the 90% CI for link (a, b). Returns 0.0 if unknown."""
        fc = self.forecasts.get((node_a, node_b))
        return fc.confidence_hi if fc is not None else 0.0

    def path_max_utilisation(self, path: List[str]) -> float:
        """
        Return the bottleneck (maximum) predicted utilisation along *path*.

        Parameters
        ----------
        path : List[str]
            Ordered node names forming the path (e.g. from FatTreeGraph.get_paths).

        Returns
        -------
        float
            max utilisation over all directed links in the path.
            Returns 0.0 if the path has fewer than 2 nodes.
        """
        if len(path) < 2:
            return 0.0
        return max(
            self.utilisation(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    def path_max_confidence_hi(self, path: List[str]) -> float:
        """Conservative bottleneck: max upper-CI utilisation along *path*."""
        if len(path) < 2:
            return 0.0
        return max(
            self.confidence_hi(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    def congested_links(self, threshold: float = 0.7) -> List[Tuple[str, str]]:
        """
        Return directed links predicted to exceed *threshold* utilisation.

        Parameters
        ----------
        threshold : float
            Congestion threshold (default 0.7 = 70% utilisation).
        """
        return [
            link for link, fc in self.forecasts.items()
            if fc.predicted_utilisation >= threshold
        ]

    def congested_links_conservative(
        self, threshold: float = 0.7
    ) -> List[Tuple[str, str]]:
        """
        Return links where the upper CI bound exceeds *threshold*.

        More conservative than congested_links() -- useful when the MILP
        optimizer wants to leave safety margin.
        """
        return [
            link for link, fc in self.forecasts.items()
            if fc.confidence_hi >= threshold
        ]

    def least_congested_path(
        self, paths: List[List[str]], conservative: bool = False
    ) -> Optional[List[str]]:
        """
        Return the path from *paths* with the lowest bottleneck utilisation.

        Parameters
        ----------
        paths : List[List[str]]
            Candidate paths (e.g. all ECMP paths for a src-dst pair).
        conservative : bool
            If True, rank by upper-CI utilisation instead of point estimate.

        Returns
        -------
        List[str] or None
            The best path, or None if paths is empty.
        """
        if not paths:
            return None
        if conservative:
            return min(paths, key=self.path_max_confidence_hi)
        return min(paths, key=self.path_max_utilisation)

    def summary(self) -> str:
        """Return a human-readable summary of the forecast."""
        n = len(self.forecasts)
        if n == 0:
            return "NetworkLoadForecast (empty)"
        utils = [fc.predicted_utilisation for fc in self.forecasts.values()]
        congested = len(self.congested_links())
        lines = [
            f"NetworkLoadForecast @ t={self.t_predict:.3f}s, "
            f"horizon={self.horizon_s * 1e3:.0f}ms, method={self.method}",
            f"  Links forecasted   : {n}",
            f"  Mean utilisation   : {sum(utils)/n:.3f}",
            f"  Max utilisation    : {max(utils):.3f}",
            f"  Congested (>=0.7)  : {congested} ({100*congested/n:.1f}%)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"NetworkLoadForecast(links={len(self.forecasts)}, "
            f"h={self.horizon_s*1e3:.0f}ms, method={self.method})"
        )


# =============================================================================
# LoadForecaster  --  fits predictors, produces NetworkLoadForecast
# =============================================================================

class LoadForecaster:
    """
    Orchestrates per-link predictor fitting and produces network-wide forecasts.

    One predictor (EWMA, ARIMA, or Hybrid) is maintained per active directed
    link.  Predictors are fit on historical LinkLoadSeries data from a
    LinkLoadSampler.

    Parameters
    ----------
    topology : FatTreeGraph
        Network topology (used for link enumeration and capacity info).
    method : str
        Prediction method: 'ewma', 'arima', or 'hybrid'.
    horizon_s : float
        Forecast horizon in seconds.  Must be a multiple of window_s.
    window_s : float
        Sampling window width in seconds.  Must match the LinkLoadSampler's
        window_s.
    ewma_alpha : float or None
        EWMA smoothing factor.  None triggers auto-selection during fit().
    arima_p, arima_d, arima_q : int
        ARIMA order parameters.

    Examples
    --------
    >>> topo      = FatTreeGraph(k=4)
    >>> sampler   = LinkLoadSampler(topo, window_s=0.1)
    >>> sampler.ingest(flows)
    >>> sampler.build_series()
    >>> forecaster = LoadForecaster(topo, method="hybrid", horizon_s=0.3)
    >>> forecaster.fit(sampler)
    >>> forecast = forecaster.predict()
    >>> paths = topo.get_paths("h_0_0_0", "h_1_0_0")
    >>> best = forecast.least_congested_path(paths)
    """

    def __init__(
        self,
        topology: FatTreeGraph,
        method: str = "hybrid",
        horizon_s: float = 0.1,
        window_s: float = 0.1,
        ewma_alpha: Optional[float] = None,
        arima_p: int = 2,
        arima_d: int = 1,
        arima_q: int = 1,
    ) -> None:
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(VALID_METHODS)}, got {method!r}"
            )
        self.topology = topology
        self.method = method
        self.horizon_s = horizon_s
        self.window_s = window_s
        self._ewma_alpha = ewma_alpha
        self._arima_p = arima_p
        self._arima_d = arima_d
        self._arima_q = arima_q

        # horizon in steps (must be >= 1).
        self._horizon_steps = max(1, round(horizon_s / window_s))

        # Per-link predictors (lazy; created when a link's series is seen).
        self._predictors: Dict[Tuple[str, str], object] = {}
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, sampler: LinkLoadSampler) -> "LoadForecaster":
        """
        Fit a predictor for every active link in *sampler*.

        Links with too few observations for the chosen method fall back
        to EWMA (for 'arima') or use the raw series mean (for 'ewma').

        Parameters
        ----------
        sampler : LinkLoadSampler
            A sampler that has already had ingest() and build_series() called.

        Returns
        -------
        self
        """
        for link, series in sampler.all_series().items():
            values = series.values()
            if not values:
                continue
            pred = self._make_predictor(method=self.method)
            pred.fit(values)
            self._predictors[link] = pred

        self._fitted = True
        return self

    def predict(self, t_now: Optional[float] = None) -> NetworkLoadForecast:
        """
        Produce a NetworkLoadForecast for all fitted links.

        Parameters
        ----------
        t_now : float, optional
            Prediction timestamp.  Defaults to time.time().

        Returns
        -------
        NetworkLoadForecast
        """
        if t_now is None:
            t_now = time.time()

        forecasts: Dict[Tuple[str, str], LinkLoadForecast] = {}
        h = self._horizon_steps

        for link, pred in self._predictors.items():
            point, lo, hi = pred.predict(horizon=h)

            # Clamp to [0, ∞).
            point = max(0.0, point)
            lo    = max(0.0, lo)
            hi    = max(0.0, hi)

            n_obs = getattr(pred, "n_updates", 0) or getattr(
                pred, "_n_updates", 0
            )
            # Fallback: check EWMA sub-predictor.
            if n_obs == 0 and hasattr(pred, "_ewma"):
                n_obs = pred._ewma.n_updates

            forecasts[link] = LinkLoadForecast(
                link=link,
                t_predict=t_now,
                horizon_s=self.horizon_s,
                predicted_utilisation=point,
                confidence_lo=lo,
                confidence_hi=hi,
                method=self.method,
                n_samples=n_obs,
            )

        return NetworkLoadForecast(
            t_predict=t_now,
            horizon_s=self.horizon_s,
            forecasts=forecasts,
            method=self.method,
        )

    def update(self, new_flows: List, t_now: Optional[float] = None) -> NetworkLoadForecast:
        """
        Ingest new flows into existing predictors (online update) and predict.

        Parameters
        ----------
        new_flows : List[Flow]
            Recently scheduled flows (with assigned_path set).
        t_now : float, optional
            Current simulation time (used to attribute flows to windows).

        Returns
        -------
        NetworkLoadForecast
            Updated forecast after incorporating new flows.
        """
        from src.metrics.link_load import LinkLoadSampler as _Sampler
        tmp = _Sampler(self.topology, window_s=self.window_s)
        tmp.ingest(new_flows)
        tmp.build_series()

        for link, series in tmp.all_series().items():
            values = series.values()
            if not values:
                continue
            if link not in self._predictors:
                pred = self._make_predictor(method=self.method)
                self._predictors[link] = pred
            pred = self._predictors[link]
            for v in values:
                pred.update(v)

        return self.predict(t_now)

    def evaluate(
        self,
        actuals: Dict[Tuple[str, str], List[float]],
        predictions: Dict[Tuple[str, str], List[float]],
    ) -> Dict[str, float]:
        """
        Compute MAPE and RMSE for a set of (actual, predicted) pairs.

        Parameters
        ----------
        actuals : Dict[link, List[float]]
            Actual utilisation values per link.
        predictions : Dict[link, List[float]]
            Corresponding predicted values.

        Returns
        -------
        Dict with keys: 'mape', 'rmse', 'mae', 'n_links', 'n_samples'.
        """
        all_abs_err, all_sq_err, all_pct_err = [], [], []

        for link in actuals:
            if link not in predictions:
                continue
            for a, p in zip(actuals[link], predictions[link]):
                ae = abs(a - p)
                all_abs_err.append(ae)
                all_sq_err.append(ae ** 2)
                if a > 1e-6:
                    all_pct_err.append(ae / a)

        n = len(all_abs_err)
        if n == 0:
            return {"mape": 0.0, "rmse": 0.0, "mae": 0.0,
                    "n_links": 0, "n_samples": 0}

        mape = (sum(all_pct_err) / len(all_pct_err) * 100) if all_pct_err else 0.0
        rmse = math.sqrt(sum(all_sq_err) / n)
        mae  = sum(all_abs_err) / n

        return {
            "mape": round(mape, 4),
            "rmse": round(rmse, 6),
            "mae":  round(mae, 6),
            "n_links":   len(actuals),
            "n_samples": n,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_fitted_links(self) -> int:
        """Number of links with fitted predictors."""
        return len(self._predictors)

    @property
    def horizon_steps(self) -> int:
        """Forecast horizon in number of sampling windows."""
        return self._horizon_steps

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_predictor(self, method: str):
        """Instantiate a fresh predictor of the specified type."""
        if method == "ewma":
            return EWMAPredictor(alpha=self._ewma_alpha or 0.3)
        elif method == "arima":
            return ARIMAPredictor(
                p=self._arima_p, d=self._arima_d, q=self._arima_q
            )
        else:  # "hybrid"
            return HybridPredictor(
                ewma_alpha=self._ewma_alpha,
                arima_p=self._arima_p,
                arima_d=self._arima_d,
                arima_q=self._arima_q,
            )

    def __repr__(self) -> str:
        return (
            f"LoadForecaster(method={self.method!r}, "
            f"horizon={self.horizon_s*1e3:.0f}ms, "
            f"fitted_links={self.n_fitted_links})"
        )
