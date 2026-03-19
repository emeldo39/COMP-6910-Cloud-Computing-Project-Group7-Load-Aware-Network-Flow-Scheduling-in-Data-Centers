"""
LAFS -- EWMA Predictor
======================
COMP 6910 -- Group 7

Exponentially Weighted Moving Average (EWMA) predictor for short-term
(<= 1 step ahead) link utilisation forecasting.

Algorithm
---------
Given a time series of observations u_0, u_1, ..., u_{t}:

    EWMA_t = alpha * u_t + (1 - alpha) * EWMA_{t-1}

Forecast for horizon h (all horizons):

    u_{t+h} = EWMA_t       (flat forecast: EWMA does not extrapolate trends)

Confidence interval (90%):

    CI = EWMA_t +/- 1.645 * sigma_residual * sqrt(h)

where sigma_residual is the running standard deviation of one-step
prediction residuals (|u_t - EWMA_{t-1}|).

Properties
----------
* O(1) update and O(1) prediction -- suitable for real-time use.
* alpha close to 1.0 => reacts quickly but is noisy (good for step changes).
* alpha close to 0.0 => smooth but slow to react (good for stable loads).
* Optimal alpha minimises one-step RMSE over a training window.

Reference
---------
Hunter, J. S. (1986). The exponentially weighted moving average.
Journal of Quality Technology, 18(4), 203-210.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple


# ── Default alpha value ───────────────────────────────────────────────────────
DEFAULT_ALPHA: float = 0.3      # reasonable default for traffic smoothing
_Z90: float = 1.645             # 90% confidence z-score
_MIN_RESIDUALS: int = 5         # minimum residuals needed for CI computation


class EWMAPredictor:
    """
    Single exponential smoothing (EWMA) predictor.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  Larger = faster reaction to changes.
    max_residuals : int
        Rolling window for residual variance computation.

    Examples
    --------
    >>> pred = EWMAPredictor(alpha=0.3)
    >>> for v in [0.1, 0.2, 0.3, 0.4]:
    ...     pred.update(v)
    >>> point, lo, hi = pred.predict(horizon=1)
    >>> point  # smoothed estimate
    0.234...
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        max_residuals: int = 200,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._ewma: Optional[float] = None          # current EWMA value
        self._residuals: deque[float] = deque(maxlen=max_residuals)
        self._n_updates: int = 0

    # ── Online update ────────────────────────────────────────────────────────

    def update(self, value: float) -> float:
        """
        Incorporate a new observation and return the updated EWMA.

        Parameters
        ----------
        value : float
            New observed utilisation (any non-negative float).

        Returns
        -------
        float
            Updated EWMA value.
        """
        if self._ewma is None:
            self._ewma = value
        else:
            residual = abs(value - self._ewma)
            self._residuals.append(residual)
            self._ewma = self.alpha * value + (1.0 - self.alpha) * self._ewma
        self._n_updates += 1
        return self._ewma

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, horizon: int = 1) -> Tuple[float, float, float]:
        """
        Return a point forecast and 90% confidence interval for h steps ahead.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to predict (>= 1).

        Returns
        -------
        (point, lo_90, hi_90) : (float, float, float)
            point  -- EWMA value (flat forecast for all horizons)
            lo_90  -- lower bound of 90% prediction interval
            hi_90  -- upper bound of 90% prediction interval

        Notes
        -----
        The forecast is always the current EWMA value regardless of horizon.
        Confidence intervals widen with sqrt(horizon) following standard
        EWMA theory.  Returns (0.0, 0.0, 0.0) before any data is seen.
        """
        if self._ewma is None:
            return (0.0, 0.0, 0.0)

        point = self._ewma
        sigma = self._sigma_residual()
        half_width = _Z90 * sigma * math.sqrt(max(1, horizon))

        lo = max(0.0, point - half_width)
        hi = point + half_width
        return (point, lo, hi)

    # ── Batch fitting ─────────────────────────────────────────────────────────

    def fit(self, series: List[float]) -> "EWMAPredictor":
        """
        Fit EWMA to a full historical series.

        Processes observations in order, updating the internal state.
        Previous state is cleared before fitting.

        Parameters
        ----------
        series : List[float]
            Historical utilisation values in chronological order.

        Returns
        -------
        self
        """
        self.reset()
        for v in series:
            self.update(v)
        return self

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def current_estimate(self) -> Optional[float]:
        """Current EWMA value, or None if no data seen yet."""
        return self._ewma

    @property
    def n_updates(self) -> int:
        """Total number of update() calls made."""
        return self._n_updates

    @property
    def rmse(self) -> float:
        """
        Root mean squared one-step prediction error.

        Each residual is |u_t - EWMA_{t-1}|.  Returns 0.0 if fewer than
        two observations have been seen.
        """
        r = list(self._residuals)
        if not r:
            return 0.0
        return math.sqrt(sum(x * x for x in r) / len(r))

    @property
    def mape(self) -> float:
        """
        Mean absolute percentage error of one-step predictions.

        Skips windows where the observed value is 0 (undefined percentage).
        Returns 0.0 if no non-zero residuals exist.
        """
        # MAPE requires actuals -- stored residuals are absolute errors.
        # Use RMSE / mean_ewma as a proxy.
        if self._ewma is None or self._ewma == 0.0:
            return 0.0
        return self.rmse / max(self._ewma, 1e-9)

    def reset(self) -> None:
        """Clear all state (EWMA value and residual history)."""
        self._ewma = None
        self._residuals.clear()
        self._n_updates = 0

    # ── Alpha selection ───────────────────────────────────────────────────────

    @staticmethod
    def optimal_alpha(
        series: List[float],
        candidates: Optional[List[float]] = None,
    ) -> float:
        """
        Find the alpha that minimises one-step RMSE on *series*.

        Parameters
        ----------
        series : List[float]
            Training time series (needs >= 10 points for reliable results).
        candidates : List[float], optional
            Alpha values to try.  Default: 19 values from 0.05 to 0.95.

        Returns
        -------
        float
            Best alpha from the candidate set.

        Notes
        -----
        Uses leave-one-out style evaluation: for each candidate alpha, fits
        EWMA on series[:-1] and evaluates prediction error on series[-1].
        For a robust estimate, fits on first 80% and evaluates on last 20%.
        """
        if len(series) < 5:
            return DEFAULT_ALPHA

        if candidates is None:
            candidates = [round(i * 0.05, 2) for i in range(1, 20)]

        split = max(3, int(len(series) * 0.8))
        train, test = series[:split], series[split:]

        best_alpha, best_rmse = DEFAULT_ALPHA, float("inf")
        for alpha in candidates:
            pred = EWMAPredictor(alpha=alpha)
            pred.fit(train)
            errors = []
            for v in test:
                p, _, _ = pred.predict(horizon=1)
                errors.append((v - p) ** 2)
                pred.update(v)
            if errors:
                rmse = math.sqrt(sum(errors) / len(errors))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_alpha = alpha

        return best_alpha

    # ── Internal ─────────────────────────────────────────────────────────────

    def _sigma_residual(self) -> float:
        """Residual standard deviation; returns 0.0 if too few samples."""
        r = list(self._residuals)
        if len(r) < _MIN_RESIDUALS:
            return 0.0
        mean_r = sum(r) / len(r)
        return math.sqrt(sum((x - mean_r) ** 2 for x in r) / len(r))

    def __repr__(self) -> str:
        return (
            f"EWMAPredictor(alpha={self.alpha}, "
            f"ewma={self._ewma:.4f if self._ewma is not None else 'None'}, "
            f"n={self._n_updates})"
        )
