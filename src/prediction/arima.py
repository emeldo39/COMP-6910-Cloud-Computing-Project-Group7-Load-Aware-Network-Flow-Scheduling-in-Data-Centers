"""
LAFS -- AR / ARIMA Predictor
=============================
COMP 6910 -- Group 7

Autoregressive predictor for medium-term (1-10 step) link utilisation
forecasting.

Two implementations are provided:

  ARPredictor  (pure numpy, always available)
    Fits AR(p) via ordinary least squares.
    Used as the guaranteed fallback.

  ARIMAPredictor  (statsmodels preferred, ARPredictor fallback)
    Attempts to use statsmodels.tsa.arima.model.ARIMA(p, d, q).
    If statsmodels is unavailable or fitting fails, falls back to ARPredictor.

AR(p) algorithm
---------------
Given differenced (if d>0) series z_0, ..., z_{n-1}:

    z_t = c + phi_1*z_{t-1} + phi_2*z_{t-2} + ... + phi_p*z_{t-p} + eps_t

Fit: OLS via numpy.linalg.lstsq on the lagged design matrix.

h-step prediction: recursive substitution of predicted values for future
unknowns.

Confidence interval:
    half-width = 1.645 * sigma_eps * sqrt(h)
    (rough approximation; statsmodels gives exact intervals when used)

Minimum data requirement
------------------------
AR(p) needs at least p + 5 observations to produce a reliable fit.
Before that threshold is reached, the predictor returns the series mean
with a wide confidence interval.

Reference
---------
Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015).
Time Series Analysis: Forecasting and Control (5th ed.). Wiley.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

_log = logging.getLogger("lafs.prediction.arima")

# ── statsmodels: optional import ─────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA
    _STATSMODELS_OK = True
except ImportError:                         # pragma: no cover
    _STATSMODELS_OK = False
    _log.debug("statsmodels not found -- ARIMAPredictor will use AR fallback")

_Z90: float = 1.645


# =============================================================================
# ARPredictor  --  pure-numpy AR(p)
# =============================================================================

class ARPredictor:
    """
    Pure-numpy AR(p) predictor via ordinary least squares.

    Parameters
    ----------
    p : int
        Autoregressive order (number of lags).  Default 2.
    d : int
        Differencing order (0 = no differencing, 1 = first difference).
        Differencing is applied before fitting and inverted after prediction.

    Examples
    --------
    >>> pred = ARPredictor(p=2)
    >>> pred.fit([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3])
    >>> point, lo, hi = pred.predict(horizon=2)
    """

    def __init__(self, p: int = 2, d: int = 1) -> None:
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        if d not in (0, 1):
            raise ValueError(f"d must be 0 or 1, got {d}")
        self.p = p
        self.d = d
        self._coeffs: Optional[np.ndarray] = None    # [const, phi_1, ..., phi_p]
        self._sigma: float = 0.0                     # residual std dev
        self._history: List[float] = []              # last p values of working series
        self._last_raw: Optional[float] = None       # last raw (undifferenced) value
        self._mean: float = 0.0                      # fallback: series mean

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, series: List[float]) -> "ARPredictor":
        """
        Fit AR(p) to *series* using OLS.

        Parameters
        ----------
        series : List[float]
            Historical observations in chronological order.
            Needs at least p + 5 + d values for a meaningful fit.

        Returns
        -------
        self
        """
        raw = list(series)
        self._mean = float(np.mean(raw)) if raw else 0.0
        self._last_raw = raw[-1] if raw else 0.0

        # Apply differencing.
        work = self._difference(raw)

        n = len(work)
        min_needed = self.p + 3
        if n < min_needed:
            # Too little data: store mean as trivial forecast.
            self._coeffs = None
            self._sigma = float(np.std(work)) if work else 0.0
            self._history = list(work[-self.p:]) if work else []
            return self

        # Build lagged design matrix (with intercept).
        # X[t] = [1, work[t-1], work[t-2], ..., work[t-p]]
        # for t = p, p+1, ..., n-1
        X = np.ones((n - self.p, self.p + 1))
        for lag in range(self.p):
            X[:, lag + 1] = work[self.p - lag - 1: n - lag - 1]
        y = np.array(work[self.p:])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self._coeffs = coeffs

        # Compute residual sigma.
        y_hat = X @ coeffs
        residuals = y - y_hat
        self._sigma = float(np.std(residuals)) if len(residuals) > 0 else 0.0

        # Store last p working-series values for recursive prediction.
        self._history = list(work[-self.p:])
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, horizon: int = 1) -> Tuple[float, float, float]:
        """
        Return (point, lo_90, hi_90) for h steps ahead.

        Uses recursive AR prediction: each step's forecast is substituted
        into the next step's lag window.

        Returns (mean, mean-wide_ci, mean+wide_ci) before fitting.
        """
        if self._coeffs is None:
            # Cold start or insufficient data.
            sigma = self._sigma or 0.1
            half = _Z90 * sigma * math.sqrt(max(1, horizon))
            return (self._mean, max(0.0, self._mean - half), self._mean + half)

        history = list(self._history)    # copy: we'll extend it with forecasts
        point = self._mean               # will be overwritten

        for h in range(1, horizon + 1):
            # Build feature vector: [1, history[-1], history[-2], ..., history[-p]]
            feats = np.ones(self.p + 1)
            for lag in range(self.p):
                idx = -(lag + 1)
                feats[lag + 1] = history[idx] if len(history) >= abs(idx) else 0.0
            z_hat = float(feats @ self._coeffs)
            history.append(z_hat)
            if h == horizon:
                point = z_hat

        # Invert differencing for the point estimate.
        if self.d == 1 and self._last_raw is not None:
            # One-step inversion for h=1; for h>1 approximate via cumsum.
            # For simplicity, cumulative sum of h predicted differences.
            diff_preds = history[len(self._history):]
            point = self._last_raw + sum(diff_preds[:horizon])

        # Confidence interval widens with sqrt(horizon).
        half = _Z90 * self._sigma * math.sqrt(max(1, horizon))
        return (point, max(0.0, point - half), point + half)

    # ── Online update ─────────────────────────────────────────────────────────

    def update(self, value: float) -> None:
        """
        Incorporate a new raw observation without refitting.

        Updates the history buffer so the next prediction uses the latest data.
        """
        if self.d == 1 and self._last_raw is not None:
            diff = value - self._last_raw
            if len(self._history) >= self.p:
                self._history.pop(0)
            self._history.append(diff)
        else:
            if len(self._history) >= self.p:
                self._history.pop(0)
            self._history.append(value)
        self._last_raw = value

    # ── Internal ─────────────────────────────────────────────────────────────

    def _difference(self, series: List[float]) -> List[float]:
        """Apply d-order differencing."""
        if self.d == 0 or len(series) < 2:
            return list(series)
        return [series[i] - series[i - 1] for i in range(1, len(series))]

    def __repr__(self) -> str:
        fitted = self._coeffs is not None
        return f"ARPredictor(p={self.p}, d={self.d}, fitted={fitted})"


# =============================================================================
# ARIMAPredictor  --  statsmodels ARIMA with ARPredictor fallback
# =============================================================================

class ARIMAPredictor:
    """
    ARIMA(p, d, q) predictor.

    Uses statsmodels.tsa.arima.model.ARIMA when available and the series
    has enough data.  Falls back to ARPredictor(p, d) otherwise.

    Parameters
    ----------
    p : int
        AR order.  Default 2.
    d : int
        Integration (differencing) order.  Default 1.
    q : int
        MA order.  Default 1.
    min_fit_samples : int
        Minimum series length required to attempt ARIMA fitting.
        Below this, ARPredictor is used.  Default 20.
    refit_every : int
        Re-fit the full ARIMA model every N update() calls.
        Between refits, the AR fallback handles online updates.
        Default 30.

    Examples
    --------
    >>> pred = ARIMAPredictor(p=2, d=1, q=1)
    >>> pred.fit([0.1, 0.15, 0.22, 0.18, 0.25, 0.3, 0.28, 0.35,
    ...           0.32, 0.40, 0.38, 0.42, 0.45, 0.48, 0.50,
    ...           0.52, 0.49, 0.55, 0.53, 0.58, 0.60])
    >>> point, lo, hi = pred.predict(horizon=3)
    """

    def __init__(
        self,
        p: int = 2,
        d: int = 1,
        q: int = 1,
        min_fit_samples: int = 20,
        refit_every: int = 30,
    ) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.min_fit_samples = min_fit_samples
        self.refit_every = refit_every

        self._fallback = ARPredictor(p=p, d=d)
        self._sm_result = None       # fitted statsmodels ARIMAResults object
        self._series: List[float] = []
        self._updates_since_refit: int = 0
        self._use_statsmodels: bool = _STATSMODELS_OK

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, series: List[float]) -> "ARIMAPredictor":
        """
        Fit ARIMA to *series*.

        Attempts statsmodels ARIMA if available and data is sufficient.
        Always fits the fallback ARPredictor as well.
        """
        self._series = list(series)
        self._fallback.fit(series)
        self._sm_result = None
        self._updates_since_refit = 0

        if self._use_statsmodels and len(series) >= self.min_fit_samples:
            self._fit_statsmodels(series)

        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, horizon: int = 1) -> Tuple[float, float, float]:
        """
        Return (point, lo_90, hi_90) for h steps ahead.

        Uses statsmodels forecast if fitted, else falls back to ARPredictor.
        """
        if self._sm_result is not None:
            try:
                return self._statsmodels_predict(horizon)
            except Exception as exc:
                _log.debug("statsmodels predict failed (%s), using fallback", exc)
        return self._fallback.predict(horizon)

    # ── Online update ─────────────────────────────────────────────────────────

    def update(self, value: float) -> None:
        """
        Incorporate a new observation online.

        Re-fits the full ARIMA model every ``refit_every`` updates.
        """
        self._series.append(value)
        self._fallback.update(value)
        self._updates_since_refit += 1

        if (
            self._use_statsmodels
            and self._updates_since_refit >= self.refit_every
            and len(self._series) >= self.min_fit_samples
        ):
            self._fit_statsmodels(self._series)
            self._updates_since_refit = 0

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def using_statsmodels(self) -> bool:
        """True if the last fit used statsmodels successfully."""
        return self._sm_result is not None

    @property
    def n_observations(self) -> int:
        return len(self._series)

    def __repr__(self) -> str:
        backend = "statsmodels" if self.using_statsmodels else "AR fallback"
        return (
            f"ARIMAPredictor(p={self.p}, d={self.d}, q={self.q}, "
            f"n={self.n_observations}, backend={backend})"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fit_statsmodels(self, series: List[float]) -> None:
        """Attempt to fit statsmodels ARIMA; suppress and log failures."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _SM_ARIMA(
                    series,
                    order=(self.p, self.d, self.q),
                    trend="n",                      # no deterministic trend term
                )
                self._sm_result = model.fit(disp=False)
        except Exception as exc:
            _log.debug("statsmodels ARIMA fit failed (%s)", exc)
            self._sm_result = None

    def _statsmodels_predict(self, horizon: int) -> Tuple[float, float, float]:
        """Extract forecast and confidence interval from a fitted ARIMA result."""
        fc = self._sm_result.get_forecast(steps=horizon)
        mean_series = fc.predicted_mean
        ci = fc.conf_int(alpha=0.10)    # 90% CI

        # Take the h-th step (last row for h-step ahead).
        point = float(mean_series.iloc[-1])
        lo    = float(ci.iloc[-1, 0])
        hi    = float(ci.iloc[-1, 1])
        return (point, max(0.0, lo), hi)
