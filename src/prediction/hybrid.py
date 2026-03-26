"""
LAFS -- Hybrid EWMA+ARIMA Predictor
=====================================
COMP-6910 -- Group 7

Combines EWMA (fast, low-latency) and ARIMA (autocorrelation-aware) predictors
into a single adaptive forecaster.

Blending strategy
-----------------
For horizon h = 1:
  Use EWMA (lowest latency, lowest variance for one-step).

For horizon h > 1:
  Use adaptive weighted blend:
    pred = w_ewma * ewma_pred + w_arima * arima_pred

  Weights are updated every `weight_update_interval` observations based on
  the recent MAPE of each model.  The model with lower recent error receives
  a higher weight.  Initial weights are equal (0.5 / 0.5).

  Weight update rule:
    err_ewma  = recent absolute error of EWMA model
    err_arima = recent absolute error of ARIMA model
    w_ewma    = err_arima / (err_ewma + err_arima)   # inverse-error weighting
    w_arima   = 1.0 - w_ewma

  If both models have zero error (constant series), weights stay at 0.5.

Rationale from proposal (Section 4.2)
--------------------------------------
"EWMA captures short-term load spikes (AllReduce synchronisation bursts
at sub-second scale) while ARIMA captures the periodic structure of ML
training traffic over multiple iterations."

Usage
-----
    pred = HybridPredictor(ewma_alpha=0.3, arima_p=2, arima_d=1, arima_q=1)
    pred.fit(series)
    point, lo, hi = pred.predict(horizon=3)   # 3 steps ahead
    pred.update(new_value)                    # online update
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

from src.prediction.ewma import EWMAPredictor, DEFAULT_ALPHA
from src.prediction.arima import ARIMAPredictor

_Z90: float = 1.645
_MIN_WEIGHT_HISTORY: int = 5   # minimum error samples before adapting weights


class HybridPredictor:
    """
    Adaptive EWMA + ARIMA hybrid predictor.

    Parameters
    ----------
    ewma_alpha : float
        EWMA smoothing factor.  If None, optimal alpha is found by grid search
        on the training series during fit().
    arima_p, arima_d, arima_q : int
        ARIMA order parameters.
    short_horizon : int
        Horizons <= this value use pure EWMA.  Default 1.
    weight_update_interval : int
        Re-compute blending weights every N update() calls.  Default 10.
    max_error_history : int
        Number of recent prediction errors to keep for weight computation.

    Examples
    --------
    >>> pred = HybridPredictor()
    >>> series = [0.2 + 0.1*math.sin(i*0.5) for i in range(50)]
    >>> pred.fit(series)
    >>> point, lo, hi = pred.predict(horizon=5)
    """

    def __init__(
        self,
        ewma_alpha: Optional[float] = None,
        arima_p: int = 2,
        arima_d: int = 1,
        arima_q: int = 1,
        short_horizon: int = 1,
        weight_update_interval: int = 10,
        max_error_history: int = 50,
    ) -> None:
        self._ewma_alpha_init = ewma_alpha
        self._ewma = EWMAPredictor(alpha=ewma_alpha or DEFAULT_ALPHA)
        self._arima = ARIMAPredictor(p=arima_p, d=arima_d, q=arima_q)
        self.short_horizon = short_horizon
        self.weight_update_interval = weight_update_interval

        # Blending weights.
        self._w_ewma: float = 0.5
        self._w_arima: float = 0.5

        # Error tracking for adaptive weighting.
        self._ewma_errors:  deque[float] = deque(maxlen=max_error_history)
        self._arima_errors: deque[float] = deque(maxlen=max_error_history)
        self._last_ewma_pred:  Optional[float] = None
        self._last_arima_pred: Optional[float] = None

        self._n_updates: int = 0
        self._fitted: bool = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, series: List[float]) -> "HybridPredictor":
        """
        Fit both sub-predictors to *series*.

        If ewma_alpha was not specified at construction, finds the optimal
        alpha by grid search on the first 80% of the series.

        Parameters
        ----------
        series : List[float]
            Historical utilisation values in chronological order.

        Returns
        -------
        self
        """
        if not series:
            return self

        # Optionally find best alpha.
        if self._ewma_alpha_init is None and len(series) >= 10:
            best_alpha = EWMAPredictor.optimal_alpha(series)
            self._ewma = EWMAPredictor(alpha=best_alpha)

        self._ewma.fit(series)
        self._arima.fit(series)
        self._fitted = True

        # Reset error tracking.
        self._ewma_errors.clear()
        self._arima_errors.clear()
        self._n_updates = 0
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, horizon: int = 1) -> Tuple[float, float, float]:
        """
        Return (point, lo_90, hi_90) for h steps ahead.

        For horizon <= short_horizon: pure EWMA forecast.
        For horizon > short_horizon: adaptive blend of EWMA and ARIMA.

        Returns (0.0, 0.0, 0.0) before any data is seen.
        """
        if not self._fitted and self._ewma.n_updates == 0:
            return (0.0, 0.0, 0.0)

        ewma_pt, ewma_lo, ewma_hi = self._ewma.predict(horizon)

        if horizon <= self.short_horizon:
            return (ewma_pt, ewma_lo, ewma_hi)

        # Blend EWMA and ARIMA.
        arima_pt, arima_lo, arima_hi = self._arima.predict(horizon)

        point = self._w_ewma * ewma_pt + self._w_arima * arima_pt
        lo    = self._w_ewma * ewma_lo + self._w_arima * arima_lo
        hi    = self._w_ewma * ewma_hi + self._w_arima * arima_hi

        # Clamp to non-negative.
        return (point, max(0.0, lo), hi)

    # ── Online update ─────────────────────────────────────────────────────────

    def update(self, value: float) -> None:
        """
        Incorporate a new observation and update blending weights.

        Records one-step prediction errors for both sub-models before
        updating them, so errors reflect genuine out-of-sample performance.
        """
        # Record errors before updating.
        if self._last_ewma_pred is not None:
            self._ewma_errors.append(abs(value - self._last_ewma_pred))
        if self._last_arima_pred is not None:
            self._arima_errors.append(abs(value - self._last_arima_pred))

        # Update sub-predictors.
        self._ewma.update(value)
        self._arima.update(value)
        self._n_updates += 1

        # Store one-step predictions for error tracking next cycle.
        ewma_pt, _, _  = self._ewma.predict(1)
        arima_pt, _, _ = self._arima.predict(1)
        self._last_ewma_pred  = ewma_pt
        self._last_arima_pred = arima_pt

        # Periodically recompute blending weights.
        if self._n_updates % self.weight_update_interval == 0:
            self._update_weights()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def ewma_weight(self) -> float:
        return self._w_ewma

    @property
    def arima_weight(self) -> float:
        return self._w_arima

    @property
    def ewma_alpha(self) -> float:
        return self._ewma.alpha

    @property
    def using_statsmodels(self) -> bool:
        return self._arima.using_statsmodels

    def recent_errors(self) -> dict:
        """Return a dict of recent MAE for each sub-model."""
        def mae(errors):
            e = list(errors)
            return sum(e) / len(e) if e else 0.0
        return {
            "ewma_mae":  mae(self._ewma_errors),
            "arima_mae": mae(self._arima_errors),
            "w_ewma":    self._w_ewma,
            "w_arima":   self._w_arima,
        }

    def reset(self) -> None:
        """Reset all state."""
        self._ewma.reset()
        self._arima = ARIMAPredictor(
            p=self._arima.p, d=self._arima.d, q=self._arima.q
        )
        self._w_ewma = 0.5
        self._w_arima = 0.5
        self._ewma_errors.clear()
        self._arima_errors.clear()
        self._last_ewma_pred = None
        self._last_arima_pred = None
        self._n_updates = 0
        self._fitted = False

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_weights(self) -> None:
        """
        Recompute blending weights using inverse-error weighting.

        Model with lower recent MAE gets a higher weight.
        """
        if (
            len(self._ewma_errors) < _MIN_WEIGHT_HISTORY
            or len(self._arima_errors) < _MIN_WEIGHT_HISTORY
        ):
            return  # not enough data yet

        mae_ewma  = sum(self._ewma_errors)  / len(self._ewma_errors)
        mae_arima = sum(self._arima_errors) / len(self._arima_errors)
        total = mae_ewma + mae_arima

        if total < 1e-12:
            # Both models are perfect; keep equal weights.
            self._w_ewma = 0.5
            self._w_arima = 0.5
        else:
            # Inverse-error: better model (lower error) gets higher weight.
            self._w_ewma  = mae_arima / total
            self._w_arima = mae_ewma  / total

    def __repr__(self) -> str:
        return (
            f"HybridPredictor("
            f"alpha={self._ewma.alpha}, "
            f"arima=({self._arima.p},{self._arima.d},{self._arima.q}), "
            f"w=[{self._w_ewma:.2f},{self._w_arima:.2f}], "
            f"n={self._n_updates})"
        )
