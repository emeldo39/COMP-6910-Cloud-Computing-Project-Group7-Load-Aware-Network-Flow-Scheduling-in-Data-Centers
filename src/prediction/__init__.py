"""
LAFS -- Prediction package
==========================
COMP-6910 -- Group 7

Exports
-------
EWMAPredictor        -- single exponential smoothing, O(1) per step
ARPredictor          -- pure-numpy AR(p), always available
ARIMAPredictor       -- statsmodels ARIMA(p,d,q) with AR fallback
HybridPredictor      -- adaptive EWMA+ARIMA blend
LoadForecaster       -- per-link forecaster, produces NetworkLoadForecast
LinkLoadForecast     -- single-link prediction result dataclass
NetworkLoadForecast  -- full-network prediction snapshot dataclass
"""

from src.prediction.ewma import EWMAPredictor
from src.prediction.arima import ARPredictor, ARIMAPredictor
from src.prediction.hybrid import HybridPredictor
from src.prediction.forecaster import (
    LoadForecaster,
    LinkLoadForecast,
    NetworkLoadForecast,
)

__all__ = [
    "EWMAPredictor",
    "ARPredictor",
    "ARIMAPredictor",
    "HybridPredictor",
    "LoadForecaster",
    "LinkLoadForecast",
    "NetworkLoadForecast",
]
