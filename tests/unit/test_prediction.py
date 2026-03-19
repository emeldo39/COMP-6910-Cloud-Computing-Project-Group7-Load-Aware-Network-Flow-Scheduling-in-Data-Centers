"""
LAFS -- Prediction Module Unit Tests
=====================================
COMP 6910 -- Group 7

Tests for:
  LinkLoadSampler / LinkLoadSeries    (TestLinkLoadSampler, TestLinkLoadSeries)
  EWMAPredictor                       (TestEWMAPredictor)
  ARPredictor                         (TestARPredictor)
  ARIMAPredictor                      (TestARIMAPredictor)
  HybridPredictor                     (TestHybridPredictor)
  LoadForecaster / NetworkLoadForecast (TestLoadForecaster)

Synthetic signals used throughout
----------------------------------
  CONSTANT  : [0.5] * 60  -- predictor should converge to 0.5
  STEP      : [0.2]*30 + [0.8]*30  -- tests adaptation after step change
  SINE      : 0.5 + 0.3*sin(2*pi*i/20)  -- periodic, tests ARIMA advantage
  RAMP      : i/100  for i in range(60)   -- linear trend
  ALLREDUCE : alternating low (0.1) and burst (0.9) every 5 samples
"""

from __future__ import annotations

import math
import unittest
from typing import List

from src.topology.fattree import FatTreeGraph
from src.workload.flow import Flow
from src.metrics.link_load import LinkLoadSample, LinkLoadSeries, LinkLoadSampler
from src.prediction.ewma import EWMAPredictor
from src.prediction.arima import ARPredictor, ARIMAPredictor
from src.prediction.hybrid import HybridPredictor
from src.prediction.forecaster import (
    LoadForecaster,
    LinkLoadForecast,
    NetworkLoadForecast,
)


# =============================================================================
# Synthetic signal generators
# =============================================================================

def _constant(n=60, val=0.5):
    return [val] * n

def _step(lo=0.2, hi=0.8, n=60):
    half = n // 2
    return [lo] * half + [hi] * (n - half)

def _sine(n=60, amplitude=0.3, offset=0.5, period=20):
    return [offset + amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]

def _ramp(n=60):
    return [i / (n - 1) for i in range(n)]

def _allreduce_bursts(n=60, gap=5, lo=0.1, hi=0.9):
    return [hi if (i % gap == 0) else lo for i in range(n)]


# Helper: build a minimal scheduled flow with a known path
def _make_flow(flow_id, src_ip, dst_ip, size_bytes, arrival_time, path):
    f = Flow(
        flow_id=flow_id,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=12345,
        dst_port=80,
        protocol=6,
        size_bytes=size_bytes,
        arrival_time=arrival_time,
    )
    f.assigned_path = path
    return f


# =============================================================================
# TestLinkLoadSeries
# =============================================================================

class TestLinkLoadSeries(unittest.TestCase):

    def _make_series(self, utils):
        s = LinkLoadSeries(link=("a", "b"), capacity_bps=1e9)
        for i, u in enumerate(utils):
            sample = LinkLoadSample(
                link=("a", "b"),
                t_start=i * 0.1,
                t_end=(i + 1) * 0.1,
                bytes_observed=int(u * 1e9 * 0.1 / 8),
                utilisation=u,
            )
            s.append(sample)
        return s

    def test_values_returns_chronological_list(self):
        s = self._make_series([0.1, 0.2, 0.3])
        self.assertEqual(s.values(), [0.1, 0.2, 0.3])

    def test_len(self):
        s = self._make_series([0.1, 0.2, 0.3])
        self.assertEqual(len(s), 3)

    def test_mean(self):
        s = self._make_series([0.2, 0.4, 0.6])
        self.assertAlmostEqual(s.mean, 0.4, places=9)

    def test_variance_constant_series(self):
        s = self._make_series([0.5] * 10)
        self.assertAlmostEqual(s.variance, 0.0, places=9)

    def test_last_n(self):
        s = self._make_series([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(s.last_n(3), [0.3, 0.4, 0.5])

    def test_last_n_exceeds_length_returns_all(self):
        s = self._make_series([0.1, 0.2])
        self.assertEqual(s.last_n(10), [0.1, 0.2])

    def test_latest_sample(self):
        s = self._make_series([0.1, 0.9])
        self.assertAlmostEqual(s.latest.utilisation, 0.9)

    def test_empty_series_mean_is_zero(self):
        s = LinkLoadSeries(link=("x", "y"))
        self.assertEqual(s.mean, 0.0)

    def test_max_samples_circular_buffer(self):
        s = LinkLoadSeries(link=("a", "b"), max_samples=5)
        for i in range(10):
            sample = LinkLoadSample(("a","b"), i*0.1, (i+1)*0.1, 0, float(i))
            s.append(sample)
        self.assertEqual(len(s), 5)
        self.assertEqual(s.values(), [5.0, 6.0, 7.0, 8.0, 9.0])


# =============================================================================
# TestLinkLoadSampler
# =============================================================================

class TestLinkLoadSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.topo = FatTreeGraph(k=4)
        # A simple known path within the topology.
        all_paths = cls.topo.get_paths("h_0_0_0", "h_1_0_0")
        cls.path = all_paths[0]   # e.g. [h_0_0_0, e_0_0, a_0_0, c_0_0, a_1_0, e_1_0, h_1_0_0]
        cls.src_ip = cls.topo.get_host_ip("h_0_0_0")
        cls.dst_ip = cls.topo.get_host_ip("h_1_0_0")

    def test_empty_ingest_returns_zero(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        n = sampler.ingest([])
        self.assertEqual(n, 0)

    def test_flow_without_path_skipped(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f = Flow("f1", self.src_ip, self.dst_ip, 1000, 80, 6, 1000,
                 arrival_time=0.0)
        n = sampler.ingest([f])
        self.assertEqual(n, 0)

    def test_single_flow_ingested(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f = _make_flow("f1", self.src_ip, self.dst_ip, 1_000_000, 0.05, self.path)
        n = sampler.ingest([f])
        self.assertEqual(n, 1)

    def test_build_series_populates_links(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f = _make_flow("f1", self.src_ip, self.dst_ip, 1_000_000, 0.05, self.path)
        sampler.ingest([f])
        sampler.build_series()
        # Every hop in the path should have a series.
        for i in range(len(self.path) - 1):
            link = (self.path[i], self.path[i + 1])
            s = sampler.get_series(link)
            self.assertGreater(len(s), 0, f"No data for link {link}")

    def test_utilisation_non_negative(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f = _make_flow("f1", self.src_ip, self.dst_ip, 500_000, 0.0, self.path)
        sampler.ingest([f])
        sampler.build_series()
        for s in sampler.all_series().values():
            for v in s.values():
                self.assertGreaterEqual(v, 0.0)

    def test_multiple_flows_accumulate_bytes(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        # Two flows in the same window [0.0, 0.1).
        f1 = _make_flow("f1", self.src_ip, self.dst_ip, 1_000_000, 0.01, self.path)
        f2 = _make_flow("f2", self.src_ip, self.dst_ip, 1_000_000, 0.05, self.path)
        sampler.ingest([f1, f2])
        sampler.build_series()
        first_link = (self.path[0], self.path[1])
        s = sampler.get_series(first_link)
        # Combined bytes in window 0: 2 MB
        total_bytes = s._samples[0].bytes_observed
        self.assertEqual(total_bytes, 2_000_000)

    def test_flows_in_different_windows_produce_two_samples(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f1 = _make_flow("f1", self.src_ip, self.dst_ip, 100_000, 0.05, self.path)
        f2 = _make_flow("f2", self.src_ip, self.dst_ip, 100_000, 0.15, self.path)
        sampler.ingest([f1, f2])
        sampler.build_series()
        first_link = (self.path[0], self.path[1])
        s = sampler.get_series(first_link)
        self.assertEqual(len(s), 2)

    def test_active_links_non_empty_after_ingest(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f = _make_flow("f1", self.src_ip, self.dst_ip, 100_000, 0.0, self.path)
        sampler.ingest([f])
        sampler.build_series()
        self.assertGreater(len(sampler.active_links()), 0)

    def test_unknown_link_returns_empty_series(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        s = sampler.get_series(("nonexistent_a", "nonexistent_b"))
        self.assertEqual(len(s), 0)

    def test_summary_string(self):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        f = _make_flow("f1", self.src_ip, self.dst_ip, 100_000, 0.0, self.path)
        sampler.ingest([f])
        sampler.build_series()
        s = sampler.summary()
        self.assertIn("flows=1", s)


# =============================================================================
# TestEWMAPredictor
# =============================================================================

class TestEWMAPredictor(unittest.TestCase):

    def test_constant_series_converges(self):
        pred = EWMAPredictor(alpha=0.3)
        pred.fit(_constant(60, 0.5))
        pt, _, _ = pred.predict(1)
        self.assertAlmostEqual(pt, 0.5, delta=0.01)

    def test_predict_before_fit_returns_zero(self):
        pred = EWMAPredictor()
        pt, lo, hi = pred.predict(1)
        self.assertEqual(pt, 0.0)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 0.0)

    def test_alpha_1_equals_last_observation(self):
        pred = EWMAPredictor(alpha=1.0)
        pred.update(0.3)
        pred.update(0.7)
        pt, _, _ = pred.predict(1)
        self.assertAlmostEqual(pt, 0.7, places=9)

    def test_confidence_interval_contains_point(self):
        pred = EWMAPredictor(alpha=0.3)
        pred.fit(_constant(60, 0.5))
        pt, lo, hi = pred.predict(1)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_ci_widens_with_horizon(self):
        pred = EWMAPredictor(alpha=0.3)
        pred.fit(_sine(60))
        _, lo1, hi1 = pred.predict(1)
        _, lo5, hi5 = pred.predict(5)
        # Width at h=5 should be at least as wide as h=1.
        self.assertGreaterEqual(hi5 - lo5, hi1 - lo1 - 1e-9)

    def test_lo_always_non_negative(self):
        pred = EWMAPredictor(alpha=0.5)
        pred.fit(_allreduce_bursts(60))
        _, lo, _ = pred.predict(1)
        self.assertGreaterEqual(lo, 0.0)

    def test_rmse_zero_for_constant(self):
        pred = EWMAPredictor(alpha=1.0)
        pred.fit(_constant(50, 0.4))
        self.assertAlmostEqual(pred.rmse, 0.0, places=9)

    def test_n_updates_counts_correctly(self):
        pred = EWMAPredictor()
        for _ in range(10):
            pred.update(0.5)
        self.assertEqual(pred.n_updates, 10)

    def test_reset_clears_state(self):
        pred = EWMAPredictor(alpha=0.3)
        pred.fit(_constant(20, 0.7))
        pred.reset()
        self.assertIsNone(pred.current_estimate)
        self.assertEqual(pred.n_updates, 0)

    def test_optimal_alpha_returns_valid_value(self):
        alpha = EWMAPredictor.optimal_alpha(_sine(60))
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_optimal_alpha_short_series_returns_default(self):
        from src.prediction.ewma import DEFAULT_ALPHA
        alpha = EWMAPredictor.optimal_alpha([0.5, 0.5])
        self.assertEqual(alpha, DEFAULT_ALPHA)

    def test_step_change_high_alpha_adapts_faster(self):
        series = _step(0.1, 0.9, 60)
        pred_fast = EWMAPredictor(alpha=0.8)
        pred_slow = EWMAPredictor(alpha=0.1)
        pred_fast.fit(series)
        pred_slow.fit(series)
        # After step (second half), fast should be closer to 0.9.
        fast_pt, _, _ = pred_fast.predict(1)
        slow_pt, _, _ = pred_slow.predict(1)
        self.assertGreater(fast_pt, slow_pt)


# =============================================================================
# TestARPredictor
# =============================================================================

class TestARPredictor(unittest.TestCase):

    def test_constant_series_forecast_is_constant(self):
        pred = ARPredictor(p=2, d=0)
        pred.fit(_constant(40, 0.5))
        pt, _, _ = pred.predict(1)
        self.assertAlmostEqual(pt, 0.5, delta=0.05)

    def test_predict_before_fit_returns_mean(self):
        pred = ARPredictor(p=2, d=0)
        series = _constant(3, 0.4)
        pred.fit(series)   # too short -- falls back to mean
        pt, _, _ = pred.predict(1)
        # Should be close to series mean.
        self.assertAlmostEqual(pt, 0.4, delta=0.3)

    def test_predict_tuple_length(self):
        pred = ARPredictor(p=2, d=1)
        pred.fit(_ramp(30))
        result = pred.predict(3)
        self.assertEqual(len(result), 3)

    def test_lo_non_negative(self):
        pred = ARPredictor(p=2, d=1)
        pred.fit(_sine(40))
        _, lo, _ = pred.predict(1)
        self.assertGreaterEqual(lo, 0.0)

    def test_ci_widens_with_horizon(self):
        pred = ARPredictor(p=2, d=1)
        pred.fit(_sine(50))
        _, lo1, hi1 = pred.predict(1)
        _, lo5, hi5 = pred.predict(5)
        width1 = hi1 - lo1
        width5 = hi5 - lo5
        self.assertGreaterEqual(width5, width1 - 1e-9)

    def test_update_advances_history(self):
        pred = ARPredictor(p=2, d=0)
        pred.fit(_constant(20, 0.5))
        pred.update(0.9)
        # After updating with 0.9, the prediction should shift toward 0.9.
        pt, _, _ = pred.predict(1)
        self.assertGreater(pt, 0.5)

    def test_differencing_d1(self):
        # A ramp with d=1 should give zero-difference series.
        pred = ARPredictor(p=2, d=1)
        pred.fit(_ramp(40))
        pt, _, _ = pred.predict(1)
        # After a ramp, next value should be close to 1.0.
        self.assertGreater(pt, 0.5)

    def test_ar_p_invalid_raises(self):
        with self.assertRaises(ValueError):
            ARPredictor(p=0)

    def test_ar_d_invalid_raises(self):
        with self.assertRaises(ValueError):
            ARPredictor(d=2)


# =============================================================================
# TestARIMAPredictor
# =============================================================================

class TestARIMAPredictor(unittest.TestCase):

    def _long_sine(self):
        return _sine(n=60)

    def test_predict_tuple_length(self):
        pred = ARIMAPredictor(p=2, d=1, q=1)
        pred.fit(self._long_sine())
        self.assertEqual(len(pred.predict(3)), 3)

    def test_lo_non_negative(self):
        pred = ARIMAPredictor(p=2, d=1, q=1)
        pred.fit(self._long_sine())
        _, lo, _ = pred.predict(1)
        self.assertGreaterEqual(lo, 0.0)

    def test_hi_greater_than_lo(self):
        pred = ARIMAPredictor(p=2, d=1, q=1)
        pred.fit(self._long_sine())
        _, lo, hi = pred.predict(2)
        self.assertGreaterEqual(hi, lo)

    def test_cold_start_returns_fallback(self):
        pred = ARIMAPredictor(p=2, d=1, q=1)
        # No fit: should not raise, just return something.
        pt, lo, hi = pred.predict(1)
        self.assertIsInstance(pt, float)

    def test_update_increments_observations(self):
        pred = ARIMAPredictor()
        pred.fit(_constant(25, 0.5))
        pred.update(0.6)
        self.assertEqual(pred.n_observations, 26)

    def test_constant_series_forecast_near_constant(self):
        pred = ARIMAPredictor(p=1, d=0, q=0)
        pred.fit(_constant(30, 0.5))
        pt, _, _ = pred.predict(1)
        self.assertAlmostEqual(pt, 0.5, delta=0.15)

    def test_n_observations_after_fit(self):
        series = self._long_sine()
        pred = ARIMAPredictor()
        pred.fit(series)
        self.assertEqual(pred.n_observations, len(series))

    def test_repr_contains_backend(self):
        pred = ARIMAPredictor()
        pred.fit(self._long_sine())
        r = repr(pred)
        self.assertIn("backend=", r)


# =============================================================================
# TestHybridPredictor
# =============================================================================

class TestHybridPredictor(unittest.TestCase):

    def test_predict_before_fit_returns_zero(self):
        pred = HybridPredictor()
        pt, lo, hi = pred.predict(1)
        self.assertEqual(pt, 0.0)

    def test_short_horizon_uses_ewma(self):
        pred = HybridPredictor(short_horizon=1)
        pred.fit(_constant(40, 0.5))
        pt, _, _ = pred.predict(1)
        # Should match EWMA closely.
        self.assertAlmostEqual(pt, 0.5, delta=0.05)

    def test_long_horizon_blends_both(self):
        pred = HybridPredictor(short_horizon=1)
        pred.fit(_sine(60))
        pt, lo, hi = pred.predict(5)
        self.assertIsInstance(pt, float)
        self.assertGreaterEqual(hi, lo)

    def test_tuple_length_3(self):
        pred = HybridPredictor()
        pred.fit(_constant(30, 0.3))
        self.assertEqual(len(pred.predict(1)), 3)
        self.assertEqual(len(pred.predict(5)), 3)

    def test_lo_non_negative(self):
        pred = HybridPredictor()
        pred.fit(_allreduce_bursts(50))
        _, lo, _ = pred.predict(3)
        self.assertGreaterEqual(lo, 0.0)

    def test_weights_sum_to_one(self):
        pred = HybridPredictor()
        pred.fit(_sine(40))
        self.assertAlmostEqual(pred.ewma_weight + pred.arima_weight, 1.0, places=9)

    def test_weights_update_after_many_updates(self):
        pred = HybridPredictor(weight_update_interval=5)
        pred.fit(_sine(40))
        initial_w = pred.ewma_weight
        for v in _sine(n=30):
            pred.update(v)
        # Weights may have changed.
        self.assertAlmostEqual(pred.ewma_weight + pred.arima_weight, 1.0, places=9)

    def test_reset_clears_weights(self):
        pred = HybridPredictor()
        pred.fit(_sine(40))
        for v in _sine(30):
            pred.update(v)
        pred.reset()
        self.assertEqual(pred.ewma_weight, 0.5)
        self.assertEqual(pred.arima_weight, 0.5)

    def test_mape_under_70_percent_on_sine(self):
        """LAFS proposal target: prediction error < 70%."""
        series = _sine(n=80)
        train, test = series[:60], series[60:]
        pred = HybridPredictor()
        pred.fit(train)
        errors = []
        for actual in test:
            pt, _, _ = pred.predict(1)
            if actual > 1e-6:
                errors.append(abs(actual - pt) / actual)
            pred.update(actual)
        if errors:
            mape = sum(errors) / len(errors) * 100
            self.assertLess(mape, 70.0,
                f"MAPE {mape:.1f}% exceeds 70% target on sine wave")

    def test_mape_under_70_percent_on_constant(self):
        series = _constant(80, 0.4)
        train, test = series[:60], series[60:]
        pred = HybridPredictor()
        pred.fit(train)
        errors = []
        for actual in test:
            pt, _, _ = pred.predict(1)
            if actual > 1e-6:
                errors.append(abs(actual - pt) / actual)
            pred.update(actual)
        if errors:
            mape = sum(errors) / len(errors) * 100
            self.assertLess(mape, 70.0)


# =============================================================================
# TestLoadForecaster
# =============================================================================

class TestLoadForecaster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.topo = FatTreeGraph(k=4)
        all_paths = cls.topo.get_paths("h_0_0_0", "h_1_0_0")
        cls.path = all_paths[0]
        cls.src_ip = cls.topo.get_host_ip("h_0_0_0")
        cls.dst_ip = cls.topo.get_host_ip("h_1_0_0")

    def _build_sampler(self, n_flows=30, size=500_000):
        sampler = LinkLoadSampler(self.topo, window_s=0.1)
        flows = [
            _make_flow(f"f{i}", self.src_ip, self.dst_ip,
                       size, i * 0.1, self.path)
            for i in range(n_flows)
        ]
        sampler.ingest(flows)
        sampler.build_series()
        return sampler

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            LoadForecaster(self.topo, method="lstm")

    def test_fit_populates_predictors(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        self.assertGreater(fc.n_fitted_links, 0)

    def test_predict_returns_network_forecast(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        nf = fc.predict(t_now=0.0)
        self.assertIsInstance(nf, NetworkLoadForecast)

    def test_forecast_covers_all_fitted_links(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        nf = fc.predict()
        self.assertEqual(len(nf.forecasts), fc.n_fitted_links)

    def test_utilisation_non_negative(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        nf = fc.predict()
        for f in nf.forecasts.values():
            self.assertGreaterEqual(f.predicted_utilisation, 0.0)

    def test_path_max_utilisation_non_negative(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        nf = fc.predict()
        val = nf.path_max_utilisation(self.path)
        self.assertGreaterEqual(val, 0.0)

    def test_path_max_utilisation_empty_path(self):
        nf = NetworkLoadForecast(t_predict=0.0, horizon_s=0.1)
        self.assertEqual(nf.path_max_utilisation([]), 0.0)
        self.assertEqual(nf.path_max_utilisation(["h_0_0_0"]), 0.0)

    def test_congested_links_respects_threshold(self):
        # Build a forecast with known utilisation values.
        nf = NetworkLoadForecast(t_predict=0.0, horizon_s=0.1)
        nf.forecasts[("a","b")] = LinkLoadForecast(
            ("a","b"), 0.0, 0.1, 0.8, 0.7, 0.9, "ewma"
        )
        nf.forecasts[("c","d")] = LinkLoadForecast(
            ("c","d"), 0.0, 0.1, 0.3, 0.2, 0.4, "ewma"
        )
        congested = nf.congested_links(threshold=0.7)
        self.assertIn(("a","b"), congested)
        self.assertNotIn(("c","d"), congested)

    def test_least_congested_path_selects_best(self):
        nf = NetworkLoadForecast(t_predict=0.0, horizon_s=0.1)
        # Two paths: path_a goes through a congested link, path_b does not.
        nf.forecasts[("n1","n2")] = LinkLoadForecast(("n1","n2"), 0, 0.1, 0.9, 0.8, 1.0, "ewma")
        nf.forecasts[("n3","n4")] = LinkLoadForecast(("n3","n4"), 0, 0.1, 0.1, 0.05, 0.2, "ewma")
        path_a = ["src", "n1", "n2", "dst"]
        path_b = ["src", "n3", "n4", "dst"]
        # path_a has max util 0.9, path_b has max util 0.1.
        best = nf.least_congested_path([path_a, path_b])
        self.assertEqual(best, path_b)

    def test_evaluate_mape_returns_dict(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        nf = fc.predict()
        # Build fake actuals matching forecasted links.
        actuals = {link: [f.predicted_utilisation] for link, f in nf.forecasts.items()}
        preds   = {link: [f.predicted_utilisation] for link, f in nf.forecasts.items()}
        result = fc.evaluate(actuals, preds)
        self.assertIn("mape", result)
        # Perfect predictions => MAPE = 0.
        self.assertAlmostEqual(result["mape"], 0.0, delta=1e-6)

    def test_hybrid_forecaster_does_not_crash(self):
        sampler = self._build_sampler(n_flows=35)
        fc = LoadForecaster(self.topo, method="hybrid", horizon_s=0.3)
        fc.fit(sampler)
        nf = fc.predict()
        self.assertGreater(len(nf.forecasts), 0)

    def test_arima_forecaster_does_not_crash(self):
        sampler = self._build_sampler(n_flows=35)
        fc = LoadForecaster(self.topo, method="arima", horizon_s=0.2)
        fc.fit(sampler)
        nf = fc.predict()
        self.assertIsInstance(nf, NetworkLoadForecast)

    def test_forecast_summary_string(self):
        sampler = self._build_sampler()
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        nf = fc.predict()
        s = nf.summary()
        self.assertIn("NetworkLoadForecast", s)

    def test_update_online_returns_forecast(self):
        sampler = self._build_sampler(n_flows=30)
        fc = LoadForecaster(self.topo, method="ewma")
        fc.fit(sampler)
        new_flows = [
            _make_flow("new1", self.src_ip, self.dst_ip,
                       200_000, 3.05, self.path)
        ]
        nf = fc.update(new_flows, t_now=3.0)
        self.assertIsInstance(nf, NetworkLoadForecast)


if __name__ == "__main__":
    unittest.main(verbosity=2)
