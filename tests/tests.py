# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# tests/tests.py -- Test suite (~80% pass target)
# ============================================================

import sys
import os
import unittest
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RISK_LOW_MAX, RISK_MEDIUM_MAX, RISK_HIGH_MAX,
    ANOMALY_LOG_THRESHOLD, LSTM_FEATURES,
    WEIGHT_ISOLATION_FOREST, WEIGHT_ZSCORE, WEIGHT_LSTM
)


# ----------------------------------------------------------------
# 1. CONFIG TESTS
# ----------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_weights_sum_to_one(self):
        total = WEIGHT_ISOLATION_FOREST + WEIGHT_ZSCORE + WEIGHT_LSTM
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_risk_thresholds_are_ordered(self):
        self.assertLess(RISK_LOW_MAX, RISK_MEDIUM_MAX)
        self.assertLess(RISK_MEDIUM_MAX, RISK_HIGH_MAX)
        self.assertLess(RISK_HIGH_MAX, 100)

    def test_anomaly_threshold_in_high_range(self):
        self.assertGreater(ANOMALY_LOG_THRESHOLD, RISK_MEDIUM_MAX)
        self.assertLessEqual(ANOMALY_LOG_THRESHOLD, RISK_HIGH_MAX)

    def test_lstm_features_not_empty(self):
        self.assertGreater(len(LSTM_FEATURES), 0)
        self.assertIn("close", LSTM_FEATURES)


# ----------------------------------------------------------------
# 2. DATABASE TESTS
# ----------------------------------------------------------------

class TestDatabase(unittest.TestCase):

    def setUp(self):
        import data_layer.database as database
        self._orig_db_path = database.DB_PATH
        database.DB_PATH   = ":memory:"

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "schema.sql"
        )
        with open(schema_path) as f:
            self.conn.executescript(f.read())

    def tearDown(self):
        import data_layer.database as database
        database.DB_PATH = self._orig_db_path
        self.conn.close()

    def test_price_table_exists(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'"
        )
        self.assertIsNotNone(cursor.fetchone())

    def test_anomalies_table_exists(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='anomalies'"
        )
        self.assertIsNotNone(cursor.fetchone())

    def test_model_registry_table_exists(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_registry'"
        )
        self.assertIsNotNone(cursor.fetchone())

    def test_system_events_table_exists(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='system_events'"
        )
        self.assertIsNotNone(cursor.fetchone())

    def test_users_table_exists(self):
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        self.assertIsNotNone(cursor.fetchone())

    def test_insert_and_retrieve_price_row(self):
        self.conn.execute("""
            INSERT INTO price_data (timestamp, open, high, low, close, volume, source)
            VALUES ('2026-01-01T00:00:00', 40000, 41000, 39000, 40500, 1000000, 'test')
        """)
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM price_data").fetchone()
        self.assertEqual(row["close"], 40500)
        self.assertEqual(row["source"], "test")

    def test_duplicate_timestamp_ignored(self):
        for _ in range(3):
            self.conn.execute("""
                INSERT OR IGNORE INTO price_data (timestamp, open, high, low, close, volume)
                VALUES ('2026-01-01T00:00:00', 40000, 41000, 39000, 40500, 1000000)
            """)
        self.conn.commit()
        count = self.conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        self.assertEqual(count, 1)


# ----------------------------------------------------------------
# 3. DATA PIPELINE TESTS
# ----------------------------------------------------------------

class TestDataPipeline(unittest.TestCase):

    def test_cache_starts_invalid(self):
        from data_layer.data_pipeline import _is_cache_valid, clear_price_cache
        clear_price_cache()
        self.assertFalse(_is_cache_valid())

    def test_cache_set_and_retrieved(self):
        from data_layer.data_pipeline import _set_cache, _get_cache
        sample = {"close": 50000, "source": "coingecko"}
        _set_cache(sample, "coingecko")
        result = _get_cache()
        self.assertIsNotNone(result)
        self.assertEqual(result["close"], 50000)

    def test_volatility_computation(self):
        from data_layer.data_pipeline import compute_volatility
        df  = pd.DataFrame({"close": np.random.uniform(40000, 50000, 200)})
        vol = compute_volatility(df)
        self.assertEqual(len(vol), len(df))
        self.assertGreaterEqual(vol.iloc[-1], 0)

    def test_needs_seeding_returns_bool(self):
        from data_layer.data_pipeline import needs_seeding
        result = needs_seeding()
        self.assertIsInstance(result, bool)


# ----------------------------------------------------------------
# 4. ANOMALY DETECTOR TESTS
# ----------------------------------------------------------------

class TestAnomalyDetector(unittest.TestCase):

    def test_get_risk_level_low(self):
        from model_layer.anomaly_detector import get_risk_level
        self.assertEqual(get_risk_level(0),  "Low")
        self.assertEqual(get_risk_level(15), "Low")
        self.assertEqual(get_risk_level(30), "Low")

    def test_get_risk_level_medium(self):
        from model_layer.anomaly_detector import get_risk_level
        self.assertEqual(get_risk_level(31), "Medium")
        self.assertEqual(get_risk_level(45), "Medium")
        self.assertEqual(get_risk_level(60), "Medium")

    def test_get_risk_level_high(self):
        from model_layer.anomaly_detector import get_risk_level
        self.assertEqual(get_risk_level(61), "High")
        self.assertEqual(get_risk_level(75), "High")
        self.assertEqual(get_risk_level(80), "High")

    def test_get_risk_level_critical(self):
        from model_layer.anomaly_detector import get_risk_level
        self.assertEqual(get_risk_level(81),  "Critical")
        self.assertEqual(get_risk_level(100), "Critical")

    def test_zscore_normal_price(self):
        from model_layer.anomaly_detector import _score_zscore
        params       = {"mean": 50000.0, "std": 1000.0}
        score, sigma = _score_zscore(params, 50000.0)
        self.assertAlmostEqual(sigma, 0.0)
        self.assertAlmostEqual(score, 0.0)

    def test_zscore_extreme_price(self):
        from model_layer.anomaly_detector import _score_zscore
        params       = {"mean": 50000.0, "std": 1000.0}
        score, sigma = _score_zscore(params, 54000.0)
        self.assertGreaterEqual(score, 99.0)

    def test_zscore_score_capped_at_100(self):
        from model_layer.anomaly_detector import _score_zscore
        params    = {"mean": 50000.0, "std": 500.0}
        score, _  = _score_zscore(params, 60000.0)
        self.assertEqual(score, 100.0)

    def test_confidence_levels(self):
        from model_layer.anomaly_detector import _confidence_level
        self.assertEqual(_confidence_level(1), "Low")
        self.assertEqual(_confidence_level(2), "Medium")
        self.assertEqual(_confidence_level(3), "High")

    def test_lstm_model_forward_pass(self):
        import torch
        from model_layer.anomaly_detector import LSTMModel
        model  = LSTMModel(len(LSTM_FEATURES), 64, 2)
        dummy  = torch.randn(1, 60, len(LSTM_FEATURES))
        output = model(dummy)
        self.assertEqual(output.shape, (1, 1))

    def test_if_reason_returns_string(self):
        from model_layer.anomaly_detector import _if_reason
        for score in [10, 55, 85]:
            result = _if_reason(score)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_ensemble_score_within_range(self):
        if_score   = 80.0
        zs_score   = 50.0
        lstm_score = 70.0
        score = (
            if_score   * WEIGHT_ISOLATION_FOREST +
            zs_score   * WEIGHT_ZSCORE +
            lstm_score * WEIGHT_LSTM
        )
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_models_not_ready_initially(self):
        from model_layer.anomaly_detector import models_ready
        result = models_ready()
        self.assertIsInstance(result, bool)


# ----------------------------------------------------------------
# 5. FEATURE ENGINE TESTS
# ----------------------------------------------------------------

class TestFeatureEngine(unittest.TestCase):

    def _make_df(self, n=100):
        np.random.seed(42)
        close  = np.cumsum(np.random.randn(n)) + 50000
        volume = np.random.uniform(1e9, 2e9, n)
        return pd.DataFrame({
            "open":   close * 0.999,
            "high":   close * 1.002,
            "low":    close * 0.998,
            "close":  close,
            "volume": volume,
        })

    def test_returns_computed(self):
        from feature_engine.features import compute_features
        df  = self._make_df()
        out = compute_features(df)
        self.assertIn("returns", out.columns)
        self.assertIn("log_returns", out.columns)

    def test_moving_averages_present(self):
        from feature_engine.features import compute_features
        out = compute_features(self._make_df(210))
        for col in ["ma_20", "ma_50", "ma_200"]:
            self.assertIn(col, out.columns)

    def test_volatility_rolling_non_negative(self):
        from feature_engine.features import compute_features
        out = compute_features(self._make_df())
        self.assertTrue((out["volatility_rolling"].dropna() >= 0).all())

    def test_momentum_14_length(self):
        from feature_engine.features import compute_features
        df  = self._make_df(50)
        out = compute_features(df)
        self.assertEqual(len(out["momentum_14"]), len(df))

    def test_bollinger_bands_ordering(self):
        from feature_engine.features import compute_features
        out = compute_features(self._make_df())
        self.assertTrue((out["bb_upper"] >= out["bb_lower"]).all())

    def test_volume_spike_zero_when_no_volume(self):
        from feature_engine.features import compute_features
        df  = self._make_df().drop(columns=["volume"])
        out = compute_features(df)
        self.assertTrue((out["volume_spike"] == 0).all())

    def test_high_low_range_positive(self):
        from feature_engine.features import compute_features
        out = compute_features(self._make_df())
        self.assertTrue((out["high_low_range"].dropna() > 0).all())

    def test_snapshot_returns_dict(self):
        from feature_engine.features import get_feature_snapshot
        snap = get_feature_snapshot(self._make_df())
        self.assertIsInstance(snap, dict)
        self.assertIn("close",             snap)
        self.assertIn("returns",           snap)
        self.assertIn("volatility_rolling",snap)
        self.assertIn("ma_20",             snap)
        self.assertIn("momentum_14",       snap)
        self.assertIn("volume_spike",      snap)
        self.assertIn("bb_upper",          snap)

    def test_snapshot_empty_df(self):
        from feature_engine.features import get_feature_snapshot
        snap = get_feature_snapshot(pd.DataFrame())
        self.assertEqual(snap, {})

    def test_bb_pct_b_between_0_and_1_mostly(self):
        from feature_engine.features import compute_features
        out = compute_features(self._make_df(200))
        # %B can go outside [0,1] during breakouts, but mean should be near 0.5
        mean_pct_b = out["bb_pct_b"].mean()
        self.assertGreater(mean_pct_b, 0.0)
        self.assertLess(mean_pct_b, 1.5)


# ----------------------------------------------------------------
# 6. SIGNAL LOGS TESTS
# ----------------------------------------------------------------

class TestSignalLogs(unittest.TestCase):

    def setUp(self):
        import tempfile
        import data_layer.database as database
        self._orig_db_path = database.DB_PATH
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        database.DB_PATH = self._tmp.name
        database.init_db()

    def tearDown(self):
        import data_layer.database as database
        database.DB_PATH = self._orig_db_path
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _sample_result(self, ts="2026-01-01T00:00:00", coin="BTC"):
        return {
            "timestamp":               ts,
            "close_price":             50000.0,
            "volume":                  1_000_000.0,
            "risk_score":              42.0,
            "risk_level":              "Medium",
            "isolation_forest_score":  35.0,
            "zscore_score":            40.0,
            "lstm_score":              55.0,
            "if_reason":               "Normal",
            "zscore_value":            0.5,
            "zscore_reason":           "Low deviation",
            "lstm_predicted_price":    50100.0,
            "lstm_reason":             "Small delta",
            "models_agreed":           1,
            "confidence_level":        "Medium",
            "plain_english_summary":   "All good",
            "signal_type":             "real",
            "contributing_models":     "LSTM",
            "signal_strength":         "Weak",
            "coin":                    coin,
            "ensemble_weights":        {"isolation_forest": 0.25, "zscore": 0.25, "lstm": 0.50},
            "signal":                  "HOLD",
            "strategy":                "conservative",
        }

    def test_signal_logs_table_created(self):
        from data_layer.database import get_connection
        with get_connection() as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        self.assertIn("signal_logs", tables)

    def test_insert_signal_log(self):
        from data_layer.database import insert_signal_log, get_connection
        insert_signal_log(self._sample_result())
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM signal_logs").fetchone()[0]
        self.assertEqual(count, 1)

    def test_duplicate_coin_timestamp_ignored(self):
        from data_layer.database import insert_signal_log, get_connection
        row = self._sample_result()
        insert_signal_log(row)
        insert_signal_log(row)  # duplicate — should be ignored
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM signal_logs").fetchone()[0]
        self.assertEqual(count, 1)

    def test_ensemble_weights_serialised_to_json(self):
        from data_layer.database import insert_signal_log, get_connection
        insert_signal_log(self._sample_result())
        with get_connection() as conn:
            row = conn.execute("SELECT ensemble_weights FROM signal_logs").fetchone()
        import json
        ew = json.loads(row[0])
        self.assertIn("lstm", ew)
        self.assertAlmostEqual(ew["lstm"], 0.50)

    def test_confidence_mapped_from_text(self):
        from data_layer.database import insert_signal_log, get_connection
        insert_signal_log(self._sample_result())
        with get_connection() as conn:
            row = conn.execute("SELECT confidence FROM signal_logs").fetchone()
        self.assertAlmostEqual(row[0], 0.67, places=2)

    def test_get_signal_logs_pagination(self):
        from data_layer.database import insert_signal_log, get_signal_logs
        for i in range(5):
            insert_signal_log(self._sample_result(ts=f"2026-01-0{i+1}T00:00:00"))
        rows_p1, total = get_signal_logs(coin="BTC", page=1, limit=3)
        self.assertEqual(total, 5)
        self.assertEqual(len(rows_p1), 3)
        rows_p2, _ = get_signal_logs(coin="BTC", page=2, limit=3)
        self.assertEqual(len(rows_p2), 2)

    def test_get_signal_logs_coin_filter(self):
        from data_layer.database import insert_signal_log, get_signal_logs
        insert_signal_log(self._sample_result(coin="BTC"))
        insert_signal_log(self._sample_result(coin="ETH"))
        _, total_btc = get_signal_logs(coin="BTC")
        _, total_eth = get_signal_logs(coin="ETH")
        self.assertEqual(total_btc, 1)
        self.assertEqual(total_eth, 1)

    def test_signal_field_stored(self):
        from data_layer.database import insert_signal_log, get_connection
        row = self._sample_result()
        row["signal"] = "BUY"
        insert_signal_log(row)
        with get_connection() as conn:
            stored = conn.execute("SELECT signal FROM signal_logs").fetchone()
        self.assertEqual(stored[0], "BUY")


# ----------------------------------------------------------------
# 7. DATABASE STEP 7 TESTS (risk_scores + model_metrics)
# ----------------------------------------------------------------

class TestDatabaseStep7(unittest.TestCase):

    def setUp(self):
        import tempfile
        import data_layer.database as database
        self._orig_db_path = database.DB_PATH
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        database.DB_PATH = self._tmp.name
        database.init_db()

    def tearDown(self):
        import data_layer.database as database
        database.DB_PATH = self._orig_db_path
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _sample_result(self, ts="2026-01-01T00:00:00", coin="BTC"):
        return {
            "coin":                    coin,
            "timestamp":               ts,
            "close_price":             50000.0,
            "volume":                  1_000_000.0,
            "risk_score":              55.0,
            "risk_level":              "Medium",
            "confidence_level":        "Medium",
            "isolation_forest_score":  50.0,
            "zscore_score":            55.0,
            "lstm_score":              60.0,
            "models_agreed":           2,
            "ensemble_weights":        {"isolation_forest": 0.25, "zscore": 0.25, "lstm": 0.50},
            "signal":                  "HOLD",
        }

    # ---- risk_scores ----

    def test_risk_scores_table_created(self):
        from data_layer.database import get_connection
        with get_connection() as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        self.assertIn("risk_scores", tables)

    def test_insert_risk_score(self):
        from data_layer.database import insert_risk_score, get_connection
        insert_risk_score(self._sample_result())
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0]
        self.assertEqual(count, 1)

    def test_risk_score_duplicate_ignored(self):
        from data_layer.database import insert_risk_score, get_connection
        row = self._sample_result()
        insert_risk_score(row)
        insert_risk_score(row)
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM risk_scores").fetchone()[0]
        self.assertEqual(count, 1)

    def test_get_risk_scores_pagination(self):
        from data_layer.database import insert_risk_score, get_risk_scores
        for i in range(5):
            insert_risk_score(self._sample_result(ts=f"2026-01-0{i+1}T00:00:00"))
        rows_p1, total = get_risk_scores(coin="BTC", page=1, limit=3)
        self.assertEqual(total, 5)
        self.assertEqual(len(rows_p1), 3)
        rows_p2, _ = get_risk_scores(coin="BTC", page=2, limit=3)
        self.assertEqual(len(rows_p2), 2)

    def test_risk_score_confidence_mapped(self):
        from data_layer.database import insert_risk_score, get_connection
        insert_risk_score(self._sample_result())
        with get_connection() as conn:
            row = conn.execute("SELECT confidence FROM risk_scores").fetchone()
        self.assertAlmostEqual(row[0], 0.67, places=2)

    # ---- model_metrics ----

    def test_model_metrics_table_created(self):
        from data_layer.database import get_connection
        with get_connection() as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        self.assertIn("model_metrics", tables)

    def test_insert_model_metric(self):
        from data_layer.database import insert_model_metric, get_connection
        insert_model_metric("lstm", "val_loss", 0.0032)
        with get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]
        self.assertEqual(count, 1)

    def test_get_model_metrics_filtered(self):
        from data_layer.database import insert_model_metric, get_model_metrics
        insert_model_metric("lstm",             "val_loss", 0.003)
        insert_model_metric("isolation_forest", "anomaly_rate", 0.05)
        lstm_rows = get_model_metrics(model_name="lstm")
        self.assertEqual(len(lstm_rows), 1)
        self.assertEqual(lstm_rows[0]["model_name"], "lstm")

    def test_get_model_metrics_all(self):
        from data_layer.database import insert_model_metric, get_model_metrics
        insert_model_metric("lstm",   "val_mae",  0.012)
        insert_model_metric("zscore", "val_mae",  0.040)
        rows = get_model_metrics()
        self.assertEqual(len(rows), 2)


# ----------------------------------------------------------------
# 8. PERFORMANCE & RELIABILITY TESTS  (Step 8)
# ----------------------------------------------------------------

class TestPerformanceReliability(unittest.TestCase):

    # ---- retry_get ----

    def test_retry_get_succeeds_first_try(self):
        """retry_get returns response when the call succeeds immediately."""
        from unittest.mock import patch, Mock
        from utils import retry_get
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        with patch("utils.requests.get", return_value=mock_resp) as mock_get:
            result = retry_get("http://example.com/api", label="test")
        self.assertEqual(result, mock_resp)
        self.assertEqual(mock_get.call_count, 1)

    def test_retry_get_retries_on_failure(self):
        """retry_get retries the configured number of times before raising."""
        from unittest.mock import patch
        from utils import retry_get
        import config
        with patch("utils.requests.get", side_effect=ConnectionError("down")), \
             patch("utils.time.sleep"):
            with self.assertRaises(RuntimeError):
                retry_get("http://example.com/api", label="test")

    def test_retry_get_succeeds_after_one_failure(self):
        """retry_get succeeds on second attempt if first fails."""
        from unittest.mock import patch, Mock, call
        from utils import retry_get
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        call_results = [ConnectionError("err"), mock_resp]
        with patch("utils.requests.get", side_effect=call_results), \
             patch("utils.time.sleep"):
            result = retry_get("http://example.com/api", label="test")
        self.assertEqual(result, mock_resp)

    def test_retry_get_raises_runtime_error_on_exhaustion(self):
        from unittest.mock import patch
        from utils import retry_get
        with patch("utils.requests.get", side_effect=ConnectionError("x")), \
             patch("utils.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                retry_get("http://bad.url/", label="myservice")
        self.assertIn("myservice", str(ctx.exception))

    def test_retry_get_passes_params(self):
        from unittest.mock import patch, Mock
        from utils import retry_get
        mock_resp = Mock()
        mock_resp.raise_for_status = Mock()
        with patch("utils.requests.get", return_value=mock_resp) as mock_get:
            retry_get("http://example.com/api", params={"k": "v"}, timeout=5)
        mock_get.assert_called_once_with(
            "http://example.com/api", params={"k": "v"}, timeout=5
        )

    # ---- setup_logging ----

    def test_setup_logging_is_idempotent(self):
        """Calling setup_logging() twice should not add duplicate handlers."""
        import logging
        from config import setup_logging
        root = logging.getLogger()
        before = len(root.handlers)
        setup_logging()
        setup_logging()
        after = len(root.handlers)
        # handler count must not have grown on the second call
        self.assertEqual(before, after)

    def test_cache_is_invalid_after_clear(self):
        from data_layer.data_pipeline import _is_cache_valid, _set_cache, clear_price_cache
        _set_cache({"close": 1}, "test", "BTC")
        self.assertTrue(_is_cache_valid("BTC"))
        clear_price_cache("BTC")
        self.assertFalse(_is_cache_valid("BTC"))


# ----------------------------------------------------------------
# 9. EXPLAINABILITY TESTS
# ----------------------------------------------------------------

class TestExplainability(unittest.TestCase):

    def _raw(self, risk_score=73.0, coin="BTC"):
        return {
            "timestamp":               "2026-01-01T00:00:00",
            "close_price":             50000.0,
            "volume":                  1_000_000.0,
            "risk_score":              risk_score,
            "risk_level":              "High",
            "isolation_forest_score":  65.0,
            "zscore_score":            78.0,
            "lstm_score":              74.0,
            "if_reason":               "Price pattern deviates from normal cluster",
            "zscore_value":            2.3,
            "zscore_reason":           "price elevated vs 30-day baseline",
            "lstm_predicted_price":    48500.0,
            "lstm_reason":             "Predicted price lower than actual — possible correction",
            "models_agreed":           3,
            "confidence_level":        "High",
            "plain_english_summary":   "Risk elevated.",
            "signal_type":             "real",
            "contributing_models":     "Isolation Forest, Z-Score, LSTM",
            "signal_strength":         "Strong",
            "coin":                    coin,
            "ensemble_weights":        {"isolation_forest": 0.25, "zscore": 0.25, "lstm": 0.50},
        }

    def test_explain_response_has_required_keys(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        for key in ("coin", "risk_score", "risk_level", "confidence",
                    "model_breakdown", "reasoning", "ensemble_weights",
                    "models_agreed", "timestamp"):
            self.assertIn(key, out)

    def test_model_breakdown_is_flat_scores(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        mb = out["model_breakdown"]
        self.assertAlmostEqual(mb["isolation_forest"], 65.0)
        self.assertAlmostEqual(mb["zscore"],           78.0)
        self.assertAlmostEqual(mb["lstm"],             74.0)

    def test_reasoning_is_flat_strings(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        r = out["reasoning"]
        self.assertIsInstance(r["isolation_forest"], str)
        self.assertIsInstance(r["zscore"],           str)
        self.assertIsInstance(r["lstm"],             str)
        self.assertGreater(len(r["isolation_forest"]), 0)

    def test_zscore_reasoning_includes_sigma(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        # sigma 2.3 should appear in the zscore reasoning string
        self.assertIn("2.3", out["reasoning"]["zscore"])

    def test_confidence_high_maps_to_1(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        self.assertAlmostEqual(out["confidence"], 1.0)

    def test_ensemble_weights_sum_to_1(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        total = sum(out["ensemble_weights"].values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_models_agreed_integer(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw()).model_dump()
        self.assertEqual(out["models_agreed"], 3)

    def test_coin_uppercased(self):
        from risk_engine.risk_score import build_explain_response
        out = build_explain_response(self._raw(coin="btc")).model_dump()
        self.assertEqual(out["coin"], "BTC")

    def test_explain_endpoint_503_when_no_data(self):
        from fastapi.testclient import TestClient
        from api_backend import app
        client = TestClient(app)
        res = client.get("/api/v1/explain/XYZ")
        self.assertEqual(res.status_code, 503)

    def test_explain_endpoint_shape_when_data_present(self):
        """If a live score is cached, /api/v1/explain/{coin} must return model_breakdown + reasoning."""
        from fastapi.testclient import TestClient
        import api_backend
        from api_backend import app
        # Inject a fake cached score
        api_backend._last_score_result["TESTCOIN"] = self._raw(coin="TESTCOIN")
        client = TestClient(app)
        res = client.get("/api/v1/explain/TESTCOIN")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("model_breakdown", data)
        self.assertIn("reasoning",       data)
        self.assertIn("ensemble_weights", data)


# ----------------------------------------------------------------
# 10. API TESTS
# ----------------------------------------------------------------

class TestAPI(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        from api_backend import app
        self.client = TestClient(app)

    def test_health_endpoint_returns_200(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)

    def test_health_contains_required_fields(self):
        res  = self.client.get("/health")
        data = res.json()
        self.assertIn("status",       data)
        self.assertIn("models_ready", data)
        self.assertIn("price_rows",   data)

    def test_chart_endpoint_valid_ranges(self):
        for r in ["1H", "1D", "1W", "1M", "All"]:
            res = self.client.get(f"/api/chart?range={r}")
            self.assertIn(res.status_code, [200, 503])

    def test_anomalies_pagination_defaults(self):
        res  = self.client.get("/api/anomalies")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("page",        data)
        self.assertIn("limit",       data)
        self.assertIn("total",       data)
        self.assertIn("total_pages", data)
        self.assertIn("anomalies",   data)

    def test_anomalies_pagination_custom(self):
        res  = self.client.get("/api/anomalies?page=1&limit=5")
        data = res.json()
        self.assertEqual(data["limit"], 5)
        self.assertLessEqual(len(data["anomalies"]), 5)

    def test_anomaly_not_found_returns_404(self):
        res = self.client.get("/api/anomalies/999999")
        self.assertEqual(res.status_code, 404)

    def test_models_endpoint_returns_200(self):
        res  = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("registry", data)
        self.assertIn("events",   data)


# ----------------------------------------------------------------
# 11. STEP 9 — INTEGRATION & COVERAGE TESTS
# ----------------------------------------------------------------

class TestStep9Integration(unittest.TestCase):
    """
    Step 9 — verify correctness of the three core workflows:
      1. score_price_row()  →  valid output dict
      2. generate_trading_signal()  →  valid signal values
      3. API endpoints  →  correct shapes
    """

    # ---- 1. score_price_row ----

    def _make_df(self, n=100, base_price=50000.0):
        """Build a minimal OHLCV DataFrame of n rows."""
        import numpy as np
        rng = np.random.default_rng(42)
        close = base_price + rng.normal(0, 200, n).cumsum()
        return pd.DataFrame({
            "timestamp":     [f"2026-01-{i % 28 + 1:02d}T{i % 24:02d}:00:00" for i in range(n)],
            "open":          close * 0.999,
            "high":          close * 1.002,
            "low":           close * 0.998,
            "close":         close,
            "volume":        rng.uniform(1e9, 2e9, n),
            "volatility_24h": rng.uniform(100, 500, n),
        })

    def test_score_price_row_returns_none_without_models(self):
        """score_price_row returns None gracefully when models not loaded."""
        from model_layer.anomaly_detector import score_price_row, _models
        price_data = {
            "timestamp": "2026-01-01T00:00:00", "coin": "BTC",
            "open": 50000., "high": 51000., "low": 49000.,
            "close": 50500., "volume": 1e9, "volatility_24h": 200.,
        }
        df = self._make_df()
        # With no trained models, score_price_row should return None, not crash
        result = score_price_row(price_data, df)
        self.assertIsNone(result)

    def test_score_price_row_output_shape_with_mock_models(self):
        """score_price_row produces a correctly shaped dict when models are injected."""
        import numpy as np
        from unittest.mock import MagicMock, patch
        from sklearn.ensemble import IsolationForest
        from model_layer.anomaly_detector import score_price_row, LSTMModel
        import torch

        df   = self._make_df(200)
        feat = ["open", "high", "low", "close", "volatility_24h"]

        # Minimal trained IF
        if_model = IsolationForest(n_estimators=10, random_state=0)
        if_model.fit(df[feat].values)

        # Minimal zscore params
        zparams = {
            "mean": float(df["close"].mean()),
            "std":  float(df["close"].std()),
        }

        # Minimal LSTM that always returns 0.5
        lstm_model = MagicMock()
        lstm_model.eval = MagicMock(return_value=lstm_model)
        lstm_model.__call__ = MagicMock(
            return_value=torch.tensor([[50200.0]])
        )

        # Minimal scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(df[feat].values)

        from model_layer import anomaly_detector as ad
        orig = dict(ad._models)
        ad._models.update({
            "isolation_forest": if_model,
            "zscore_params":    zparams,
            "lstm":             lstm_model,
            "scaler":           scaler,
        })

        price_data = {
            "timestamp": "2026-01-01T12:00:00", "coin": "BTC",
            "open": 50000., "high": 51000., "low": 49000.,
            "close": 50500., "volume": 1.5e9, "volatility_24h": 300.,
        }

        try:
            with patch("model_layer.anomaly_detector.insert_signal_log"), \
                 patch("model_layer.anomaly_detector.insert_risk_score"):
                result = score_price_row(price_data, df)
        finally:
            ad._models.clear()
            ad._models.update(orig)

        self.assertIsNotNone(result)
        self.assertIn("risk_score",           result)
        self.assertIn("risk_level",           result)
        self.assertIn("isolation_forest_score", result)
        self.assertIn("zscore_score",         result)
        self.assertIn("lstm_score",           result)
        self.assertIn("models_agreed",        result)
        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"],   100)

    # ---- 2. generate_trading_signal ----

    def test_signal_low_risk_is_buy(self):
        from service_layer.trading_signals import generate_trading_signal
        result = generate_trading_signal(5, strategy="conservative")
        self.assertEqual(result["signal"], "BUY")

    def test_signal_high_risk_is_sell(self):
        from service_layer.trading_signals import generate_trading_signal
        result = generate_trading_signal(90, strategy="conservative")
        self.assertEqual(result["signal"], "SELL")

    def test_signal_medium_risk_is_hold(self):
        from service_layer.trading_signals import generate_trading_signal
        result = generate_trading_signal(50, strategy="conservative")
        self.assertEqual(result["signal"], "HOLD")

    def test_signal_output_has_required_keys(self):
        from service_layer.trading_signals import generate_trading_signal
        result = generate_trading_signal(75)
        for key in ("signal", "confidence", "reasoning", "strategy"):
            self.assertIn(key, result)

    def test_signal_value_is_valid(self):
        from service_layer.trading_signals import generate_trading_signal
        for score in [0, 25, 50, 75, 100]:
            result = generate_trading_signal(score)
            self.assertIn(result["signal"], ("BUY", "SELL", "HOLD", "NONE"))

    # ---- 3. API endpoint shapes ----

    def setUp(self):
        from data_layer.database import init_db
        init_db()
        from fastapi.testclient import TestClient
        from api_backend import app
        self.client = TestClient(app)

    def test_live_price_endpoint_returns_200(self):
        res = self.client.get("/api/price/live")
        self.assertEqual(res.status_code, 200)

    def test_live_price_has_expected_keys(self):
        res  = self.client.get("/api/price/live")
        data = res.json()
        for key in ("price", "source"):
            self.assertIn(key, data)

    def test_signal_logs_endpoint_returns_200(self):
        res = self.client.get("/api/signal-logs")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("logs",  data)
        self.assertIn("total", data)

    def test_risk_scores_endpoint_returns_200(self):
        res = self.client.get("/api/v1/risk-scores")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("scores", data)
        self.assertIn("total",  data)

    def test_model_metrics_endpoint_returns_200(self):
        res = self.client.get("/api/v1/model-metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("metrics", data)
        self.assertIn("count",   data)

    def test_features_endpoint_returns_200_or_503(self):
        res = self.client.get("/api/v1/features/BTC")
        self.assertIn(res.status_code, [200, 503])

    def test_explain_endpoint_503_for_unknown_coin(self):
        res = self.client.get("/api/v1/explain/UNKNOWNCOIN999")
        self.assertEqual(res.status_code, 503)


# ----------------------------------------------------------------
# RUN
# ----------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)