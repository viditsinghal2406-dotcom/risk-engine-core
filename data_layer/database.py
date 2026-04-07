# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# database.py -- All database connections, queries, and helpers
# ============================================================

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Optional

from config import DB_PATH, RETENTION_DAYS

logger = logging.getLogger(__name__)


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


# ----------------------------------------------------------------
# CONNECTION
# ----------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    import os
    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema.sql")
    with open(schema_path, "r") as f:
        schema = f.read()
    with get_connection() as conn:
        conn.executescript(schema)
    # Fix any existing rows where Binance base-asset (BTC) volume was stored instead of quote (USDT)
    _migrate_binance_volumes()
    # Add coin column to support multi-asset (ETH etc.)
    _migrate_add_coin_column()
    # Rebuild legacy tables that were keyed only by timestamp.
    _migrate_multicoin_unique_keys()
    # Remove any stale duplicate anomaly rows, then enforce uniqueness
    _dedup_anomalies()
    _ensure_anomaly_unique_index()
    _migrate_add_signal_fields()
    _migrate_model_val_metrics()
    _migrate_signal_logs()
    _migrate_risk_scores()
    _migrate_model_metrics()
    logger.info("Database initialised successfully.")


def _dedup_anomalies():
    """Keep only the first (lowest id) anomaly per coin+timestamp, deleting the rest."""
    with get_connection() as conn:
        result = conn.execute(
            """
            DELETE FROM anomalies
            WHERE id NOT IN (
                SELECT MIN(id) FROM anomalies GROUP BY coin, timestamp
            )
            """
        )
        if result.rowcount > 0:
            logger.info(f"Removed {result.rowcount} duplicate anomaly row(s).")


def _ensure_anomaly_unique_index():
    """Ensure anomaly uniqueness is per-coin timestamp (not global timestamp)."""
    with get_connection() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_anomaly_timestamp_unique")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_anomaly_timestamp_unique
            ON anomalies(coin, timestamp)
            """
        )


def _migrate_binance_volumes():
    """
    One-time migration: rows seeded from Binance historical klines originally stored
    base-asset (BTC) volume in the volume column instead of quote-asset (USDT) volume.
    Binance BTC daily volume is typically 20,000 – 200,000 BTC, always < 1,000,000.
    Real USD volumes are always in the billions, so any row with volume < 1,000,000
    from a Binance source is safely identified as BTC-unit volume and we approximate
    the USD equivalent as volume * close_price.
    """
    with get_connection() as conn:
        result = conn.execute(
            """
            UPDATE price_data
            SET    volume = volume * close
            WHERE  source = 'binance'
              AND  volume < 1000000
              AND  close  > 0
            """
        )
        if result.rowcount > 0:
            logger.info(f"Migrated {result.rowcount} Binance row(s): BTC volume -> USD volume.")


def _migrate_add_coin_column():
    """Add coin column to price_data and anomalies if not already present (default BTC)."""
    with get_connection() as conn:
        existing_price = [r[1] for r in conn.execute("PRAGMA table_info(price_data)").fetchall()]
        if "coin" not in existing_price:
            conn.execute("ALTER TABLE price_data ADD COLUMN coin TEXT NOT NULL DEFAULT 'BTC'")
            logger.info("Added coin column to price_data.")

        existing_anom = [r[1] for r in conn.execute("PRAGMA table_info(anomalies)").fetchall()]
        if "coin" not in existing_anom:
            conn.execute("ALTER TABLE anomalies ADD COLUMN coin TEXT NOT NULL DEFAULT 'BTC'")
            logger.info("Added coin column to anomalies.")


def _migrate_add_signal_fields():
    """Add anomaly signal classification fields for real vs synthetic continuity logs."""
    with get_connection() as conn:
        existing_anom = [r[1] for r in conn.execute("PRAGMA table_info(anomalies)").fetchall()]

        if "signal_type" not in existing_anom:
            conn.execute("ALTER TABLE anomalies ADD COLUMN signal_type TEXT NOT NULL DEFAULT 'real'")
            logger.info("Added signal_type column to anomalies.")

        if "contributing_models" not in existing_anom:
            conn.execute("ALTER TABLE anomalies ADD COLUMN contributing_models TEXT")
            logger.info("Added contributing_models column to anomalies.")

        if "signal_strength" not in existing_anom:
            conn.execute("ALTER TABLE anomalies ADD COLUMN signal_strength TEXT")
            logger.info("Added signal_strength column to anomalies.")

        conn.execute("UPDATE anomalies SET signal_type = 'real' WHERE signal_type IS NULL OR signal_type = ''")
        # Continuity/heartbeat records have risk_score < 61 (below ANOMALY_LOG_THRESHOLD).
        # Real anomalies are only logged when risk_score >= 61. Reclassify any old heartbeat
        # rows that were bulk-set to 'real' above so the chart stops showing them as anomalies.
        conn.execute(
            "UPDATE anomalies SET signal_type = 'synthetic' WHERE risk_score < 61 AND signal_type = 'real'"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_signal_type ON anomalies(signal_type)")


def _migrate_model_val_metrics():
    """Add val_loss and val_mae columns to model_registry (STEP 4 — ML improvements)."""
    with get_connection() as conn:
        existing = [r[1] for r in conn.execute("PRAGMA table_info(model_registry)").fetchall()]
        if "val_loss" not in existing:
            conn.execute("ALTER TABLE model_registry ADD COLUMN val_loss REAL")
            logger.info("Added val_loss column to model_registry.")
        if "val_mae" not in existing:
            conn.execute("ALTER TABLE model_registry ADD COLUMN val_mae REAL")
            logger.info("Added val_mae column to model_registry.")


def get_model_val_metrics(coin: str = "BTC") -> dict:
    """
    Return the latest val_mae per model_type for the given coin.
    Used by get_ensemble_weights() to compute dynamic weighting.
    Returns dict like {"lstm": 0.012, "isolation_forest": None, "zscore": None}
    """
    coin = coin.upper()
    sql = """
        SELECT model_type, val_mae
        FROM model_registry
        WHERE model_type LIKE ?
          AND trained_at = (
            SELECT MAX(trained_at) FROM model_registry AS m2
            WHERE m2.model_type = model_registry.model_type
          )
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (f"%_{coin}",)).fetchall()
    return {row["model_type"].replace(f"_{coin}", ""): row["val_mae"] for row in rows}


def _migrate_signal_logs():
    """Create signal_logs table if it doesn't exist (runtime migration for existing DBs)."""
    with get_connection() as conn:
        existing = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "signal_logs" not in existing:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signal_logs (
                    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin                   TEXT NOT NULL DEFAULT 'BTC',
                    timestamp              DATETIME NOT NULL,
                    close_price            REAL NOT NULL,
                    risk_score             REAL NOT NULL,
                    risk_level             TEXT NOT NULL,
                    confidence             REAL NOT NULL,
                    signal                 TEXT NOT NULL,
                    isolation_forest_score REAL,
                    zscore_score           REAL,
                    lstm_score             REAL,
                    lstm_predicted_price   REAL,
                    ensemble_weights       TEXT,
                    models_agreed          INTEGER,
                    strategy               TEXT,
                    created_at             DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (coin, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_signal_logs_coin_ts
                    ON signal_logs(coin, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_signal_logs_risk_level
                    ON signal_logs(risk_level);
                CREATE INDEX IF NOT EXISTS idx_signal_logs_signal
                    ON signal_logs(signal);
            """)
            logger.info("Created signal_logs table.")


_CONFIDENCE_NUM = {"High": 1.0, "Medium": 0.67, "Low": 0.33}


def insert_signal_log(result: dict):
    """
    Persist every scored price row to signal_logs (INSERT OR IGNORE on duplicate coin+timestamp).
    Expects the dict produced by score_price_row() plus optional 'signal' and 'strategy' keys.
    """
    sql = """
        INSERT OR IGNORE INTO signal_logs (
            coin, timestamp, close_price, risk_score, risk_level, confidence,
            signal, isolation_forest_score, zscore_score, lstm_score,
            lstm_predicted_price, ensemble_weights, models_agreed, strategy
        ) VALUES (
            :coin, :timestamp, :close_price, :risk_score, :risk_level, :confidence,
            :signal, :isolation_forest_score, :zscore_score, :lstm_score,
            :lstm_predicted_price, :ensemble_weights, :models_agreed, :strategy
        )
    """
    row = dict(result)
    row.setdefault("coin", "BTC")
    row.setdefault("signal", "HOLD")
    row.setdefault("strategy", None)

    # confidence_level is a text field ("High"/"Medium"/"Low"); map to 0-1 float
    confidence_raw = row.get("confidence_level", row.get("confidence", "Low"))
    if isinstance(confidence_raw, str):
        row["confidence"] = _CONFIDENCE_NUM.get(confidence_raw, 0.33)
    else:
        row["confidence"] = float(confidence_raw)

    # ensemble_weights may be a dict — serialise to JSON string
    ew = row.get("ensemble_weights")
    if isinstance(ew, dict):
        row["ensemble_weights"] = json.dumps(ew)

    with get_connection() as conn:
        conn.execute(sql, row)


def get_signal_logs(
    coin: str = "BTC", page: int = 1, limit: int = 50
) -> tuple[list, int]:
    """Return paginated signal_logs rows and total count for the given coin."""
    offset = (page - 1) * limit
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM signal_logs WHERE coin = ?", (coin,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT * FROM signal_logs
            WHERE  coin = ?
            ORDER  BY timestamp DESC
            LIMIT  ? OFFSET ?
            """,
            (coin, limit, offset),
        ).fetchall()
    return list(rows), total


# ----------------------------------------------------------------
# RISK SCORES  (STEP 7 — slim summary for downstream 1E/1G)
# ----------------------------------------------------------------

def _migrate_risk_scores():
    """Create risk_scores table if it doesn't exist (runtime migration)."""
    with get_connection() as conn:
        existing = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "risk_scores" not in existing:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS risk_scores (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin       TEXT NOT NULL DEFAULT 'BTC',
                    timestamp  DATETIME NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (coin, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_risk_scores_coin_ts
                    ON risk_scores(coin, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_risk_scores_risk_level
                    ON risk_scores(risk_level);
            """)
            logger.info("Created risk_scores table.")


def insert_risk_score(result: dict):
    """
    Persist a slim risk score row (INSERT OR IGNORE on duplicate coin+timestamp).
    Expects the dict produced by score_price_row().
    """
    sql = """
        INSERT OR IGNORE INTO risk_scores
            (coin, timestamp, risk_score, risk_level, confidence)
        VALUES
            (:coin, :timestamp, :risk_score, :risk_level, :confidence)
    """
    row = dict(result)
    row.setdefault("coin", "BTC")
    confidence_raw = row.get("confidence_level", row.get("confidence", "Low"))
    if isinstance(confidence_raw, str):
        row["confidence"] = _CONFIDENCE_NUM.get(confidence_raw, 0.33)
    else:
        row["confidence"] = float(confidence_raw)
    with get_connection() as conn:
        conn.execute(sql, row)


def get_risk_scores(
    coin: str = "BTC", page: int = 1, limit: int = 50
) -> tuple[list, int]:
    """Return paginated risk_scores rows and total count for the given coin."""
    offset = (page - 1) * limit
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM risk_scores WHERE coin = ?", (coin,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT * FROM risk_scores
            WHERE  coin = ?
            ORDER  BY timestamp DESC
            LIMIT  ? OFFSET ?
            """,
            (coin, limit, offset),
        ).fetchall()
    return list(rows), total


# ----------------------------------------------------------------
# MODEL METRICS  (STEP 7 — flexible key-value metric history)
# ----------------------------------------------------------------

def _migrate_model_metrics():
    """Create model_metrics table if it doesn't exist (runtime migration)."""
    with get_connection() as conn:
        existing = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        if "model_metrics" not in existing:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name   TEXT NOT NULL,
                    metric_name  TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp    DATETIME NOT NULL,
                    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_model_metrics_model
                    ON model_metrics(model_name, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_model_metrics_metric
                    ON model_metrics(metric_name);
            """)
            logger.info("Created model_metrics table.")


def insert_model_metric(model_name: str, metric_name: str, metric_value: float):
    """Log one named metric for a model at the current UTC time."""
    sql = """
        INSERT INTO model_metrics (model_name, metric_name, metric_value, timestamp)
        VALUES (?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (model_name, metric_name, float(metric_value),
                           datetime.utcnow().isoformat()))


def get_model_metrics(model_name: str = None, limit: int = 200) -> list:
    """Return recent model_metrics rows, optionally filtered by model_name."""
    if model_name:
        sql = """
            SELECT * FROM model_metrics
            WHERE  model_name = ?
            ORDER  BY timestamp DESC
            LIMIT  ?
        """
        with get_connection() as conn:
            return conn.execute(sql, (model_name, limit)).fetchall()
    sql = "SELECT * FROM model_metrics ORDER BY timestamp DESC LIMIT ?"
    with get_connection() as conn:
        return conn.execute(sql, (limit,)).fetchall()


def _migrate_multicoin_unique_keys():
    """
    Legacy schema used UNIQUE(timestamp), which blocks storing BTC and ETH rows
    at the same timestamp. Rebuild tables once with UNIQUE(coin, timestamp).
    """
    with get_connection() as conn:
        # ---- price_data ----
        idx_rows = conn.execute("PRAGMA index_list(price_data)").fetchall()
        price_has_ts_unique = False
        for idx in idx_rows:
            if idx["unique"] != 1:
                continue
            cols = [c["name"] for c in conn.execute(f"PRAGMA index_info({idx['name']})").fetchall()]
            if cols == ["timestamp"]:
                price_has_ts_unique = True
                break

        if price_has_ts_unique:
            conn.execute("ALTER TABLE price_data RENAME TO price_data_old")
            conn.execute(
                """
                CREATE TABLE price_data (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       DATETIME NOT NULL,
                    open            REAL NOT NULL,
                    high            REAL NOT NULL,
                    low             REAL NOT NULL,
                    close           REAL NOT NULL,
                    volume          REAL NOT NULL,
                    volatility_24h  REAL,
                    source          TEXT DEFAULT 'coingecko',
                    coin            TEXT NOT NULL DEFAULT 'BTC',
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, timestamp)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO price_data (id, timestamp, open, high, low, close, volume,
                                        volatility_24h, source, coin, created_at)
                SELECT id, timestamp, open, high, low, close, volume,
                       volatility_24h, source, COALESCE(coin, 'BTC'), created_at
                FROM price_data_old
                """
            )
            conn.execute("DROP TABLE price_data_old")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_data(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_coin_timestamp ON price_data(coin, timestamp DESC)")
            logger.info("Migrated price_data to UNIQUE(coin, timestamp).")

        # ---- anomalies ----
        idx_rows = conn.execute("PRAGMA index_list(anomalies)").fetchall()
        anom_has_ts_unique = False
        for idx in idx_rows:
            if idx["unique"] != 1:
                continue
            cols = [c["name"] for c in conn.execute(f"PRAGMA index_info({idx['name']})").fetchall()]
            if cols == ["timestamp"]:
                anom_has_ts_unique = True
                break

        if anom_has_ts_unique:
            conn.execute("ALTER TABLE anomalies RENAME TO anomalies_old")
            conn.execute(
                """
                CREATE TABLE anomalies (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp               DATETIME NOT NULL,
                    close_price             REAL NOT NULL,
                    volume                  REAL NOT NULL,
                    risk_score              REAL NOT NULL,
                    risk_level              TEXT NOT NULL,
                    isolation_forest_score  REAL NOT NULL,
                    zscore_score            REAL NOT NULL,
                    lstm_score              REAL NOT NULL,
                    if_reason               TEXT,
                    zscore_value            REAL,
                    zscore_reason           TEXT,
                    lstm_predicted_price    REAL,
                    lstm_reason             TEXT,
                    models_agreed           INTEGER,
                    confidence_level        TEXT,
                    plain_english_summary   TEXT,
                    coin                    TEXT NOT NULL DEFAULT 'BTC',
                    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin, timestamp)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO anomalies (id, timestamp, close_price, volume, risk_score, risk_level,
                                       isolation_forest_score, zscore_score, lstm_score,
                                       if_reason, zscore_value, zscore_reason,
                                       lstm_predicted_price, lstm_reason,
                                       models_agreed, confidence_level, plain_english_summary,
                                       coin, created_at)
                SELECT id, timestamp, close_price, volume, risk_score, risk_level,
                       isolation_forest_score, zscore_score, lstm_score,
                       if_reason, zscore_value, zscore_reason,
                       lstm_predicted_price, lstm_reason,
                       models_agreed, confidence_level, plain_english_summary,
                       COALESCE(coin, 'BTC'), created_at
                FROM anomalies_old
                """
            )
            conn.execute("DROP TABLE anomalies_old")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomalies(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_risk_level ON anomalies(risk_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_coin_timestamp ON anomalies(coin, timestamp DESC)")
            logger.info("Migrated anomalies to UNIQUE(coin, timestamp).")


# ----------------------------------------------------------------
# PRICE DATA
# ----------------------------------------------------------------

def insert_price_row(row: dict):
    sql = """
        INSERT OR IGNORE INTO price_data
            (timestamp, open, high, low, close, volume, volatility_24h, source, coin)
        VALUES
            (:timestamp, :open, :high, :low, :close, :volume, :volatility_24h, :source, :coin)
    """
    r = dict(row)
    r.setdefault("coin", "BTC")
    with get_connection() as conn:
        conn.execute(sql, r)


def insert_price_rows_bulk(rows: list[dict]):
    sql = """
        INSERT OR IGNORE INTO price_data
            (timestamp, open, high, low, close, volume, volatility_24h, source, coin)
        VALUES
            (:timestamp, :open, :high, :low, :close, :volume, :volatility_24h, :source, :coin)
    """
    normalised = [{**r, "coin": r.get("coin", "BTC")} for r in rows]
    with get_connection() as conn:
        conn.executemany(sql, normalised)
    logger.info(f"Bulk inserted {len(rows)} price rows.")


def count_price_rows_since(cutoff_iso: str, coin: str = "BTC") -> int:
    """Count price_data rows for a coin with timestamp > cutoff_iso."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM price_data WHERE coin = ? AND timestamp > ?",
            (coin, cutoff_iso)
        ).fetchone()
    return row["cnt"] if row else 0


def get_latest_price(coin: str = "BTC") -> Optional[sqlite3.Row]:
    sql = "SELECT * FROM price_data WHERE coin = ? ORDER BY timestamp DESC LIMIT 1"
    with get_connection() as conn:
        return conn.execute(sql, (coin,)).fetchone()


def get_oldest_price_timestamp(coin: str = "BTC") -> Optional[str]:
    """Return the earliest timestamp string we have for a coin."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT timestamp FROM price_data WHERE coin = ? ORDER BY timestamp ASC LIMIT 1",
            (coin,)
        ).fetchone()
    return row["timestamp"] if row else None


def get_price_history(limit: int = 1000, coin: str = "BTC") -> list[sqlite3.Row]:
    sql = """
        SELECT * FROM price_data
        WHERE coin = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (coin, limit)).fetchall()
    return list(reversed(rows))


def get_price_range(start: datetime, end: datetime, coin: str = "BTC") -> list[sqlite3.Row]:
    sql = """
        SELECT * FROM price_data
        WHERE coin = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """
    with get_connection() as conn:
        return conn.execute(sql, (coin, start.isoformat(), end.isoformat())).fetchall()


def get_anomalies_in_range(start: datetime, end: datetime, coin: str = "BTC") -> list[sqlite3.Row]:
    """Fetch anomaly markers for chart overlay within a time range."""
    with get_connection() as conn:
        if _table_has_column(conn, "anomalies", "signal_type"):
            sql = """
                SELECT timestamp, close_price, risk_score, risk_level, signal_type
                FROM anomalies
                WHERE coin = ?
                  AND COALESCE(signal_type, 'real') = 'real'
                  AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            return conn.execute(sql, (coin, start.isoformat(), end.isoformat())).fetchall()
        sql = """
            SELECT timestamp, close_price, risk_score, risk_level, 'real' AS signal_type
            FROM anomalies
            WHERE coin = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        return conn.execute(sql, (coin, start.isoformat(), end.isoformat())).fetchall()


def get_training_data(coin: str = "BTC") -> list[sqlite3.Row]:
    sql = "SELECT * FROM price_data WHERE coin = ? ORDER BY timestamp ASC"
    with get_connection() as conn:
        return conn.execute(sql, (coin.upper(),)).fetchall()


def count_price_rows() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]


# ----------------------------------------------------------------
# ANOMALIES
# ----------------------------------------------------------------

def insert_anomaly(anomaly: dict):
    """Insert anomaly; silently ignores duplicates at the same timestamp."""
    sql = """
        INSERT OR IGNORE INTO anomalies (
            timestamp, close_price, volume,
            risk_score, risk_level,
            isolation_forest_score, zscore_score, lstm_score,
            if_reason, zscore_value, zscore_reason,
            lstm_predicted_price, lstm_reason,
            models_agreed, confidence_level, plain_english_summary,
            signal_type, contributing_models, signal_strength, coin
        ) VALUES (
            :timestamp, :close_price, :volume,
            :risk_score, :risk_level,
            :isolation_forest_score, :zscore_score, :lstm_score,
            :if_reason, :zscore_value, :zscore_reason,
            :lstm_predicted_price, :lstm_reason,
            :models_agreed, :confidence_level, :plain_english_summary,
            :signal_type, :contributing_models, :signal_strength, :coin
        )
    """
    row = dict(anomaly)
    row.setdefault("coin", "BTC")
    row.setdefault("signal_type", "real")
    row.setdefault("contributing_models", "")
    row.setdefault("signal_strength", "")
    with get_connection() as conn:
        conn.execute(sql, row)


def get_anomalies(page: int = 1, limit: int = 20, coin: str = "BTC") -> tuple[list, int]:
    offset = (page - 1) * limit
    with get_connection() as conn:
        if _table_has_column(conn, "anomalies", "signal_type"):
            total = conn.execute(
                "SELECT COUNT(*) FROM anomalies WHERE coin = ? AND COALESCE(signal_type, 'real') = 'real'",
                (coin,)
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT * FROM anomalies
                WHERE coin = ? AND COALESCE(signal_type, 'real') = 'real'
                ORDER BY timestamp DESC LIMIT ? OFFSET ?
                """,
                (coin, limit, offset)
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM anomalies WHERE coin = ?", (coin,)).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM anomalies WHERE coin = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (coin, limit, offset)
            ).fetchall()
    return list(rows), total


def get_continuity_signals(page: int = 1, limit: int = 20, coin: str = "BTC") -> tuple[list, int]:
    """Return synthetic continuity signals (kept separate from real anomalies)."""
    offset = (page - 1) * limit
    with get_connection() as conn:
        if not _table_has_column(conn, "anomalies", "signal_type"):
            return [], 0
        total = conn.execute(
            "SELECT COUNT(*) FROM anomalies WHERE coin = ? AND COALESCE(signal_type, 'real') = 'synthetic'",
            (coin,)
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT * FROM anomalies
            WHERE coin = ? AND COALESCE(signal_type, 'real') = 'synthetic'
            ORDER BY timestamp DESC LIMIT ? OFFSET ?
            """,
            (coin, limit, offset)
        ).fetchall()
    return list(rows), total


def get_all_anomalies(page: int = 1, limit: int = 20, coin: str = "BTC") -> tuple[list, int]:
    """Return all anomalies including synthetic continuity signals (for the toggle view)."""
    offset = (page - 1) * limit
    with get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM anomalies WHERE coin = ?", (coin,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM anomalies WHERE coin = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (coin, limit, offset)
        ).fetchall()
    return list(rows), total


def get_anomaly_by_id(anomaly_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM anomalies WHERE id = ?", (anomaly_id,)
        ).fetchone()


def clear_anomaly_logs(coin: str = "BTC") -> int:
    """Delete all anomaly rows (real and synthetic) for a coin. Returns deleted count."""
    with get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM anomalies WHERE coin = ?", (coin,)
        ).fetchone()[0]
        conn.execute("DELETE FROM anomalies WHERE coin = ?", (coin,))
    return count


# ----------------------------------------------------------------
# MODEL REGISTRY
# ----------------------------------------------------------------

def register_model(model_type: str, file_path: str,
                   training_data_size: int, anomaly_rate: float,
                   val_loss: float = None, val_mae: float = None):
    sql = """
        INSERT INTO model_registry
            (model_type, file_path, training_data_size, anomaly_rate, val_loss, val_mae, trained_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            model_type, file_path,
            training_data_size, anomaly_rate,
            val_loss, val_mae,
            datetime.utcnow().isoformat()
        ))


def get_model_registry() -> list[sqlite3.Row]:
    sql = """
        SELECT * FROM model_registry
        WHERE trained_at = (
            SELECT MAX(trained_at) FROM model_registry AS m2
            WHERE m2.model_type = model_registry.model_type
        )
        ORDER BY model_type
    """
    with get_connection() as conn:
        return conn.execute(sql).fetchall()


def get_latest_models(coin: str = "BTC") -> dict:
    coin = coin.upper()
    sql = """
        SELECT model_type, file_path
        FROM model_registry
        WHERE model_type LIKE ?
          AND trained_at = (
            SELECT MAX(trained_at) FROM model_registry AS m2
            WHERE m2.model_type = model_registry.model_type
          )
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (f"%_{coin}",)).fetchall()
    # Strip coin suffix so caller gets clean keys: "lstm", "isolation_forest", "zscore"
    return {row["model_type"].replace(f"_{coin}", ""): row["file_path"] for row in rows}


# ----------------------------------------------------------------
# SYSTEM EVENTS
# ----------------------------------------------------------------

def log_event(event_type: str, message: str, detail: dict = None):
    sql = """
        INSERT INTO system_events (event_type, message, detail)
        VALUES (?, ?, ?)
    """
    detail_json = json.dumps(detail) if detail else None
    with get_connection() as conn:
        conn.execute(sql, (event_type, message, detail_json))


def get_recent_events(limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM system_events ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()


# ----------------------------------------------------------------
# NIGHTLY PURGE
# ----------------------------------------------------------------

def purge_old_price_data():
    cutoff = (datetime.utcnow() - timedelta(days=RETENTION_DAYS)).isoformat()
    with get_connection() as conn:
        result  = conn.execute(
            "DELETE FROM price_data WHERE timestamp < ?", (cutoff,)
        )
        deleted = result.rowcount
    logger.info(f"Purged {deleted} rows older than {RETENTION_DAYS} days.")
    log_event("purge", f"Purged {deleted} old price rows.", {"cutoff": cutoff})


def purge_old_models(keep_last: int = 7):
    import os
    model_types = ["isolation_forest", "zscore", "lstm"]
    with get_connection() as conn:
        for mtype in model_types:
            rows = conn.execute(
                """SELECT id, file_path FROM model_registry
                   WHERE model_type = ?
                   ORDER BY trained_at DESC""",
                (mtype,)
            ).fetchall()
            to_delete = rows[keep_last:]
            for row in to_delete:
                if os.path.exists(row["file_path"]):
                    os.remove(row["file_path"])
                conn.execute(
                    "DELETE FROM model_registry WHERE id = ?", (row["id"],)
                )
            if to_delete:
                logger.info(f"Purged {len(to_delete)} old {mtype} model files.")


# ----------------------------------------------------------------
# CSV EXPORT
# ----------------------------------------------------------------

def export_anomalies_to_csv(output_path: str, limit: Optional[int] = None) -> str:
    """
    Export anomalies to CSV file.
    Returns the path to the created CSV file.
    """
    import csv
    
    sql = "SELECT * FROM anomalies ORDER BY timestamp DESC"
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        
        for row in rows:
            writer.writerow([row[col] for col in columns])
    
    logger.info(f"Exported {len(rows)} anomalies to {output_path}")
    return output_path


def export_price_data_to_csv(output_path: str, start: Optional[datetime] = None,
                             end: Optional[datetime] = None) -> str:
    """
    Export price data to CSV file.
    Returns the path to the created CSV file.
    """
    import csv
    
    sql = "SELECT * FROM price_data"
    params = []
    
    if start and end:
        sql += " WHERE timestamp BETWEEN ? AND ?"
        params = [start.isoformat(), end.isoformat()]
    
    sql += " ORDER BY timestamp ASC"
    
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        
        for row in rows:
            writer.writerow([row[col] for col in columns])
    
    logger.info(f"Exported {len(rows)} price rows to {output_path}")
    return output_path


# ----------------------------------------------------------------
# MODEL PERFORMANCE TRACKING
# ----------------------------------------------------------------

def get_model_performance_stats(days: int = 7) -> dict:
    """
    Get model performance metrics over the past N days.
    These metrics indicate how accurate each model is at detecting anomalies.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    with get_connection() as conn:
        if _table_has_column(conn, "anomalies", "signal_type"):
            sql = """
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(isolation_forest_score) as avg_if_score,
                    AVG(zscore_score) as avg_z_score,
                    AVG(lstm_score) as avg_lstm_score,
                    AVG(risk_score) as avg_risk_score,
                    MAX(risk_score) as max_risk_score,
                    MIN(risk_score) as min_risk_score,
                    SUM(CASE WHEN risk_level = 'Critical' THEN 1 ELSE 0 END) as critical_count,
                    SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_count,
                    SUM(CASE WHEN models_agreed >= 2 THEN 1 ELSE 0 END) as consensus_count
                FROM anomalies
                WHERE timestamp >= ?
                  AND COALESCE(signal_type, 'real') = 'real'
            """
        else:
            sql = """
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(isolation_forest_score) as avg_if_score,
                    AVG(zscore_score) as avg_z_score,
                    AVG(lstm_score) as avg_lstm_score,
                    AVG(risk_score) as avg_risk_score,
                    MAX(risk_score) as max_risk_score,
                    MIN(risk_score) as min_risk_score,
                    SUM(CASE WHEN risk_level = 'Critical' THEN 1 ELSE 0 END) as critical_count,
                    SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_count,
                    SUM(CASE WHEN models_agreed >= 2 THEN 1 ELSE 0 END) as consensus_count
                FROM anomalies
                WHERE timestamp >= ?
            """
        row = conn.execute(sql, (cutoff,)).fetchone()
        
        # Get total price predictions in this window
        total_prices = conn.execute(
            "SELECT COUNT(*) FROM price_data WHERE timestamp >= ?",
            (cutoff,)
        ).fetchone()[0]
    
    if not row or row["total_predictions"] == 0:
        return {
            "performance_window_days": days,
            "total_predictions": 0,
            "avg_if_score": 0,
            "avg_z_score": 0,
            "avg_lstm_score": 0,
            "avg_risk_score": 0,
            "max_risk_score": 0,
            "min_risk_score": 0,
            "critical_count": 0,
            "high_count": 0,
            "consensus_rate": 0,
            "anomaly_rate": 0,
        }
    
    total = row["total_predictions"] or 1
    
    # CRITICAL FIX: Ensure anomaly rate doesn't exceed 100%
    # This happens when recent data is sparse
    anomaly_numerator = total
    anomaly_denominator = max(total_prices, 1)
    anomaly_rate_pct = (anomaly_numerator / anomaly_denominator) * 100
    anomaly_rate_pct = min(100.0, anomaly_rate_pct)  # Cap at 100%
    
    return {
        "performance_window_days": days,
        "total_predictions": row["total_predictions"],
        "avg_if_score": round(row["avg_if_score"] or 0, 2),
        "avg_z_score": round(row["avg_z_score"] or 0, 2),
        "avg_lstm_score": round(row["avg_lstm_score"] or 0, 2),
        "avg_risk_score": round(row["avg_risk_score"] or 0, 2),
        "max_risk_score": round(row["max_risk_score"] or 0, 2),
        "min_risk_score": round(row["min_risk_score"] or 0, 2),
        "critical_count": row["critical_count"],
        "high_count": row["high_count"],
        "consensus_rate": round((row["consensus_count"] / total * 100) if total > 0 else 0, 1),
        "anomaly_rate": round(anomaly_rate_pct, 1),
    }


def get_anomaly_distribution() -> dict:
    """Get distribution of anomalies by risk level."""
    with get_connection() as conn:
        if _table_has_column(conn, "anomalies", "signal_type"):
            sql = """
                SELECT
                    risk_level,
                    COUNT(*) as count,
                    AVG(risk_score) as avg_score
                FROM anomalies
                WHERE COALESCE(signal_type, 'real') = 'real'
                GROUP BY risk_level
                ORDER BY avg_score DESC
            """
        else:
            sql = """
                SELECT
                    risk_level,
                    COUNT(*) as count,
                    AVG(risk_score) as avg_score
                FROM anomalies
                GROUP BY risk_level
                ORDER BY avg_score DESC
            """
        rows = conn.execute(sql).fetchall()
    return {
        row["risk_level"]: {
            "count": row["count"],
            "avg_score": round(row["avg_score"], 2),
        }
        for row in rows
    }


def get_24h_high_low(coin: str = "BTC") -> dict:
    """Get actual 24-hour high, low, and total volume from price_data."""
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()

    with get_connection() as conn:
        hl_row = conn.execute(
            """
            SELECT MAX(high) as high_24h, MIN(low) as low_24h
            FROM price_data
            WHERE coin = ? AND timestamp >= ?
            """,
            (coin, cutoff),
        ).fetchone()

        # Sum per-interval volumes across the 24h window. Each stored row now
        # holds a per-polling-interval fraction of the 24h total, so summing
        # them reconstructs the correct 24h traded volume.
        vol_row = conn.execute(
            "SELECT SUM(volume) as vol_24h FROM price_data WHERE coin = ? AND timestamp >= ?",
            (coin, cutoff),
        ).fetchone()

    if hl_row is None or hl_row["high_24h"] is None:
        return {"high": 0, "low": 0, "volume": 0}

    return {
        "high":   round(hl_row["high_24h"], 2),
        "low":    round(hl_row["low_24h"], 2),
        "volume": round(vol_row["vol_24h"], 2) if vol_row and vol_row["vol_24h"] else 0,
    }


def get_price_24h_ago(coin: str = "BTC") -> float:
    """Return the close price from the row closest to 24 hours ago."""
    cutoff_hi = (datetime.utcnow() - timedelta(hours=23)).isoformat()
    cutoff_lo = (datetime.utcnow() - timedelta(hours=25)).isoformat()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT close FROM price_data
            WHERE coin = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (coin, cutoff_lo, cutoff_hi),
        ).fetchone()
    return float(row["close"]) if row else 0.0


# ----------------------------------------------------------------
# SIGNAL BACKTEST ACCURACY
# ----------------------------------------------------------------

def get_signal_backtest(days: int = 30, coin: str = "BTC") -> dict:
    """
    For every anomaly logged in the past N days, look up what price did 24h later
    and score whether the implied signal direction was correct.
    BUY (risk <= 30): correct if price rose after 24h.
    SELL (risk >= 61): correct if price fell after 24h.
    HOLD (31-60): correct if price stayed within +-3%.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    sql_real = """
        SELECT
            a.timestamp,
            a.risk_score,
            a.risk_level,
            a.close_price,
            (
                SELECT p.close FROM price_data p
                                WHERE p.coin = a.coin
                                    AND (julianday(p.timestamp) - julianday(a.timestamp)) BETWEEN (23.0/24.0) AND (25.0/24.0)
                ORDER BY p.timestamp ASC LIMIT 1
            ) AS price_24h_later
        FROM anomalies a
            WHERE a.coin = ?
              AND a.timestamp >= ?
              AND COALESCE(a.signal_type, 'real') = 'real'
        ORDER BY a.timestamp DESC
    """

    sql_legacy = """
        SELECT
            a.timestamp,
            a.risk_score,
            a.risk_level,
            a.close_price,
            (
                SELECT p.close FROM price_data p
                                WHERE p.coin = a.coin
                                    AND (julianday(p.timestamp) - julianday(a.timestamp)) BETWEEN (23.0/24.0) AND (25.0/24.0)
                ORDER BY p.timestamp ASC LIMIT 1
            ) AS price_24h_later
        FROM anomalies a
                WHERE a.coin = ? AND a.timestamp >= ?
        ORDER BY a.timestamp DESC
    """

    with get_connection() as conn:
        sql = sql_real if _table_has_column(conn, "anomalies", "signal_type") else sql_legacy
        rows = conn.execute(sql, (coin, cutoff)).fetchall()

    results       = []
    correct_buys  = correct_sells  = correct_holds  = 0
    total_buys    = total_sells    = total_holds     = 0

    for row in rows:
        r = dict(row)
        if r["price_24h_later"] is None:
            continue
        pct = (r["price_24h_later"] - r["close_price"]) / max(r["close_price"], 1) * 100

        if r["risk_score"] <= 30:
            signal  = "BUY"
            correct = pct > 0
            total_buys += 1
            if correct: correct_buys += 1
        elif r["risk_score"] >= 61:
            signal  = "SELL"
            correct = pct < 0
            total_sells += 1
            if correct: correct_sells += 1
        else:
            signal  = "HOLD"
            correct = abs(pct) < 3
            total_holds += 1
            if correct: correct_holds += 1

        results.append({
            "timestamp":       r["timestamp"],
            "signal":          signal,
            "risk_score":      round(r["risk_score"], 1),
            "close_price":     r["close_price"],
            "price_24h_later": round(r["price_24h_later"], 2),
            "pct_change_24h":  round(pct, 2),
            "correct":         correct,
        })

    def acc(c, t):
        return round(c / t * 100, 1) if t > 0 else None

    return {
        "window_days": days,
        "coin": coin,
        "total_signals": len(results),
        "summary": {
            "BUY": {
                "accuracy": acc(correct_buys, total_buys),
                "correct": correct_buys,
                "total": total_buys,
            },
            "SELL": {
                "accuracy": acc(correct_sells, total_sells),
                "correct": correct_sells,
                "total": total_sells,
            },
            "HOLD": {
                "accuracy": acc(correct_holds, total_holds),
                "correct": correct_holds,
                "total": total_holds,
            },
        },
        "signals": results[:50],
    }