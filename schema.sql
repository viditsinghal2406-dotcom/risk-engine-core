-- ============================================================
-- Project 1a: Crypto Market Risk Intelligence System
-- schema.sql
-- ============================================================

CREATE TABLE IF NOT EXISTS price_data (
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
    UNIQUE (coin, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_price_coin_timestamp ON price_data(coin, timestamp DESC);

CREATE TABLE IF NOT EXISTS anomalies (
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
    signal_type             TEXT NOT NULL DEFAULT 'real',
    contributing_models     TEXT,
    signal_strength         TEXT,
    coin                    TEXT NOT NULL DEFAULT 'BTC',
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (coin, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomalies(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_risk_level ON anomalies(risk_level);
CREATE INDEX IF NOT EXISTS idx_anomaly_coin_timestamp ON anomalies(coin, timestamp DESC);

CREATE TABLE IF NOT EXISTS model_registry (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type          TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    training_data_size  INTEGER,
    anomaly_rate        REAL,
    val_loss            REAL,
    val_mae             REAL,
    trained_at          DATETIME NOT NULL,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    message     TEXT NOT NULL,
    detail      TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_created ON system_events(created_at DESC);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT DEFAULT 'viewer',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login      DATETIME
);

-- ----------------------------------------------------------------
-- SIGNAL LOGS  (every prediction stored for audit + downstream use)
-- STEP 5 — Series 1 Foundation
-- ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS signal_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    coin                    TEXT NOT NULL DEFAULT 'BTC',
    timestamp               DATETIME NOT NULL,
    close_price             REAL NOT NULL,
    risk_score              REAL NOT NULL,
    risk_level              TEXT NOT NULL,
    confidence              REAL NOT NULL,
    signal                  TEXT NOT NULL,           -- BUY | SELL | HOLD
    isolation_forest_score  REAL,
    zscore_score            REAL,
    lstm_score              REAL,
    lstm_predicted_price    REAL,
    ensemble_weights        TEXT,                    -- JSON: {"isolation_forest":0.25,...}
    models_agreed           INTEGER,
    strategy                TEXT,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (coin, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_signal_logs_coin_ts ON signal_logs(coin, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signal_logs_risk    ON signal_logs(risk_level);
CREATE INDEX IF NOT EXISTS idx_signal_logs_signal  ON signal_logs(signal);

-- ----------------------------------------------------------------
-- RISK SCORES  (slim summary — fast lookup for downstream 1E/1G)
-- STEP 7 — Database Upgrade
-- ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS risk_scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    coin         TEXT NOT NULL DEFAULT 'BTC',
    timestamp    DATETIME NOT NULL,
    risk_score   REAL NOT NULL,
    risk_level   TEXT NOT NULL,
    confidence   REAL NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (coin, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_risk_scores_coin_ts    ON risk_scores(coin, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_risk_level ON risk_scores(risk_level);

-- ----------------------------------------------------------------
-- MODEL METRICS  (flexible key-value metric history per model)
-- STEP 7 — Database Upgrade
-- ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp    DATETIME NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_model  ON model_metrics(model_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_metrics_metric ON model_metrics(metric_name);