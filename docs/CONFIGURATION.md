# Configuration Guide

This guide summarizes key settings from `config.py`.

## Paths and storage

- `BASE_DIR`: project root
- `DATA_DIR`: seed data folder
- `MODELS_DIR`: model artifacts
- `LOGS_DIR`: runtime logs
- `DB_PATH`: SQLite file path

## Server port

- `PORT` — read from `$PORT` env var, defaults to `8000`.
- On Railway, `$PORT` is injected automatically. The `Procfile` passes it to uvicorn.
- `FLASK_PORT` remains in config for backward import compatibility but is not used.

## Logging

- `LOG_FILE`: path to the rotating log file
- `LOG_MAX_BYTES`: `10_485_760` (10 MB per file before rotation)
- `LOG_BACKUP_COUNT`: `5` (keep last 5 rotated files)
- `setup_logging()`: call once at startup — idempotent, configures both console and rotating file handler

## Data behavior

- `POLLING_INTERVAL_SECONDS`
- `CACHE_TTL_SECONDS`
- `API_MAX_RETRIES`
- `API_RETRY_DELAY`
- `COIN_CONFIG` (asset mapping)

## Retention

- `RETENTION_DAYS` controls historical cleanup window.

## Risk weights

- `WEIGHT_ISOLATION_FOREST`
- `WEIGHT_ZSCORE`
- `WEIGHT_LSTM`

These should sum to `1.0`.

## Risk thresholds

- `RISK_LOW_MAX`
- `RISK_MEDIUM_MAX`
- `RISK_HIGH_MAX`
- `ANOMALY_LOG_THRESHOLD`

## LSTM settings

- `LSTM_SEQUENCE_LENGTH`
- `LSTM_EPOCHS`
- `LSTM_BATCH_SIZE`
- `LSTM_HIDDEN_SIZE`
- `LSTM_NUM_LAYERS`

## Scheduler timing (UTC)

- `RETRAIN_HOUR`, `RETRAIN_MINUTE`
- `PURGE_HOUR`, `PURGE_MINUTE`

## Chart range mapping

`CHART_RANGE_CONFIG` maps each UI range to:
- aggregation frequency
- lookback window

## Strategy config

`TRADING_STRATEGIES` contains threshold policies:
- conservative
- balanced
- aggressive
- asymmetric

`CURRENT_TRADING_STRATEGY` sets startup default.

## Alerts

Email:
- `ALERT_EMAIL_ENABLED`
- `ALERT_EMAIL_FROM`
- `ALERT_EMAIL_TO`
- `ALERT_EMAIL_SMTP_URL`
- `ALERT_EMAIL_PASSWORD`

Slack:
- `ALERT_SLACK_ENABLED`
- `ALERT_SLACK_WEBHOOK`

Trigger:
- `ALERT_RISK_THRESHOLD`

Only real high-severity signals should trigger alerts.
Synthetic continuity signals are excluded.

## Practical setup recommendation

1. Run `uvicorn api_backend:app --host 0.0.0.0 --port 8000` locally.
2. Validate ingestion and chart output before changing any thresholds.
3. Enable alerts only after email/Slack credentials are verified.
4. Tune thresholds only with documented test windows.
5. For Railway deploy: set `$PORT` is handled automatically, set any alert env vars in Railway's Variables tab.
