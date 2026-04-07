# Architecture Guide

This guide explains runtime components and data flow for the platform.

## Runtime topology

The app runs as a **single FastAPI service** on port `$PORT` (default `8000`).

- Dashboard HTML is rendered by Jinja2 templates served by the same FastAPI app.
- All API endpoints, static assets, and the scheduler share one process.
- Port is injected as `$PORT` env var — Railway-compatible out of the box.

A `BackgroundScheduler` (APScheduler) starts in a daemon thread on startup for fetch, retrain, and retention jobs.

## Core components

1. **FastAPI service** (`api_backend.py`)
   - Serves dashboard, all REST endpoints, static files.
   - Startup runs in a daemon thread via the `lifespan` context manager.

2. **Scheduler** (APScheduler)
   - Periodic price fetch every `POLLING_INTERVAL_SECONDS`.
   - Nightly model retrain (midnight UTC).
   - Hourly data retention cleanup.

3. **SQLite** (`crypto_risk.db`)
   - Stores OHLCV history, anomaly logs, signal logs, risk scores, model metrics, model registry, and system events.

## Data to decision pipeline

1. **Ingestion** — CoinGecko primary, Binance fallback, SQLite fallback. Retry logic via `utils.retry_get()`.

2. **Feature engineering** — `feature_engine/features.py` computes MA, Bollinger Bands, momentum, volatility, volume spike, returns.

3. **Ensemble scoring** — Isolation Forest + Z-Score + LSTM error blended with dynamic weights into a 0–100 risk score.

4. **Standardised output** — `risk_engine/risk_score.py` builds `RiskScore` and `ExplainResponse` Pydantic models consumed by all downstream systems.

5. **Signal typing** — Real signal for meaningful anomaly threshold crossings. Synthetic continuity signal for observability only.

6. **Decision Layer** — Strategy thresholds map score to BUY / HOLD / SELL. Every scored row is persisted to `signal_logs`.

7. **Persistence** — `risk_scores` table feeds downstream `/api/v1/risk-scores`. `model_metrics` records training performance history.

8. **Alerting and explainability** — Alerts run only for real severe signals. `/api/v1/explain/{coin}` returns per-model reasoning and dynamic weights.

## Startup lifecycle

1. Initialize DB and run all migrations (idempotent).
2. Seed and backfill data when needed.
3. Ensure minimum 700-day history depth per asset.
4. Load saved model artifacts or train from scratch if missing.
5. Start APScheduler (price fetch, retrain, purge jobs).
6. FastAPI lifespan yields — service is ready to handle requests.

## Chart architecture notes

- Real-time chart uses Plotly.
- Volume is rendered inline on a secondary y-axis.
- HOLD markers are omitted to reduce marker noise.
- Sparse historical ranges use automatic frequency fallback to avoid empty views.

## Package layout

```
data_layer/     database.py, data_pipeline.py
model_layer/    anomaly_detector.py
feature_engine/ features.py
risk_engine/    schemas.py, risk_score.py
service_layer/  trading_signals.py, alerts.py
```

## Storage model

| Table | Contents |
|---|---|
| `price_data` | OHLCV rows by coin |
| `anomalies` | Risk outputs with per-model scores, confidence, signal type |
| `signal_logs` | Every scored price row with signal and ensemble weights |
| `risk_scores` | Slim coin+timestamp+score+level+confidence feed |
| `model_metrics` | Per-model training metric history (val_loss, val_mae, anomaly_rate) |
| `model_registry` | Artifact metadata and training statistics |
| `system_events` | Operational timeline for retrain, fallback, and purge events |
| `users` | Auth placeholder (not implemented yet) |

## Reliability design

- `utils.retry_get()` wraps all upstream HTTP calls with configurable retries and backoff.
- Multi-layer fallback: CoinGecko → Binance → SQLite last known price.
- In-memory cache per coin to reduce rate-limit pressure.
- Rotating log files (10 MB, 5 backups) via `setup_logging()`.
- Separation of real vs synthetic signals to protect metric and alert integrity.
