# API Reference

Base URL: `http://localhost:8000`  
Railway: `https://<your-service>.railway.app`

All responses are JSON unless noted otherwise.  
Interactive docs: `GET /docs`

---

## Service status

### GET /health
Service health, model readiness, and total price row count.

### GET /api/status
Indicates whether seeding is complete and models are ready. Polled by the dashboard loading overlay.

---

## Live price and risk

### GET /api/price/live
Returns a combined block: live price data, ML risk result, standardised `RiskScore`, and market insight summary.

Query parameters:
- `coin` — `BTC` (default) or `ETH`
- `force_refresh` — bypass the 60-second cache

Response shape:
```json
{
  "price": { ... },
  "risk": { ... },
  "risk_standard": { ... },
  "insight": { "headline": "...", "detail": "..." },
  "source": "coingecko",
  "fetched_at": "2026-04-08T..."
}
```

---

## Chart data

### GET /api/chart
Returns resampled OHLCV candles, real anomaly markers, and trading signal overlays.

Query parameters:
- `range` — `1H`, `1D`, `1W`, `1M`, `All` (default `1D`)
- `strategy` — `conservative`, `balanced`, `aggressive`, `asymmetric`
- `coin` — `BTC` or `ETH`

When the requested frequency produces fewer than 5 candles, the endpoint automatically steps down to a finer frequency.

---

## Anomaly logs

### GET /api/anomalies
Paginated real anomaly rows (score >= `ANOMALY_LOG_THRESHOLD`).

Query parameters:
- `page`, `limit`, `coin`
- `include_synthetic` — `false` (default) excludes continuity-only signals

### GET /api/anomalies/{id}
Single anomaly row by ID.

### DELETE /api/anomalies/clear
Permanently delete all anomaly rows for a coin.

Query parameters:
- `coin`

### GET /api/continuity-signals
Synthetic observability-only signals. Excluded from all metrics, alerts, and backtests.

Query parameters:
- `page`, `limit`, `coin`

### GET /api/signal-logs
Every scored price row (not just anomalies). Includes signal, confidence, ensemble weights, and per-model scores.

Query parameters:
- `page`, `limit`, `coin`

---

## v1 Standardised API (downstream Series 1 feed)

All v1 endpoints use stable Pydantic schemas. These are the canonical outputs consumed by downstream systems 1B-1G.

### GET /api/v1/risk/{coin}
Canonical `RiskScore` for a coin. Returns the most recent scored result.

```json
{
  "coin": "BTC",
  "risk_score": 42.5,
  "risk_level": "Medium",
  "confidence": "High",
  "timestamp": "..."
}
```

### GET /api/v1/features/{coin}
Full technical feature snapshot for the latest candle.
Used by 1B (Regime), 1C (Volatility), 1D (Contagion), 1G (Execution).

Query parameters:
- `limit` — rows of history to compute features over (default 200, max 1000)

### GET /api/v1/explain/{coin}
Risk score WITH per-model reasoning, dynamic ensemble weights, and human-readable justifications.

```json
{
  "coin": "BTC",
  "risk_score": 42.5,
  "model_breakdown": { "isolation_forest": 30.0, "zscore": 22.0, "lstm": 55.0 },
  "reasoning": { "isolation_forest": "Mild outlier...", "zscore": "1.1 sigma above mean...", "lstm": "Prediction error 2.3%" },
  "ensemble_weights": { "isolation_forest": 0.25, "zscore": 0.25, "lstm": 0.50 },
  "models_agreed": 1,
  "confidence": "Low"
}
```

### GET /api/v1/risk-scores
Slim paginated risk score history. Optimised for fast querying by 1E/1G.

Query parameters:
- `page`, `limit`, `coin`

### GET /api/v1/model-metrics
Model performance metric history (val_loss, val_mae, anomaly_rate per model).

Query parameters:
- `model` — filter to one model name, e.g. `lstm`
- `limit` — max rows (default 100, max 1000)

---

## Model intelligence

### GET /api/models  /  GET /api/model-intelligence
Model registry entries, system events, and current anomaly rate.

---

## Decision layer

### GET /api/trading-signal
BUY/HOLD/SELL signal from the cached risk score. No redundant ML inference.

Query parameters:
- `strategy` — override active strategy for this request
- `coin`

### GET /api/trading-strategies
All available strategy profiles with threshold, risk level, and description.

### GET /api/trading-strategy/current
Currently active strategy.

### POST /api/trading-strategy/set
Change active strategy for this session.

Query parameters:
- `strategy` (required)

---

## Analytics

### GET /api/performance
Aggregate anomaly detection stats and score distribution.

Query parameters:
- `days` — 1 to 90 (default 7)

### GET /api/backtest
Strategy backtest metrics using historical real signals.

Query parameters:
- `days`, `coin`

---

## Forecast

### GET /api/forecast
Iterative LSTM price forecast for the next N hours.

Query parameters:
- `steps` — 1 to 48 (default 12)
- `coin`

---

## Export

### GET /api/export/anomalies
Download anomaly log as CSV.

Query parameters:
- `limit` — max rows (default 1000, max 10000)

### GET /api/export/price-data
Download historical price data as CSV.

Query parameters:
- `days` — 1 to 90 (default 30)

---

## Static documentation endpoints

### GET /api/explain/models
Plain-language summaries of each ML model.

### GET /api/explain/risk-levels
Risk level definitions and recommendations.

---

## Admin

### POST /api/admin/retrain
Trigger immediate background model retrain.

### POST /api/admin/alerts/test
Send a test alert through all configured channels.

Query parameters:
- `coin`

### GET /api/admin/alerts/status
Alert channel enablement state and configured thresholds.

---

## Auth (placeholder)

### POST /api/auth/login
### POST /api/auth/logout

Not implemented. Returns placeholder responses.
