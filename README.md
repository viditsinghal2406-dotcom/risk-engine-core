# Risk Engine Core

A real-time crypto market risk intelligence system built using machine learning and statistical modeling.

This system ingests live market data, detects anomalies, scores risk, and generates actionable signals through a modular backend architecture.

This project is part of **Series 1: Market Intelligence Systems**.

---

## Overview

The goal of this system is to provide a real-time understanding of market risk for crypto assets.

It combines multiple approaches:

- Anomaly detection using Isolation Forest
- Statistical deviation using Z-Score
- Sequence-based prediction using LSTM

The system produces a unified risk score between 0 and 100 along with signals and confidence levels.

---

## Features

- Live market data ingestion using CoinGecko and Binance
- Multi-model risk scoring ensemble
- Modular architecture with reusable components
- REST API using FastAPI
- Dashboard interface using HTML, CSS, and JavaScript
- Signal generation for trading insights
- Logging and structured outputs for downstream systems

---

## Architecture

The system is structured into multiple layers for scalability and reuse:

```
data_layer      →  data ingestion and storage
feature_engine  →  feature creation and transformations
model_layer     →  ML models and anomaly detection
risk_engine     →  risk scoring and aggregation
service_layer   →  signals and alerts
api_backend     →  API endpoints and serving
```

**Flow:**

```
Market Data → Features → Models → Risk Engine → API → Dashboard
```

---

## Tech Stack

- Python 3.11
- FastAPI
- SQLite
- Pandas
- NumPy
- scikit-learn
- PyTorch
- Plotly

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/viditsinghal2406-dotcom/risk-engine-core.git
cd risk-engine-core
```

Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn api_backend:app --host 0.0.0.0 --port 8000
```

Open:

- Dashboard: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | System health and model readiness |
| `GET /api/price/live` | Live price with risk score and insight |
| `GET /api/chart` | OHLCV candles with anomaly and signal overlays |
| `GET /api/anomalies` | Paginated anomaly log |
| `GET /api/signal-logs` | Every scored row with signal and confidence |
| `GET /api/trading-signal` | Current BUY / HOLD / SELL signal |
| `GET /api/performance` | Aggregate detection analytics |
| `GET /api/forecast` | LSTM iterative price forecast |
| `GET /api/backtest` | Strategy backtest metrics |
| `GET /api/v1/risk/{coin}` | Canonical risk score for downstream systems |
| `GET /api/v1/features/{coin}` | Full technical feature snapshot |
| `GET /api/v1/explain/{coin}` | Per-model reasoning and ensemble weights |
| `POST /api/admin/retrain` | Trigger immediate model retrain |

---

## Sample Output

```json
{
  "coin": "BTC",
  "risk_score": 72.4,
  "risk_level": "High",
  "confidence_level": "High",
  "model_breakdown": {
    "isolation_forest": 65.0,
    "zscore": 78.0,
    "lstm": 74.0
  },
  "signal": "SELL",
  "plain_english_summary": "BTC is showing elevated risk with strong model agreement.",
  "timestamp": "2026-04-08T12:00:00Z"
}
```

---

## Project Structure

```
risk-engine-core/
│
├── data_layer/             data ingestion, caching, and database
├── feature_engine/         technical feature computation
├── model_layer/            ML training, scoring, and forecasting
├── risk_engine/            risk schema and response builders
├── service_layer/          trading signals and alert dispatch
│
├── templates/              dashboard HTML
├── static/                 CSS and JavaScript
├── docs/                   architecture, API reference, models, config
├── tests/                  test suite
├── data/                   seed data
│
├── api_backend.py          FastAPI routes and scheduler
├── config.py               all constants and environment bindings
├── main.py                 thin uvicorn entrypoint
├── utils.py                shared HTTP retry helper
├── schema.sql              canonical database schema
├── requirements.txt
└── README.md
```

---

## Documentation

- `docs/ARCHITECTURE.md` - runtime topology and data flow
- `docs/API_REFERENCE.md` - every endpoint with parameters
- `docs/MODELS.md` - ensemble design, training, and inference
- `docs/TRADING_SIGNALS.md` - strategy profiles and decision layer
- `docs/CONFIGURATION.md` - all config.py settings explained
- `docs/MODEL_SELECTION_RATIONALE.md` - why this model stack
- `docs/LEARNING_PLAYBOOK.md` - demo walkthrough and interview prep

---

## Current Scope

This system focuses on:

- Real-time anomaly detection
- Statistical risk estimation
- LSTM-based price forecasting
- Strategy-aware trading signal generation

---

## Future Work

- Market regime classification
- Volatility forecasting engine
- Contagion detection across assets
- Composite risk scoring
- Explainability layer
- Algorithmic trading integration

---

## Disclaimer

This project is for analytics and education only. It is not financial advice.


## What this project includes

1. **Single-service FastAPI runtime**
   - Dashboard, APIs, and scheduler all run in one process.
   - Port is injected via `$PORT` env var (default `8000`). Railway-ready.

2. **Data reliability pipeline**
   - CoinGecko primary source, Binance fallback, SQLite fallback.
   - Retry logic with configurable backoff.
   - Rotating log files (10 MB, 5 backups).

3. **Risk Engine (dynamic ensemble)**
   - Isolation Forest (25%), Z-Score (25%), LSTM prediction error (50%).
   - Weights are dynamic — recalibrated after each training run based on validation performance.

4. **Standardised risk output (`/api/v1/risk/{coin}`)**
   - Single score `0–100`, levels: Low / Medium / High / Critical.
   - Pydantic `RiskScore` schema consumed by all downstream systems.

5. **Feature engine (`/api/v1/features/{coin}`)**
   - Full technical feature snapshot: MA, Bollinger Bands, momentum, volatility, volume spike, returns.
   - Reusable by 1B–1G without reimplementing.

6. **Explainability (`/api/v1/explain/{coin}`)**
   - Per-model scores, human-readable reasoning, and dynamic ensemble weights.
   - Consumed by 1E and 1F.

7. **Decision Layer**
   - BUY / HOLD / SELL from strategy thresholds.
   - Profiles: conservative, balanced, aggressive, asymmetric.
   - Signal logs persisted to `signal_logs` table.

8. **Database upgrade**
   - `signal_logs` — every scored price row with signal and confidence.
   - `risk_scores` — slim paginated risk history feed.
   - `model_metrics` — per-model metric history (val_loss, val_mae, anomaly_rate).

9. **Signal integrity**
   - Real anomaly signals drive charts, alerts, performance, and backtests.
   - Synthetic continuity signals are observability-only, excluded from metrics.

## Risk formula

```
Risk Score = IF_w * IF_score + Z_w * Z_score + LSTM_w * LSTM_score
```

Default weights: `IF=0.25, Z=0.25, LSTM=0.50`. Recalibrated dynamically after each retrain.

Input features: `open`, `high`, `low`, `close`, `volatility_24h`

## Quick start

```bash
# Activate the venv
1a\Scripts\activate.bat

# Start the server
uvicorn api_backend:app --host 0.0.0.0 --port 8000

# Or via the thin entrypoint
python main.py
```

Open:
- Dashboard: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`

On Railway the `Procfile` handles startup automatically:
```
web: uvicorn api_backend:app --host 0.0.0.0 --port $PORT
```

## Run tests

```bash
python -m pytest tests/tests.py -q
# 92 tests, 0 failures
```

## Project layout

```
api_backend.py          FastAPI routes (all endpoints)
main.py                 Thin uvicorn entrypoint
config.py               All constants and env-var bindings
utils.py                Shared HTTP retry helper (retry_get)
schema.sql              Canonical DB schema reference
Procfile                Railway deployment command

data_layer/
  database.py           Schema, migrations, query layer
  data_pipeline.py      Ingest, cache, seed, backfill

model_layer/
  anomaly_detector.py   ML training, scoring, forecasting

feature_engine/
  features.py           Technical feature computation

risk_engine/
  schemas.py            Pydantic output models (RiskScore, ExplainResponse)
  risk_score.py         build_risk_response, build_explain_response

service_layer/
  trading_signals.py    Strategy profiles, signal generation
  alerts.py             Email/Slack notification dispatch

templates/index.html    Dashboard SPA markup
static/js/app.js        Frontend behaviour and rendering
static/css/styles.css   Visual system and responsive layout
tests/tests.py          92-test suite
docs/                   Architecture, API reference, models, config, learning
```

## API surface summary

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health and model readiness |
| `GET /api/status` | Seeding and model ready status |
| `GET /api/price/live` | Live price + risk + insight block |
| `GET /api/chart` | OHLCV candles + anomaly + signal overlays |
| `GET /api/anomalies` | Paginated anomaly log |
| `GET /api/signal-logs` | Every scored row with signal and confidence |
| `GET /api/trading-signal` | Current BUY/HOLD/SELL signal |
| `GET /api/trading-strategies` | All strategy profiles |
| `GET /api/performance` | Aggregate detection analytics |
| `GET /api/forecast` | LSTM iterative price forecast |
| `GET /api/backtest` | Strategy backtest metrics |
| `GET /api/v1/risk/{coin}` | Canonical RiskScore (downstream feed) |
| `GET /api/v1/features/{coin}` | Full technical feature snapshot |
| `GET /api/v1/explain/{coin}` | Per-model reasoning + ensemble weights |
| `GET /api/v1/risk-scores` | Slim paginated risk score history |
| `GET /api/v1/model-metrics` | Model metric history |
| `POST /api/admin/retrain` | Trigger immediate retrain |

Full reference: `docs/API_REFERENCE.md`

## Documentation map

- `docs/ARCHITECTURE.md` — runtime topology and data flow
- `docs/API_REFERENCE.md` — every endpoint with parameters
- `docs/MODELS.md` — ensemble design, training, inference lifecycle
- `docs/TRADING_SIGNALS.md` — strategy profiles and decision layer
- `docs/CONFIGURATION.md` — all config.py settings
- `docs/MODEL_SELECTION_RATIONALE.md` — why this model stack
- `docs/LEARNING_PLAYBOOK.md` — demo walkthrough and interview prep

## Disclaimer

This project is for analytics and education. It is not financial advice.
