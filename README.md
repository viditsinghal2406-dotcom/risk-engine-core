# Risk Engine Core

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Enabled-orange)
![Made by](https://img.shields.io/badge/Made%20by-Vidit%20Singhal-blueviolet)

A real-time crypto market risk intelligence system built using machine learning and statistical modeling.

This system ingests live market data, detects anomalies, scores risk, and generates actionable signals through a modular backend architecture.

This project is part of **Series 1: Market Intelligence Systems**.

---

## Why This Project

Crypto markets are highly volatile and difficult to interpret in real time.

This system is designed to:

- Detect abnormal market behavior early
- Quantify risk using multiple models
- Provide structured signals for decision making

It acts as the foundational layer for a larger market intelligence platform.

---

## Overview

The system combines three approaches into a single unified risk score (0-100):

- **Isolation Forest** - unsupervised anomaly detection
- **Z-Score** - statistical deviation from historical mean
- **LSTM** - sequence-based prediction error

Scores map to: `Low -> Medium -> High -> Critical`

---

## Features

- Live market data ingestion via CoinGecko and Binance
- Multi-model risk scoring ensemble with dynamic weights
- Modular architecture with reusable components
- REST API using FastAPI
- Interactive dashboard using HTML, CSS, and JavaScript
- Strategy-aware trading signals (BUY / HOLD / SELL)
- Per-coin ML scoring for BTC and ETH
- Logging and structured outputs for downstream systems

---

## Architecture

```
data_layer      -> data ingestion and storage
feature_engine  -> feature creation and transformations
model_layer     -> ML models and anomaly detection
risk_engine     -> risk scoring and aggregation
service_layer   -> trading signals and alerts
api_backend     -> API endpoints and scheduler
```

**Flow:**

```
Market Data -> Features -> Models -> Risk Engine -> API -> Dashboard
```

---

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- SQLite
- Pandas / NumPy
- scikit-learn
- PyTorch
- Plotly

---

## Quick Start

```bash
git clone https://github.com/viditsinghal2406-dotcom/risk-engine-core.git
cd risk-engine-core

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

uvicorn api_backend:app --host 0.0.0.0 --port 8000
```

Open:

- Dashboard: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

Run tests:

```bash
python -m pytest tests/tests.py -q
```

---

## API

Key endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | System health and model readiness |
| `GET /api/price/live` | Live price with risk score and insight |
| `GET /api/chart` | OHLCV candles with anomaly and signal overlays |
| `GET /api/trading-signal` | Current BUY / HOLD / SELL signal |
| `GET /api/forecast` | LSTM price forecast |
| `GET /api/v1/risk/{coin}` | Canonical risk score for downstream systems |
| `GET /api/v1/explain/{coin}` | Per-model reasoning and ensemble weights |
| `POST /api/admin/retrain` | Trigger immediate model retrain |

Full reference: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

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
|
|-- data_layer/         data ingestion, caching, and database
|-- feature_engine/     technical feature computation
|-- model_layer/        ML training, scoring, and forecasting
|-- risk_engine/        risk schema and response builders
|-- service_layer/      trading signals and alert dispatch
|
|-- templates/          dashboard HTML
|-- static/             CSS and JavaScript
|-- docs/               architecture, API reference, models, config
|-- tests/              test suite
|-- data/               seed data
|
|-- api_backend.py      FastAPI routes and scheduler
|-- config.py           all constants and environment bindings
|-- main.py             thin uvicorn entrypoint
|-- schema.sql          canonical database schema
`-- requirements.txt
```

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - runtime topology and data flow
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) - every endpoint with parameters
- [`docs/MODELS.md`](docs/MODELS.md) - ensemble design, training, and inference
- [`docs/TRADING_SIGNALS.md`](docs/TRADING_SIGNALS.md) - strategy profiles and decision layer
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) - all config.py settings explained
- [`docs/MODEL_SELECTION_RATIONALE.md`](docs/MODEL_SELECTION_RATIONALE.md) - why this model stack
- [`docs/LEARNING_PLAYBOOK.md`](docs/LEARNING_PLAYBOOK.md) - demo walkthrough and interview prep

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

---

Built by **Vidit Singhal** - Series 1, System 1A.

<!-- Made by Vidit Singhal - github.com/viditsinghal2406-dotcom -->

