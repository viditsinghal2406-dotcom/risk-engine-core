# Learning Playbook

Use this file to learn the system quickly, especially for demos, viva, or interviews.

## Learning path

1. Read architecture first
   - `docs/ARCHITECTURE.md`

2. Understand model choices
   - `docs/MODELS.md`
   - `docs/MODEL_SELECTION_RATIONALE.md`

3. Understand action output
   - `docs/TRADING_SIGNALS.md`

4. Understand runtime controls
   - `docs/CONFIGURATION.md`

5. Understand APIs
   - `docs/API_REFERENCE.md`

## Explain the project in 60 seconds

"This system ingests live BTC and ETH data, scores anomaly risk with a 3-model dynamic-weight ensemble, converts that score into strategy-aware BUY/HOLD/SELL labels, exposes a standardised v1 API for downstream Series 1 systems, and presents explainable outputs in a single production-grade FastAPI service."

## Demo walkthrough

1. Open Chart tab.
   - Show risk gauge and overlays.
   - Explain why HOLD markers are hidden to reduce clutter.

2. Open Anomaly Log tab.
   - Show real vs synthetic separation.
   - Toggle include_synthetic behavior.

3. Open Learn tab.
   - Show model interpretation and glossary context.

4. Open Decision Layer tab.
   - Switch strategy profile and compare signal behavior.

5. Show API docs at `/docs`.
   - Demonstrate `/api/v1/risk/BTC`, `/api/v1/features/BTC`, `/api/v1/explain/BTC`.
   - Explain these are the upstream feed for downstream Series 1 systems.

## Common interview questions

### Why this model mix?
- It combines structural, statistical, and temporal anomaly views.
- No single method handles noisy non-stationary crypto data reliably.

### Why dynamic weights instead of fixed?
- Validation performance drives weight allocation after each retrain.
- A model with lower val_mae earns more influence, protecting against drift.

### Why separate real and synthetic signals?
- To protect metric and alert integrity while keeping observability continuity.

### Why a single FastAPI service instead of Flask + FastAPI?
- Simpler to deploy to Railway (one `Procfile` line, one `$PORT`).
- FastAPI handles both template rendering and typed APIs with fewer dependencies.
- Easier to reason about when debugging in production.

### How is reliability handled?
- `retry_get()` in `utils.py` wraps all upstream HTTP calls.
- Three-layer fallback: CoinGecko → Binance → SQLite last known price.
- Rotating log files with 10 MB cap and 5 backup files.

### What are known limitations?
- Score quality depends on data continuity and model freshness.
- External APIs can rate-limit and require fallback.
- Per-coin ML models are only enabled for BTC; ETH gets a placeholder 50 score.

## Checklist before showing to others

1. Service running on port 8000 (`uvicorn api_backend:app ...`).
2. Data fetch running and prices updating (check `/health`).
3. Chart overlays toggling correctly.
4. Header metrics and risk badge updating.
5. Strategy switch changes decision output.
6. Anomaly log pagination and clear endpoint work.
7. `/api/v1/risk/BTC` returns a valid RiskScore JSON.
