# Models Guide

This document explains the Risk Engine model stack and why it is weighted this way.

## Objective

Convert noisy market behavior into one stable, interpretable risk score from `0` to `100`.

## Ensemble formula

```
Risk Score = IF_w * IF_score + Z_w * Z_score + LSTM_w * LSTM_score
```

Default weights: `IF=0.25, Z=0.25, LSTM=0.50`.

Weights are **dynamic** — after each retrain the validation performance of each model (val_mae for LSTM and Z-Score, anomaly_rate for IF) is used to recalibrate the weights. A model with lower validation error earns a higher weight. This protects against a single model dominating during regime drift.

Why this split:
- IF captures structural outliers in feature space.
- Z-score catches statistical distance shocks quickly.
- LSTM captures temporal regime shifts and trend breaks.

The heavier LSTM default weight improves time-series sensitivity while still anchoring with statistical and structural checks.

## Input features

The model path consumes:
- open
- high
- low
- close
- volatility_24h

## Model details

### Isolation Forest (25%)
Strengths:
- Detects non-linear outliers in OHLCV geometry.

Limitations:
- Does not retain sequence memory.

### Z-Score (25%)
Strengths:
- Fast indicator for abnormal distance from baseline.

Limitations:
- Mean/variance assumptions can drift in long regime shifts.

### LSTM error score (50%)
Strengths:
- Learns sequence context and catches temporal pattern breaks.

Limitations:
- Needs enough stable history and can lag under sudden shocks.

## Confidence and agreement

Confidence is tied to model agreement count.

- 3 agreeing models: high confidence.
- 2 agreeing models: medium confidence.
- 0 to 1 agreeing model: low confidence.

## Signal typing

### Real signal
- True anomaly threshold crossing.
- Included in overlays, alerts, performance, and backtests.

### Synthetic continuity signal
- Emitted for observability continuity when no real signal appears for one hour.
- Excluded from overlays, alerts, performance, and backtests.

## Training lifecycle

1. Load historical rows from `price_data`.
2. Train IF and derive Z-Score mean/std params.
3. Train LSTM on a 60-candle sliding window with an 80/20 train/val split.
4. Recalibrate ensemble weights from validation performance.
5. Save artifacts to `models/`.
6. Register metadata in `model_registry`.
7. Write val_loss, val_mae, and anomaly_rate entries to `model_metrics`.

## Inference lifecycle

1. Build current feature row.
2. Score via IF, Z, and LSTM error.
3. Blend with dynamic weights into 0–100 risk score.
4. Build confidence and reason fields.
5. Classify signal type and persist to `anomalies` and `signal_logs`.
6. Write slim entry to `risk_scores` (consumed by `/api/v1/risk-scores`).
7. Emit `ExplainResponse` with per-model reasoning available at `/api/v1/explain/{coin}`.

## Why this stack fits this project

- It provides both statistical simplicity and temporal depth.
- It is explainable enough for interviews and demos.
- It remains practical for local runtime with SQLite and scheduler jobs.
