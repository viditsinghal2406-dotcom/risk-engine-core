# Model Selection Rationale

This file explains why this project uses Isolation Forest, Z-Score, and LSTM instead of a single model.

## Problem constraints

Crypto data is:
- noisy,
- regime-shifting,
- non-stationary,
- prone to sudden spikes and liquidity shocks.

No single method handles all of those consistently.

## Why these three models

## 1) Isolation Forest

Reason selected:
- Good unsupervised outlier detector for multivariate patterns.
- Works well when anomaly labels are scarce.

What it adds:
- Structural anomaly detection in OHLCV feature space.

## 2) Z-Score

Reason selected:
- Transparent, fast statistical baseline.
- Easy to explain during demos and interviews.

What it adds:
- Immediate sensitivity to extreme distribution shifts.

## 3) LSTM

Reason selected:
- Captures sequence-dependent behavior and trend context.
- Detects regime breaks through prediction error.

What it adds:
- Temporal intelligence that static detectors miss.

## Why LSTM starts with the highest weight

LSTM receives the highest default weight (50%) because:
- Market anomalies often appear as pattern breaks over time, not only point outliers.
- Sequence memory reduces false positives from single-candle noise.

IF and Z still anchor the score with fast structural and statistical checks.

Weights are not fixed. After every nightly retrain the system recalibrates each model's
weight based on validation-set MAE (LSTM) and anomaly_rate deviation (IF/Z-Score).
The updated weights are persisted to the `model_metrics` table and applied on the next
scoring cycle. This means the ensemble self-tunes as market conditions shift.

## Why not only deep learning

A pure deep model can be harder to explain and monitor.
The hybrid approach gives:
- stronger interpretability,
- easier debugging,
- safer behavior during drift periods.

## Real vs synthetic signal separation

This is a core product design choice.

- Real signals represent meaningful market anomalies.
- Synthetic continuity signals keep observability alive when real anomalies are absent.

Keeping them separate avoids metric pollution and protects trust in alerts/backtests.

## Tradeoffs accepted

- Simplicity over extreme model complexity.
- Explainability over marginal benchmark gains.
- Reliability and deterministic behavior over heavy infrastructure dependencies.

## Future model evolution options

- Add regime classifier to gate which weight profile to apply (stable vs crisis).
- Add feature drift monitor and automated retrain trigger policy.
- Add asset-specific weight profiles for BTC vs ETH behavior.
- Extend per-coin scoring beyond BTC (currently BTC-only for full ML; other coins return a neutral 50 placeholder).
