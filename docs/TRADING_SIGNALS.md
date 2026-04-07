# Decision Layer Guide

This guide explains how risk scores become action labels.

## Core mapping

Each strategy has two thresholds:
- `buy_threshold`
- `sell_threshold`

Rules:
1. `risk <= buy_threshold` => `BUY`
2. `risk >= sell_threshold` => `SELL`
3. otherwise => `HOLD`

## Why this layer exists

Risk score describes market condition.
Decision label describes action zone.

That separation keeps the product understandable for non-ML users.

## Strategy profiles

### Conservative
- Lower BUY threshold and earlier SELL threshold.
- Fewer actions, lower noise.

### Balanced
- Midpoint profile for default usage and demos.

### Aggressive
- Wider tolerance and more action frequency.

### Asymmetric
- Permissive BUY side with stricter risk exit behavior.

## Confidence interpretation

Signal confidence increases when score is deeper into BUY or SELL zones.
Scores near boundaries produce lower confidence.

## UI integration

- Decision tab shows current strategy output and rationale.
- Chart overlays can display BUY/SELL markers.
- Strategy switch updates signal mapping in real time.

## Integrity rules

- Real anomaly signals feed analytics and backtests.
- Synthetic continuity signals are observability-only and excluded from performance metrics.

## Signal persistence

Every scored price row — whether it crosses an anomaly threshold or not — is written
to the `signal_logs` table. This gives a complete audit trail of the Decision Layer
output and lets downstream systems replay or backtest a full decision history.

The slim historical feed is also available via the `risk_scores` table, which is
optimised for fast range queries by coin and timestamp.

## Related APIs

- `GET /api/trading-signal`
- `GET /api/trading-strategies`
- `GET /api/trading-strategy/current`
- `POST /api/trading-strategy/set`
- `GET /api/signal-logs` — full scored-row audit trail
- `GET /api/v1/risk-scores` — slim paginated risk score feed for downstream systems

## Demo workflow

1. Start with balanced strategy.
2. Compare same window under aggressive strategy.
3. Explain threshold movement and resulting action density changes.
