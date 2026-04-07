# ============================================================
# risk_engine/risk_score.py
# Builder: maps the raw score_price_row dict → RiskScore schema.
#
# Usage (within 1A or any downstream Series 1 service):
#   from risk_engine.risk_score import build_risk_response
#   standardized = build_risk_response(raw_score_dict, price_data)
# ============================================================

from __future__ import annotations
from typing import Optional
from risk_engine.schemas import RiskScore, ModelBreakdown, FeatureSnapshot, ExplainResponse

_CONFIDENCE_MAP = {
    "High":   1.0,
    "Medium": 0.67,
    "Low":    0.33,
}


def build_risk_response(
    raw: dict,
    price_data: Optional[dict] = None,
) -> RiskScore:
    """
    Convert the dict returned by anomaly_detector.score_price_row()
    into the canonical RiskScore schema.

    Parameters
    ----------
    raw        : dict from score_price_row()
    price_data : original price fetch dict (adds open/high/low/vol_24h
                 when present; raw already has close & volume)
    """
    pd_extra = price_data or {}

    breakdown = ModelBreakdown(
        isolation_forest={
            "score":  raw.get("isolation_forest_score", 0.0),
            "reason": raw.get("if_reason", ""),
        },
        zscore={
            "score":  raw.get("zscore_score", 0.0),
            "sigma":  raw.get("zscore_value", 0.0),
            "reason": raw.get("zscore_reason", ""),
        },
        lstm={
            "score":           raw.get("lstm_score", 0.0),
            "predicted_price": raw.get("lstm_predicted_price"),
            "reason":          raw.get("lstm_reason", ""),
        },
    )

    snapshot = FeatureSnapshot(
        close          = raw.get("close_price") or pd_extra.get("close", 0.0),
        volume         = raw.get("volume") or pd_extra.get("volume", 0.0),
        open           = pd_extra.get("open"),
        high           = pd_extra.get("high"),
        low            = pd_extra.get("low"),
        volatility_24h = pd_extra.get("volatility_24h"),
    )

    confidence_label = raw.get("confidence_level", "Low")
    confidence_float = _CONFIDENCE_MAP.get(confidence_label, 0.33)

    return RiskScore(
        coin             = raw.get("coin", "BTC").upper(),
        risk_score       = round(float(raw.get("risk_score", 0.0)), 2),
        risk_level       = raw.get("risk_level", "Unknown"),
        confidence       = confidence_float,
        model_breakdown  = breakdown,
        feature_snapshot = snapshot,
        timestamp        = raw.get("timestamp", ""),
    )


def build_explain_response(
    raw: dict,
    price_data: Optional[dict] = None,
) -> ExplainResponse:
    """
    Convert the dict returned by score_price_row() into an ExplainResponse.

    model_breakdown  — flat scores keyed by model name
    reasoning        — flat human-readable strings keyed by model name
    ensemble_weights — dynamic weights used for this prediction

    Parameters
    ----------
    raw        : dict from score_price_row()
    price_data : original price fetch dict (unused here, kept for signature parity)
    """
    confidence_label = raw.get("confidence_level", "Low")
    confidence_float = _CONFIDENCE_MAP.get(confidence_label, 0.33)

    # ensemble_weights comes in as a dict {"isolation_forest": 0.25, ...}
    weights = raw.get("ensemble_weights") or {
        "isolation_forest": 0.25,
        "zscore":           0.25,
        "lstm":             0.50,
    }

    # Build the zscore reasoning string: include sigma when available
    zscore_reason = raw.get("zscore_reason", "")
    sigma = raw.get("zscore_value")
    if sigma is not None and zscore_reason:
        zscore_reason = f"{round(float(sigma), 2)} sigma above mean — {zscore_reason}"
    elif sigma is not None:
        zscore_reason = f"{round(float(sigma), 2)} sigma above mean"

    return ExplainResponse(
        coin          = raw.get("coin", "BTC").upper(),
        risk_score    = round(float(raw.get("risk_score", 0.0)), 2),
        risk_level    = raw.get("risk_level", "Unknown"),
        confidence    = confidence_float,
        model_breakdown = {
            "isolation_forest": round(float(raw.get("isolation_forest_score", 0.0)), 2),
            "zscore":           round(float(raw.get("zscore_score",           0.0)), 2),
            "lstm":             round(float(raw.get("lstm_score",             0.0)), 2),
        },
        reasoning = {
            "isolation_forest": raw.get("if_reason",    ""),
            "zscore":           zscore_reason,
            "lstm":             raw.get("lstm_reason",  ""),
        },
        ensemble_weights = weights,
        models_agreed    = int(raw.get("models_agreed", 0)),
        timestamp        = raw.get("timestamp", ""),
    )
