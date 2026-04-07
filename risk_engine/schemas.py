# ============================================================
# risk_engine/schemas.py
# Standardized output contract for Series 1 — Market Intelligence Systems.
#
# ALL downstream systems (1B Regime, 1C Volatility, 1D Contagion,
# 1E Master Risk, 1G Execution …) consume RiskScore as their
# canonical upstream input.  Never change field names without
# bumping schema_version.
# ============================================================

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ModelBreakdown(BaseModel):
    isolation_forest: dict = Field(default_factory=dict)
    zscore:           dict = Field(default_factory=dict)
    lstm:             dict = Field(default_factory=dict)


class FeatureSnapshot(BaseModel):
    close:     float
    volume:    float
    open:      Optional[float] = None
    high:      Optional[float] = None
    low:       Optional[float] = None
    volatility_24h: Optional[float] = None


class RiskScore(BaseModel):
    """
    Canonical risk assessment produced by 1A: Risk Engine Core.

    Fields
    ------
    coin            : Asset symbol, e.g. "BTC"
    risk_score      : Ensemble score 0-100 (higher = riskier)
    risk_level      : "Low" | "Medium" | "High" | "Critical"
    confidence      : 0.0-1.0  (derived from model agreement)
    model_breakdown : Per-model scores and reasons
    feature_snapshot: Raw feature values used for this prediction
    timestamp       : ISO-8601 UTC candle timestamp
    schema_version  : Bumped on breaking schema changes
    """
    model_config = ConfigDict(protected_namespaces=())

    coin:             str
    risk_score:       float = Field(ge=0, le=100)
    risk_level:       str
    confidence:       float = Field(ge=0.0, le=1.0)
    model_breakdown:  ModelBreakdown
    feature_snapshot: FeatureSnapshot
    timestamp:        str
    schema_version:   str = "1.0"


class ExplainResponse(BaseModel):
    """
    Step 6 — Explainability output.
    Answers "WHY is the risk score X?" in a human-readable, API-friendly form.

    Consumed by downstream systems 1E (Master Risk), 1F (Explainability Engine),
    1G (Execution).  Never change field names without bumping schema_version.

    Fields
    ------
    coin             : Asset symbol, e.g. "BTC"
    risk_score       : Ensemble score 0-100
    risk_level       : "Low" | "Medium" | "High" | "Critical"
    confidence       : 0.0-1.0
    model_breakdown  : Flat per-model scores  {"isolation_forest": 65, "zscore": 78, "lstm": 74}
    reasoning        : Flat per-model text    {"isolation_forest": "...", "zscore": "...", "lstm": "..."}
    ensemble_weights : Dynamic weights used   {"isolation_forest": 0.25, "zscore": 0.25, "lstm": 0.50}
    models_agreed    : Number of models that fired (0-3)
    timestamp        : ISO-8601 UTC candle timestamp
    schema_version   : Bumped on breaking schema changes
    """
    model_config = ConfigDict(protected_namespaces=())

    coin:             str
    risk_score:       float = Field(ge=0, le=100)
    risk_level:       str
    confidence:       float = Field(ge=0.0, le=1.0)
    model_breakdown:  dict
    reasoning:        dict
    ensemble_weights: dict
    models_agreed:    int
    timestamp:        str
    schema_version:   str = "1.0"
