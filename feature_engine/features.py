# ============================================================
# feature_engine/features.py
# Reusable technical feature computation for Series 1.
#
# Accepts any OHLCV DataFrame.  Returns the same DataFrame
# with additional computed columns — no DB calls, no side effects.
#
# Consumed by:
#   1A  — /api/features/{coin}  (live snapshot)
#   1B  — Regime Engine         (MA crossovers, trend)
#   1C  — Volatility Engine     (rolling vol, BBands)
#   1D  — Contagion Engine      (returns, correlations)
#   1G  — Execution Engine      (momentum, signals)
# ============================================================

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# CORE ENGINE
# ============================================================

def compute_features(
    df: pd.DataFrame,
    price_col:  str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Enrich an OHLCV DataFrame with reusable technical features.

    Input columns expected (all optional except `price_col`):
        open, high, low, close, volume

    Output columns added (all float64):
        returns            — 1-period pct change
        log_returns        — log(close_t / close_t-1)
        volatility_rolling — 20-period rolling std of returns
        ma_20              — 20-period simple moving average
        ma_50              — 50-period simple moving average
        ma_200             — 200-period simple moving average
        momentum_14        — 14-period rate-of-change (%)
        volume_ma_20       — 20-period volume moving average
        volume_spike       — z-score of volume (vs last 20 periods)
        bb_upper           — Bollinger upper band  (20-period, 2σ)
        bb_lower           — Bollinger lower band  (20-period, 2σ)
        bb_width           — Band width normalised by mid-band
        bb_pct_b           — %B: where price sits in the band (0=lower, 1=upper)
        high_low_range     — (high - low) / close   (if high/low present)
    """
    out = df.copy()
    p   = out[price_col]

    # ---- Returns ---------------------------------------------------
    out["returns"]     = p.pct_change()
    out["log_returns"] = np.log(p / p.shift(1))

    # ---- Volatility ------------------------------------------------
    out["volatility_rolling"] = out["returns"].rolling(window=20, min_periods=1).std()

    # ---- Moving Averages -------------------------------------------
    out["ma_20"]  = p.rolling(window=20,  min_periods=1).mean()
    out["ma_50"]  = p.rolling(window=50,  min_periods=1).mean()
    out["ma_200"] = p.rolling(window=200, min_periods=1).mean()

    # ---- Momentum (Rate of Change) ---------------------------------
    out["momentum_14"] = p.pct_change(periods=14) * 100

    # ---- Volume features (if volume column present) ----------------
    if volume_col in out.columns:
        v = out[volume_col]
        out["volume_ma_20"] = v.rolling(window=20, min_periods=1).mean()
        vol_std = v.rolling(window=20, min_periods=1).std().replace(0, np.nan)
        out["volume_spike"] = (v - out["volume_ma_20"]) / vol_std
        out["volume_spike"] = out["volume_spike"].fillna(0.0)
    else:
        out["volume_ma_20"] = np.nan
        out["volume_spike"] = 0.0

    # ---- Bollinger Bands -------------------------------------------
    rolling_mean = p.rolling(window=20, min_periods=1).mean()
    rolling_std  = p.rolling(window=20, min_periods=1).std().fillna(0)
    out["bb_upper"] = rolling_mean + 2 * rolling_std
    out["bb_lower"] = rolling_mean - 2 * rolling_std
    band_width      = out["bb_upper"] - out["bb_lower"]
    out["bb_width"]  = (band_width / rolling_mean).fillna(0)
    out["bb_pct_b"]  = ((p - out["bb_lower"]) / band_width.replace(0, np.nan)).fillna(0.5)

    # ---- High-Low Range (if available) -----------------------------
    if "high" in out.columns and "low" in out.columns:
        out["high_low_range"] = (out["high"] - out["low"]) / p.replace(0, np.nan)
    else:
        out["high_low_range"] = np.nan

    return out


# ============================================================
# SNAPSHOT  (latest row as a plain dict — for API responses)
# ============================================================

def get_feature_snapshot(df: pd.DataFrame) -> dict:
    """
    Compute features on df and return the LAST row as a dict.
    None values are used for columns that couldn't be computed.
    """
    if df is None or df.empty:
        return {}

    try:
        enriched = compute_features(df)
        row      = enriched.iloc[-1]

        def _val(col: str) -> Optional[float]:
            v = row.get(col, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(float(v), 6)

        return {
            "close":             _val("close"),
            "returns":           _val("returns"),
            "log_returns":       _val("log_returns"),
            "volatility_rolling":_val("volatility_rolling"),
            "ma_20":             _val("ma_20"),
            "ma_50":             _val("ma_50"),
            "ma_200":            _val("ma_200"),
            "momentum_14":       _val("momentum_14"),
            "volume":            _val("volume"),
            "volume_ma_20":      _val("volume_ma_20"),
            "volume_spike":      _val("volume_spike"),
            "bb_upper":          _val("bb_upper"),
            "bb_lower":          _val("bb_lower"),
            "bb_width":          _val("bb_width"),
            "bb_pct_b":          _val("bb_pct_b"),
            "high_low_range":    _val("high_low_range"),
        }
    except Exception as e:
        logger.warning(f"get_feature_snapshot failed: {e}")
        return {}
