# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# anomaly_detector.py -- ML ensemble, scoring, explainability
# ============================================================

import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from config import (
    MODELS_DIR,
    WEIGHT_ISOLATION_FOREST, WEIGHT_ZSCORE, WEIGHT_LSTM,
    RISK_LOW_MAX, RISK_MEDIUM_MAX, RISK_HIGH_MAX,
    ANOMALY_LOG_THRESHOLD,
    LSTM_SEQUENCE_LENGTH, LSTM_EPOCHS, LSTM_BATCH_SIZE,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_FEATURES
)
from data_layer.database import (
    get_training_data, insert_anomaly,
    register_model, log_event, get_model_val_metrics,
    insert_signal_log, insert_risk_score, insert_model_metric
)

logger = logging.getLogger(__name__)


# ================================================================
# LSTM MODEL ARCHITECTURE
# ----------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ----------------------------------------------------------------
# GLOBAL MODEL STATE  (keyed by coin, e.g. "BTC", "ETH")
# ----------------------------------------------------------------

_models_by_coin: dict[str, dict] = {}


def _get_coin_models(coin: str) -> dict:
    """Return (and lazily initialise) the model-state dict for a coin."""
    coin = coin.upper()
    if coin not in _models_by_coin:
        _models_by_coin[coin] = {
            "isolation_forest": None,
            "zscore_params":    None,
            "lstm":             None,
            "scaler":           None,
        }
    return _models_by_coin[coin]

_last_heartbeat_log_by_coin: dict[str, Optional[datetime]] = {}  # per-coin heartbeat timestamps


# ----------------------------------------------------------------
# RISK LEVEL HELPER
# ----------------------------------------------------------------

def get_risk_level(score: float) -> str:
    if score <= RISK_LOW_MAX:
        return "Low"
    elif score <= RISK_MEDIUM_MAX:
        return "Medium"
    elif score <= RISK_HIGH_MAX:
        return "High"
    else:
        return "Critical"


# ----------------------------------------------------------------
# ISOLATION FOREST
# ----------------------------------------------------------------

def _preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with only the features used by models, all numeric."""
    return df[LSTM_FEATURES].fillna(0).copy()


def _train_isolation_forest(df: pd.DataFrame) -> IsolationForest:
    model = IsolationForest(
        n_estimators  = 100,
        contamination = 0.05,
        random_state  = 42
    )
    model.fit(_preprocess_features(df))
    return model


def _score_isolation_forest(model: IsolationForest, row: pd.DataFrame) -> float:
    raw_score = model.decision_function(_preprocess_features(row))[0]
    clipped   = np.clip(raw_score, -0.5, 0.5)
    return float(np.interp(clipped, [-0.5, 0.5], [100, 0]))


def _if_reason(score: float) -> str:
    if score >= 75:
        return "Price pattern is far outside the normal trading cluster."
    elif score >= 50:
        return "Price pattern shows moderate deviation from normal cluster."
    else:
        return "Price pattern within expected range."


# ----------------------------------------------------------------
# Z-SCORE
# ----------------------------------------------------------------

def _compute_zscore_params(df: pd.DataFrame) -> dict:
    return {
        "mean": df["close"].mean(),
        "std":  df["close"].std()
    }


def _score_zscore(params: dict, close_price: float) -> tuple[float, float]:
    std   = params["std"] if params["std"] > 0 else 1.0
    z     = abs((close_price - params["mean"]) / std)
    return float(min(z / 4 * 100, 100)), round(z, 2)


def _zscore_reason(sigma: float, close_price: float, mean: float) -> str:
    direction = "above" if close_price > mean else "below"
    return f"{sigma} sigma deviation {direction} the historical mean (mean: ${mean:,.0f})."


# ----------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------

def _prepare_lstm_sequences(df: pd.DataFrame, scaler: MinMaxScaler):
    scaled = scaler.transform(_preprocess_features(df))
    X, y   = [], []
    for i in range(LSTM_SEQUENCE_LENGTH, len(scaled)):
        X.append(scaled[i - LSTM_SEQUENCE_LENGTH:i])
        y.append(scaled[i, LSTM_FEATURES.index("close")])
    return np.array(X), np.array(y)


def _train_lstm(df: pd.DataFrame, scaler: MinMaxScaler) -> LSTMModel:
    X, y  = _prepare_lstm_sequences(df, scaler)
    X_t   = torch.tensor(X, dtype=torch.float32)
    y_t   = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model     = LSTMModel(len(LSTM_FEATURES), LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(LSTM_EPOCHS):
        for i in range(0, len(X_t), LSTM_BATCH_SIZE):
            xb = X_t[i:i + LSTM_BATCH_SIZE]
            yb = y_t[i:i + LSTM_BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            logger.info(f"LSTM epoch {epoch + 1}/{LSTM_EPOCHS} - loss: {loss.item():.6f}")

    return model


def _compute_lstm_val_metrics(
    model: LSTMModel,
    scaler: MinMaxScaler,
    val_df: pd.DataFrame,
) -> dict:
    """Compute val_loss (MSE) and val_mae on the held-out validation split."""
    if len(val_df) < LSTM_SEQUENCE_LENGTH + 1:
        return {"val_loss": None, "val_mae": None}

    X, y = _prepare_lstm_sequences(val_df, scaler)
    if len(X) == 0:
        return {"val_loss": None, "val_mae": None}

    model.eval()
    close_idx = LSTM_FEATURES.index("close")
    criterion = nn.MSELoss()
    with torch.no_grad():
        X_t     = torch.tensor(X, dtype=torch.float32)
        y_t     = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        preds   = model(X_t)
        val_loss = criterion(preds, y_t).item()

        # Inverse-transform predictions and targets to real price space for MAE
        preds_np = preds.numpy().flatten()
        y_np     = y_t.numpy().flatten()
        mae_list = []
        for p_scaled, y_scaled in zip(preds_np, y_np):
            dummy_p = np.zeros((1, len(LSTM_FEATURES)))
            dummy_y = np.zeros((1, len(LSTM_FEATURES)))
            dummy_p[0, close_idx] = p_scaled
            dummy_y[0, close_idx] = y_scaled
            p_real = scaler.inverse_transform(dummy_p)[0, close_idx]
            y_real = scaler.inverse_transform(dummy_y)[0, close_idx]
            mae_list.append(abs(p_real - y_real))

    val_mae = float(np.mean(mae_list)) if mae_list else None
    return {"val_loss": float(val_loss), "val_mae": val_mae}


def _score_lstm(
    model: LSTMModel,
    scaler: MinMaxScaler,
    recent_df: pd.DataFrame,
    actual_close: float
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        scaled      = scaler.transform(_preprocess_features(recent_df))
        seq         = torch.tensor(scaled[-LSTM_SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(seq).item()

    dummy               = np.zeros((1, len(LSTM_FEATURES)))
    close_idx           = LSTM_FEATURES.index("close")
    dummy[0, close_idx] = pred_scaled
    predicted_price     = scaler.inverse_transform(dummy)[0, close_idx]

    pct_error = abs(actual_close - predicted_price) / max(actual_close, 1) * 100
    score     = float(min(pct_error * 5, 100))
    return score, round(predicted_price, 2)


def _lstm_reason(predicted: float, actual: float) -> str:
    diff      = actual - predicted
    direction = "dropped" if diff < 0 else "surged"
    pct       = abs(diff / max(predicted, 1) * 100)
    return (
        f"LSTM predicted ${predicted:,.0f} but actual was ${actual:,.0f} "
        f"({direction} {pct:.1f}%)."
    )


# ----------------------------------------------------------------
# CONFIDENCE LEVEL
# ----------------------------------------------------------------

def _confidence_level(models_agreed: int) -> str:
    if models_agreed == 3:
        return "High"
    elif models_agreed == 2:
        return "Medium"
    else:
        return "Low"


def _signal_strength_label(models_agreed: int, risk_score: float, signal_type: str) -> str:
    if signal_type == "synthetic":
        return "System Continuity Signal"
    if models_agreed >= 3 and risk_score >= 75:
        return "Very Strong Signal"
    if models_agreed >= 2:
        return "Good Signal"
    return "Watch Signal"


# ----------------------------------------------------------------
# PLAIN ENGLISH SUMMARY
# ----------------------------------------------------------------

def _plain_english_summary(
    risk_level: str,
    risk_score: float,
    close_price: float,
    zscore_sigma: float,
    lstm_predicted: float,
    models_agreed: int,
    confidence: str,
    coin: str = "BTC"
) -> str:
    price_change = abs(close_price - lstm_predicted)
    pct          = price_change / max(lstm_predicted, 1) * 100
    direction    = "drop" if close_price < lstm_predicted else "surge"

    return (
        f"{coin.upper()} showed a {risk_level.lower()} risk signal with a score of {risk_score:.0f}/100. "
        f"The price {direction} of {pct:.1f}% was flagged (actual: ${close_price:,.0f}, "
        f"expected: ${lstm_predicted:,.0f}). "
        f"Z-Score deviation was {zscore_sigma}σ from the mean. "
        f"{models_agreed}/3 models agreed on this anomaly (confidence: {confidence})."
    )


# ----------------------------------------------------------------
# TRAIN ALL MODELS
# ----------------------------------------------------------------

def train_models(coin: str = "BTC"):
    coin = coin.upper()
    logger.info(f"Starting model training for {coin}...")
    log_event("retrain", f"Model retraining started for {coin}.")

    rows = get_training_data(coin=coin)
    if len(rows) < LSTM_SEQUENCE_LENGTH + 10:
        logger.warning(f"{coin}: Not enough data to train. Skipping.")
        log_event("error", f"Training skipped for {coin} -- insufficient data.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df                = df.sort_values("timestamp").reset_index(drop=True)
    df[LSTM_FEATURES] = df[LSTM_FEATURES].fillna(0)

    # ---- Chronological 80/20 validation split ----
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    val_df    = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"{coin}: Train rows: {len(train_df)}  Val rows: {len(val_df)}")

    today  = datetime.utcnow().strftime("%Y-%m-%d")
    n_rows = len(df)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Isolation Forest  (train on train_df, measure anomaly rate on val_df)
    if_model     = _train_isolation_forest(train_df)
    if_path      = os.path.join(MODELS_DIR, f"isolation_forest_{coin}_{today}.pkl")
    with open(if_path, "wb") as f:
        pickle.dump(if_model, f)
    anomaly_rate = (
        (val_df.shape[0] - np.sum(if_model.predict(_preprocess_features(val_df)) == 1))
        / val_df.shape[0]
    ) if len(val_df) > 0 else 0.0
    register_model(f"isolation_forest_{coin}", if_path, n_rows, float(anomaly_rate))
    insert_model_metric(f"isolation_forest_{coin}", "anomaly_rate", float(anomaly_rate))
    logger.info(f"{coin}: Isolation Forest saved: {if_path} | val anomaly rate: {anomaly_rate:.3f}")

    # 2. Z-Score params  (computed on train_df only to avoid data leakage)
    zparams = _compute_zscore_params(train_df)
    zpath   = os.path.join(MODELS_DIR, f"zscore_params_{coin}_{today}.pkl")
    with open(zpath, "wb") as f:
        pickle.dump(zparams, f)
    # Val MAE for Z-Score: mean absolute z-sigma of val set (proxy metric)
    val_z_mae = None
    if len(val_df) > 0:
        z_std = zparams["std"] if zparams["std"] > 0 else 1.0
        val_z_mae = float(abs((val_df["close"] - zparams["mean"]) / z_std).mean())
    register_model(f"zscore_{coin}", zpath, n_rows, 0.0, val_mae=val_z_mae)
    if val_z_mae is not None:
        insert_model_metric(f"zscore_{coin}", "val_mae", val_z_mae)
    logger.info(f"{coin}: Z-Score params saved: {zpath} | val MAE (sigma): {val_z_mae}")

    # 3. Scaler + LSTM  (fit scaler on train_df, validate on val_df)
    scaler = MinMaxScaler()
    scaler.fit(_preprocess_features(train_df))
    lstm_model = _train_lstm(train_df, scaler)
    val_metrics = _compute_lstm_val_metrics(lstm_model, scaler, val_df)
    logger.info(f"{coin}: LSTM val_loss: {val_metrics['val_loss']}  val_mae: {val_metrics['val_mae']}")

    lstm_path  = os.path.join(MODELS_DIR, f"lstm_{coin}_{today}.pt")
    torch.save(lstm_model.state_dict(), lstm_path)
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{coin}_{today}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    register_model(
        f"lstm_{coin}", lstm_path, n_rows, 0.0,
        val_loss=val_metrics["val_loss"],
        val_mae=val_metrics["val_mae"],
    )
    if val_metrics["val_loss"] is not None:
        insert_model_metric(f"lstm_{coin}", "val_loss", val_metrics["val_loss"])
    if val_metrics["val_mae"] is not None:
        insert_model_metric(f"lstm_{coin}", "val_mae",  val_metrics["val_mae"])
    logger.info(f"{coin}: LSTM saved: {lstm_path}")

    m = _get_coin_models(coin)
    m["isolation_forest"] = if_model
    m["zscore_params"]    = zparams
    m["lstm"]             = lstm_model
    m["scaler"]           = scaler

    log_event("retrain", f"{coin}: Retraining complete. Trained on {n_rows} rows.", {"rows": n_rows, "coin": coin})
    logger.info(f"{coin}: Model training complete.")


# ----------------------------------------------------------------
# LOAD MODELS FROM DISK
# ----------------------------------------------------------------

def load_models(coin: str = "BTC"):
    from data_layer.database import get_latest_models
    coin = coin.upper()
    latest = get_latest_models(coin=coin)

    m = _get_coin_models(coin)
    try:
        if "isolation_forest" in latest:
            with open(latest["isolation_forest"], "rb") as f:
                m["isolation_forest"] = pickle.load(f)

        if "zscore" in latest:
            with open(latest["zscore"], "rb") as f:
                m["zscore_params"] = pickle.load(f)

        if "lstm" in latest:
            scaler_path = latest["lstm"].replace(f"lstm_{coin}_", f"scaler_{coin}_").replace(".pt", ".pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    m["scaler"] = pickle.load(f)
            else:
                logger.warning(f"{coin}: Scaler file not found: {scaler_path}. LSTM cannot run without it — will retrain.")
            model = LSTMModel(len(LSTM_FEATURES), LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS)
            model.load_state_dict(torch.load(latest["lstm"], map_location="cpu"))
            model.eval()
            m["lstm"] = model

        if models_ready(coin):
            logger.info(f"{coin}: Models loaded from disk successfully.")
        else:
            logger.warning(f"{coin}: Models loaded from disk but one or more components are missing — retraining will be triggered.")
    except Exception as e:
        logger.error(f"{coin}: Failed to load models from disk: {e}")


def models_ready(coin: str = "BTC") -> bool:
    return all(v is not None for v in _get_coin_models(coin.upper()).values())


# ----------------------------------------------------------------
# DYNAMIC ENSEMBLE WEIGHTS
# ----------------------------------------------------------------

def get_ensemble_weights(coin: str = "BTC") -> dict:
    """
    Compute ensemble weights from stored val_mae for a specific coin.
    Lower val_mae → higher weight (inverse-error weighting).
    Falls back to static config weights when metrics unavailable.
    """
    try:
        metrics = get_model_val_metrics(coin=coin.upper())
        mae_if  = metrics.get("isolation_forest")
        mae_z   = metrics.get("zscore")
        mae_lstm= metrics.get("lstm")

        # If any val_mae is available, use inverse-error weighting.
        # Models without a mae (IF, Z-Score have no continuous mae)
        # keep their static config weights re-normalised proportionally.
        if mae_lstm is not None and mae_lstm > 0:
            # Use reciprocal of mae as raw weight for LSTM
            # IF and Z-Score retain their config ratio relative to each other
            static_if_z_total = WEIGHT_ISOLATION_FOREST + WEIGHT_ZSCORE
            lstm_raw  = 1.0 / mae_lstm
            # Map lstm_raw to [0.3, 0.7] to avoid degenerate weights
            lstm_w    = float(np.clip(lstm_raw / (lstm_raw + 1.0), 0.30, 0.70))
            remainder = 1.0 - lstm_w
            ratio     = WEIGHT_ISOLATION_FOREST / max(static_if_z_total, 1e-9)
            return {
                "isolation_forest": round(remainder * ratio, 4),
                "zscore":           round(remainder * (1 - ratio), 4),
                "lstm":             round(lstm_w, 4),
            }
    except Exception:
        pass  # Fall back silently

    return {
        "isolation_forest": WEIGHT_ISOLATION_FOREST,
        "zscore":           WEIGHT_ZSCORE,
        "lstm":             WEIGHT_LSTM,
    }


# ----------------------------------------------------------------
# MAIN SCORING ENTRY POINT
# ----------------------------------------------------------------

def score_price_row(price_data: dict, recent_df: pd.DataFrame) -> Optional[dict]:
    coin = price_data.get("coin", "BTC").upper()
    if not models_ready(coin):
        logger.warning(f"{coin}: Models not ready. Skipping scoring.")
        return None

    close_price = price_data["close"]
    m = _get_coin_models(coin)

    row_df    = _preprocess_features(pd.DataFrame([price_data]))
    if_score  = _score_isolation_forest(m["isolation_forest"], row_df)
    if_reason = _if_reason(if_score)

    zs_score, sigma = _score_zscore(m["zscore_params"], close_price)
    zs_reason       = _zscore_reason(sigma, close_price, m["zscore_params"]["mean"])

    lstm_score, lstm_pred = _score_lstm(
        m["lstm"], m["scaler"], recent_df, close_price
    )
    lstm_reason = _lstm_reason(lstm_pred, close_price)

    weights    = get_ensemble_weights(coin=coin)
    risk_score = (
        if_score   * weights["isolation_forest"] +
        zs_score   * weights["zscore"] +
        lstm_score * weights["lstm"]
    )
    risk_level = get_risk_level(risk_score)

    threshold  = 50.0
    agreed     = sum([if_score >= threshold, zs_score >= threshold, lstm_score >= threshold])
    confidence = _confidence_level(agreed)
    contributing_models = []
    if if_score >= threshold:
        contributing_models.append("Isolation Forest")
    if zs_score >= threshold:
        contributing_models.append("Z-Score")
    if lstm_score >= threshold:
        contributing_models.append("LSTM")

    summary    = _plain_english_summary(
        risk_level, risk_score, close_price,
        sigma, lstm_pred, agreed, confidence,
        coin=price_data.get("coin", "BTC")
    )

    result = {
        "timestamp":               price_data["timestamp"],
        "close_price":             close_price,
        "volume":                  price_data["volume"],
        "risk_score":              round(risk_score, 2),
        "risk_level":              risk_level,
        "isolation_forest_score":  round(if_score, 2),
        "zscore_score":            round(zs_score, 2),
        "lstm_score":              round(lstm_score, 2),
        "if_reason":               if_reason,
        "zscore_value":            sigma,
        "zscore_reason":           zs_reason,
        "lstm_predicted_price":    lstm_pred,
        "lstm_reason":             lstm_reason,
        "models_agreed":           agreed,
        "confidence_level":        confidence,
        "plain_english_summary":   summary,
        "signal_type":             "real",
        "contributing_models":     ", ".join(contributing_models),
        "signal_strength":         _signal_strength_label(agreed, risk_score, "real"),
        "coin":                    price_data.get("coin", "BTC"),
        "ensemble_weights":        weights,
    }

    # ---- STEP 5: log every scored row to signal_logs ----
    try:
        from service_layer.trading_signals import generate_trading_signal
        sig = generate_trading_signal(risk_score)
        result["signal"]   = sig["signal"]
        result["strategy"] = sig["strategy"]
    except Exception as _sig_err:
        logger.debug(f"Signal generation skipped: {_sig_err}")
        result.setdefault("signal",   "HOLD")
        result.setdefault("strategy", None)
    try:
        insert_signal_log(result)
    except Exception as _log_err:
        logger.warning(f"Signal log insert failed: {_log_err}")
    try:
        insert_risk_score(result)
    except Exception as _rs_err:
        logger.warning(f"Risk score insert failed: {_rs_err}")

    coin_key = price_data.get("coin", "BTC").upper()
    global _last_heartbeat_log_by_coin
    now_utc = datetime.utcnow()
    last_hb = _last_heartbeat_log_by_coin.get(coin_key)
    hourly_due = (
        last_hb is None or
        (now_utc - last_hb).total_seconds() >= 3600
    )

    if risk_score >= ANOMALY_LOG_THRESHOLD:
        insert_anomaly(result)
        _last_heartbeat_log_by_coin[coin_key] = now_utc
        logger.info(f"Anomaly logged: score={risk_score:.1f} level={risk_level}")

        # Send alerts for critical anomalies
        try:
            from alerts import send_critical_alert
            alert_count = send_critical_alert(risk_score, risk_level, result)
            if alert_count > 0:
                logger.info(f"Sent {alert_count} alert(s) for critical anomaly.")
        except ImportError:
            logger.debug("Alerts module not available.")
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    elif hourly_due:
        synthetic = dict(result)
        synthetic["signal_type"] = "synthetic"
        synthetic["confidence_level"] = "Low"
        synthetic["signal_strength"] = _signal_strength_label(agreed, risk_score, "synthetic")
        synthetic["plain_english_summary"] = (
            f"No real anomaly detected for the last hour. Logged continuity signal at "
            f"risk {risk_score:.0f}/100 for system monitoring."
        )
        insert_anomaly(synthetic)
        _last_heartbeat_log_by_coin[coin_key] = now_utc
        logger.info(f"Continuity signal logged: score={risk_score:.1f} level={risk_level} (no real anomaly in last hour)")

    return result


# ----------------------------------------------------------------
# LSTM ITERATIVE PRICE FORECAST
# ----------------------------------------------------------------

def forecast_prices(steps: int = 12, coin: str = "BTC") -> list:
    """
    Iteratively roll the LSTM forward `steps` times to produce a price forecast.
    Each prediction is fed back into the sliding window as the next input.
    Returns a list of {timestamp, price} dicts representing the next N hours.
    """
    coin = coin.upper()
    if not models_ready(coin):
        return []

    from data_layer.database import get_price_history
    from datetime import timedelta as _td

    rows = get_price_history(limit=LSTM_SEQUENCE_LENGTH + 10, coin=coin)
    if len(rows) < LSTM_SEQUENCE_LENGTH:
        return []

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.sort_values("timestamp").reset_index(drop=True)

    m         = _get_coin_models(coin)
    model     = m["lstm"]
    scaler    = m["scaler"]
    close_idx = LSTM_FEATURES.index("close")

    model.eval()
    scaled = scaler.transform(_preprocess_features(df))
    window = list(scaled[-LSTM_SEQUENCE_LENGTH:])   # mutable sliding window

    last_ts     = pd.to_datetime(df["timestamp"].iloc[-1])
    predictions = []

    with torch.no_grad():
        for step in range(steps):
            seq         = torch.tensor([window[-LSTM_SEQUENCE_LENGTH:]], dtype=torch.float32)
            pred_scaled = model(seq).item()

            dummy               = np.zeros((1, len(LSTM_FEATURES)))
            dummy[0, close_idx] = pred_scaled
            price               = float(scaler.inverse_transform(dummy)[0, close_idx])

            ts = last_ts + _td(hours=step + 1)
            predictions.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "price":     round(price, 2),
            })

            # Slide window: copy last row, update close to predicted value
            new_row              = window[-1].copy()
            new_row[close_idx]   = pred_scaled
            window.append(new_row)

    return predictions