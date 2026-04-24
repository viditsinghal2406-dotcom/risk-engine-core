# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# api_backend.py  -- Single-service FastAPI app (port $PORT)
# ============================================================

import atexit
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from config import (
    DEFAULT_PAGE, DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, CHART_RANGE_CONFIG,
    LSTM_SEQUENCE_LENGTH, RETRAIN_HOUR, RETRAIN_MINUTE, PURGE_HOUR, PURGE_MINUTE,
    POLLING_INTERVAL_SECONDS, COIN_CONFIG, LOGS_DIR, LOG_FILE,
    setup_logging,
)
from data_layer.database import (
    init_db,
    get_latest_price, get_price_range, get_price_history,
    get_anomalies, get_all_anomalies, get_anomaly_by_id, get_anomalies_in_range, get_continuity_signals,
    get_model_registry, get_recent_events,
    count_price_rows, export_anomalies_to_csv,
    export_price_data_to_csv, get_model_performance_stats,
    get_anomaly_distribution, get_24h_high_low, get_price_24h_ago,
    get_signal_backtest, clear_anomaly_logs,
    purge_old_price_data, purge_old_models, log_event,
    get_signal_logs,
    get_risk_scores, get_model_metrics,
)
from data_layer.data_pipeline import (
    fetch_latest_price, clear_price_cache,
    seed_database, needs_seeding, seed_recent_hourly, needs_hourly_seed,
    backfill_missing_candles, needs_deep_history, fetch_historical_candles,
)
from model_layer.anomaly_detector import (
    score_price_row, models_ready, get_risk_level, forecast_prices,
    train_models, load_models,
)
from service_layer.trading_signals import generate_trading_signal
from risk_engine.risk_score import build_risk_response, build_explain_response
from feature_engine.features import get_feature_snapshot

# ----------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------

setup_logging()
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------
# SCHEDULER JOBS
# ----------------------------------------------------------------

_scheduler:      BackgroundScheduler | None = None
_last_scored_ts: dict                       = {}   # last candle timestamp per coin


def _retrain_job():
    logger.info("Scheduled retrain starting…")
    for coin in COIN_CONFIG:
        train_models(coin)
    purge_old_models(keep_last=7)
    log_event("retrain", "Scheduled nightly retrain complete.")


def _purge_job():
    logger.info("Scheduled purge starting…")
    purge_old_price_data()


def _price_fetch_job():
    global _last_scored_ts
    for coin in COIN_CONFIG:
        try:
            data = fetch_latest_price(coin)
            if data:
                logger.debug(f"{coin} fetched: {data['close']} via {data['source']}")
                if models_ready(coin):
                    ts = data.get("timestamp", "")
                    if ts != _last_scored_ts.get(coin):
                        rows = get_price_history(limit=200, coin=coin)
                        if rows:
                            recent_df = pd.DataFrame([dict(r) for r in rows])
                            score_price_row(data, recent_df)
                            _last_scored_ts[coin] = ts
        except Exception as exc:
            logger.error(f"Price fetch error ({coin}): {exc}")


def _startup():
    logger.info("=" * 60)
    logger.info("Crypto Risk Intelligence System starting up…")
    logger.info("=" * 60)

    init_db()
    logger.info("Database ready.")

    for coin in COIN_CONFIG:
        if needs_seeding(coin):
            logger.info(f"{coin}: no data found — seeding historical prices…")
            seed_database(coin=coin)
        else:
            logger.info(f"{coin}: seed present.")

        if needs_deep_history(coin=coin, min_days=700):
            logger.info(f"{coin}: seeding 2yr daily data…")
            from data_layer.data_pipeline import _compute_and_attach_volatility
            from data_layer.database import insert_price_rows_bulk
            rows = fetch_historical_candles(interval="1d", days=730, coin=coin)
            if rows:
                rows = _compute_and_attach_volatility(rows)
                insert_price_rows_bulk(rows)
                logger.info(f"{coin}: 2yr seed done ({len(rows)} rows).")
        else:
            logger.info(f"{coin}: sufficient history present.")

        if needs_hourly_seed(coin=coin):
            logger.info(f"{coin}: seeding 30d hourly candles…")
            seed_recent_hourly(days=30, coin=coin)
        else:
            logger.info(f"{coin}: hourly data present.")

        backfill_missing_candles(coin)

    for coin in COIN_CONFIG:
        if models_ready(coin):
            logger.info(f"{coin}: Models already in memory.")
        else:
            load_models(coin)
            if not models_ready(coin):
                logger.info(f"{coin}: Models not ready after load — training from scratch…")
                train_models(coin)

    global _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_retrain_job,     "cron",     hour=RETRAIN_HOUR,   minute=RETRAIN_MINUTE)
    _scheduler.add_job(_purge_job,       "cron",     hour=PURGE_HOUR,     minute=PURGE_MINUTE)
    _scheduler.add_job(_price_fetch_job, "interval", seconds=POLLING_INTERVAL_SECONDS)
    _scheduler.start()
    atexit.register(lambda: _scheduler.shutdown(wait=False))
    logger.info(f"Scheduler running. Price fetch every {POLLING_INTERVAL_SECONDS}s.")
    logger.info("=" * 60)


# ----------------------------------------------------------------
# LIFESPAN
# ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_startup, daemon=True, name="startup")
    thread.start()
    yield
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)


# ----------------------------------------------------------------
# APP
# ----------------------------------------------------------------

# Track last score per coin to prevent duplicate logging and cache results
_last_scored_timestamp: dict = {}  # {coin: timestamp}
_last_score_result: dict     = {}  # {coin: result_dict}

app = FastAPI(
    title       = "Crypto Risk Intelligence API",
    description = "BTC market risk scoring through the Risk Engine and Decision Layer",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(_BASE_DIR, "static")),
    name="static",
)


def _derive_lstm_trend(risk: dict | None) -> str:
    if not risk:
        return "Flat"
    actual = risk.get("close_price")
    predicted = risk.get("lstm_predicted_price")
    if actual is None or predicted is None:
        return "Flat"
    if actual > predicted:
        return "Upward"
    if actual < predicted:
        return "Downward"
    return "Flat"


def _build_market_insight(risk: dict | None, recent_df: pd.DataFrame, anomaly_density_pct: float) -> dict:
    risk_score = float((risk or {}).get("risk_score") or 0.0)
    volatility_pct = 0.0
    if not recent_df.empty and "close" in recent_df.columns and len(recent_df) > 2:
        returns = recent_df["close"].pct_change().dropna()
        if not returns.empty:
            volatility_pct = float(returns.std() * 100)

    trend = _derive_lstm_trend(risk)
    density_label = "elevated" if anomaly_density_pct >= 6 else "contained"
    vol_label = "high" if volatility_pct >= 1.5 else "steady"

    headline = (
        f"Risk Engine score is {risk_score:.0f}/100 with {density_label} anomaly density "
        f"({anomaly_density_pct:.1f}% over the active window)."
    )
    detail = (
        f"Volatility is {vol_label} ({volatility_pct:.2f}% return sigma) and LSTM trend direction is {trend.lower()}, "
        f"so the Decision Layer is tuned for {'defensive' if risk_score >= 61 else 'measured'} posture."
    )

    return {
        "headline": headline,
        "detail": detail,
        "risk_score": round(risk_score, 2),
        "anomaly_density_pct": round(anomaly_density_pct, 2),
        "volatility_pct": round(volatility_pct, 4),
        "lstm_trend": trend,
    }


# ----------------------------------------------------------------
# DASHBOARD  (serves the React-less SPA template)
# ----------------------------------------------------------------

@app.get("/")
def dashboard(request: Request):
    return FileResponse(os.path.join(_BASE_DIR, "templates", "index.html"))


@app.head("/")
def dashboard_head():
    return Response(status_code=200)


# ----------------------------------------------------------------
# SEEDING STATUS  (polled by the frontend overlay)
# ----------------------------------------------------------------

@app.get("/api/status")
def seeding_status():
    return {
        "seeding_complete": all(not needs_seeding(c) for c in COIN_CONFIG),
        "models_ready":     {c: models_ready(c) for c in COIN_CONFIG},
    }


# ----------------------------------------------------------------
# HEALTH
# ----------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "models_ready": {c: models_ready(c) for c in COIN_CONFIG},
        "price_rows":   count_price_rows(),
        "timestamp":    datetime.utcnow().isoformat()
    }


# ----------------------------------------------------------------
# ADMIN -- RETRAIN
# ----------------------------------------------------------------

@app.post("/api/admin/retrain")
def admin_retrain():
    """Force an immediate model retrain. Use when features or data have changed."""
    import threading
    from model_layer.anomaly_detector import train_models as _train
    def _retrain_all():
        for c in COIN_CONFIG:
            _train(c)
    t = threading.Thread(target=_retrain_all, daemon=True)
    t.start()
    return {"status": "retraining started", "coins": list(COIN_CONFIG.keys())}


# ----------------------------------------------------------------
# LIVE PRICE
# ----------------------------------------------------------------

@app.get("/api/price/live")
def live_price(force_refresh: bool = Query(default=False), coin: str = Query(default="BTC")):
    global _last_scored_timestamp, _last_score_result

    coin = coin.upper()
    if force_refresh:
        clear_price_cache(coin)

    data = fetch_latest_price(coin)
    if data is None:
        raise HTTPException(status_code=503, detail="Price data unavailable.")

    score_result = None
    rows = get_price_history(limit=200, coin=coin)
    recent_df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    # Score all coins when their models are ready
    if models_ready(coin) and data.get("timestamp") != _last_scored_timestamp.get(coin):
        if not recent_df.empty and len(recent_df) >= LSTM_SEQUENCE_LENGTH:
            score_result                   = score_price_row(data, recent_df)
            _last_scored_timestamp[coin]   = data.get("timestamp")
            _last_score_result[coin]       = score_result
    elif models_ready(coin):
        score_result = _last_score_result.get(coin)

    # Get ACTUAL 24-hour high/low from database (not just current candle)
    high_low_24h = get_24h_high_low(coin=coin)
    if high_low_24h["high"] > 0:
        data["high"] = high_low_24h["high"]
    if high_low_24h["low"] > 0:
        data["low"] = high_low_24h["low"]
    # For the displayed volume stat, prefer the actual 24h total from the live
    # API response (volume_24h). Only fall back to the DB value if it isn't set.
    if "volume_24h" in data:
        data["volume"] = data["volume_24h"]
    elif high_low_24h["volume"] > 0:
        data["volume"] = high_low_24h["volume"]

    if data.get("change_24h_pct") is None:
        price_24h_ago = get_price_24h_ago(coin=coin)
        current_close = data.get("close", 0)
        if price_24h_ago and price_24h_ago > 0 and current_close:
            data["change_24h_pct"] = round((current_close - price_24h_ago) / price_24h_ago * 100, 2)
    if data.get("change_24h_pct") is not None:
        data["change_24h_pct"] = round(float(data["change_24h_pct"]), 2)

    perf_stats = get_model_performance_stats(days=1)
    anomaly_density = float(perf_stats.get("anomaly_rate", 0.0) or 0.0)
    insight = _build_market_insight(score_result, recent_df, anomaly_density)

    risk_standard = build_risk_response(score_result, data).model_dump() if score_result else None

    return {
        "price":        data,
        "risk":         score_result,
        "risk_standard": risk_standard,
        "insight":      insight,
        "source":       data.get("source", "unknown"),
        "fetched_at":   datetime.utcnow().isoformat()
    }


# ----------------------------------------------------------------
# CHART DATA
# ----------------------------------------------------------------

@app.get("/api/chart")
def chart_data(range: str = Query(default="1D"), strategy: str = Query(default=None), coin: str = Query(default="BTC")):
    """
    Returns OHLCV candles for the given range.
    Each range button sets both candle size and the look-back window:
      1H  = 1-hour candles, last 7 days
      1D  = daily candles,  last 90 days
      1W  = weekly candles, last year
      1M  = monthly candles, last 2 years
      All = daily candles, all available data

    When the available data is too sparse for the requested frequency (< 5
    candles after resampling), the endpoint automatically steps down to a
    finer frequency so we never return a single giant candlestick.
    """
    coin  = coin.upper()
    cfg   = CHART_RANGE_CONFIG.get(range.upper(), CHART_RANGE_CONFIG["1D"])
    now   = datetime.utcnow()
    start = now - timedelta(days=cfg["days"])

    rows = get_price_range(start, now, coin=coin)

    # For ALL range (or when empty) try to use every row in the DB
    if not rows or range.upper() == "ALL":
        rows = get_price_range(datetime(2010, 1, 1), now, coin=coin)

    if not rows:
        return {"range": range, "count": 0, "candles": []}

    # Convert to DataFrame for resampling
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Frequency fallback chain — step down if resampled result is too sparse
    FREQ_FALLBACK_CHAIN = {
        "1ME": ["1W", "1D", "1h"],
        "1M":  ["1W", "1D", "1h"],
        "1W":  ["1D", "1h"],
        "1D":  ["1h"],
        "1h":  [],
    }

    freq = cfg["freq"]

    def _resample(df, freq):
        try:
            return df.resample(freq).agg(
                open   = ("open",   "first"),
                high   = ("high",   "max"),
                low    = ("low",    "min"),
                close  = ("close",  "last"),
                volume = ("volume", "sum"),
            ).dropna().reset_index()
        except ValueError:
            # Handle pandas version alias differences (1M vs 1ME)
            alt = None
            if freq.endswith("ME"):
                alt = freq[:-2] + "M"
            elif freq == "1M":
                alt = "1ME"
            if alt:
                return df.resample(alt).agg(
                    open   = ("open",   "first"),
                    high   = ("high",   "max"),
                    low    = ("low",    "min"),
                    close  = ("close",  "last"),
                    volume = ("volume", "sum"),
                ).dropna().reset_index()
            raise

    ohlcv = _resample(df, freq)

    # Step down frequency only if we have fewer than 2 candles — avoids 1M falling
    # back to 1W just because there are only 3 months of data in the DB
    for fallback in FREQ_FALLBACK_CHAIN.get(freq.upper(), FREQ_FALLBACK_CHAIN.get(freq, [])):
        if len(ohlcv) >= 2:
            break
        logger.warning(f"Chart range {range}: only {len(ohlcv)} candles at {freq}, stepping down to {fallback}.")
        ohlcv = _resample(df, fallback)
        freq  = fallback

    candles = [
        {
            "timestamp": row["timestamp"].isoformat(),
            "open":      row["open"],
            "high":      row["high"],
            "low":       row["low"],
            "close":     row["close"],
            "volume":    row["volume"],
        }
        for _, row in ohlcv.iterrows()
    ]

    # Fetch anomaly/heartbeat rows for this date range
    anomaly_rows = get_anomalies_in_range(start, now, coin=coin)
    anomalies    = []
    signals      = []

    for r in anomaly_rows:
        rd = dict(r)
        anomalies.append({
            "timestamp":  rd["timestamp"],
            "price":      rd["close_price"],
            "risk_level": rd["risk_level"],
            "score":      rd["risk_score"],
        })
        # Compute trading signal for this row using the requested strategy
        sig = generate_trading_signal(rd["risk_score"], strategy=strategy)
        signals.append({
            "timestamp":  rd["timestamp"],
            "price":      rd["close_price"],
            "signal":     sig["signal"],
            "confidence": sig["confidence"],
            "risk_score": rd["risk_score"],
            "strategy":   sig["strategy"],
        })

    return {
        "range":     range,
        "count":     len(candles),
        "candles":   candles,
        "anomalies": anomalies,
        "signals":   signals,
    }


# ----------------------------------------------------------------
# ANOMALIES (paginated)
# ----------------------------------------------------------------

@app.get("/api/anomalies")
def list_anomalies(
    page:              int  = Query(default=DEFAULT_PAGE,       ge=1),
    limit:             int  = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    coin:              str  = Query(default="BTC"),
    include_synthetic: bool = Query(default=False),
):
    if include_synthetic:
        rows, total = get_all_anomalies(page=page, limit=limit, coin=coin.upper())
    else:
        rows, total = get_anomalies(page=page, limit=limit, coin=coin.upper())
    return {
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": -(-total // limit),
        "anomalies":   [dict(r) for r in rows]
    }


@app.delete("/api/anomalies/clear")
def delete_anomaly_logs(coin: str = Query(default="BTC")):
    """Permanently delete all anomaly logs (real + synthetic) for a coin."""
    deleted = clear_anomaly_logs(coin=coin.upper())
    return {"deleted": deleted, "coin": coin.upper(), "message": f"Cleared {deleted} anomaly log entries."}


@app.get("/api/continuity-signals")
def list_continuity_signals(
    page: int = Query(default=DEFAULT_PAGE, ge=1),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    coin: str = Query(default="BTC")
):
    rows, total = get_continuity_signals(page=page, limit=limit, coin=coin.upper())
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": -(-total // limit),
        "signals": [dict(r) for r in rows],
        "label": "System Continuity Signals",
    }


@app.get("/api/signal-logs")
def list_signal_logs(
    page: int = Query(default=DEFAULT_PAGE, ge=1),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    coin: str = Query(default="BTC"),
):
    """Return every scored price row (all signals, not just anomalies)."""
    rows, total = get_signal_logs(coin=coin.upper(), page=page, limit=limit)
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": -(-total // limit),
        "logs": [dict(r) for r in rows],
    }


@app.get("/api/v1/risk-scores")
def list_risk_scores(
    page: int = Query(default=DEFAULT_PAGE, ge=1),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    coin: str = Query(default="BTC"),
):
    """
    Step 7 — slim historical risk score feed.
    Returns paginated coin+timestamp+risk_score+risk_level+confidence rows.
    Designed for fast querying by downstream 1E/1G systems.
    """
    rows, total = get_risk_scores(coin=coin.upper(), page=page, limit=limit)
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": -(-total // limit),
        "coin": coin.upper(),
        "scores": [dict(r) for r in rows],
    }


@app.get("/api/v1/model-metrics")
def list_model_metrics(
    model: str = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Step 7 — model performance metric history.
    Optional ?model=lstm to filter to one model.
    """
    rows = get_model_metrics(model_name=model, limit=limit)
    return {
        "model": model,
        "count": len(rows),
        "metrics": [dict(r) for r in rows],
    }


@app.get("/api/anomalies/{anomaly_id}")
def get_anomaly(anomaly_id: int):
    row = get_anomaly_by_id(anomaly_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Anomaly not found.")
    return dict(row)


# ----------------------------------------------------------------
# MODEL INTELLIGENCE
# ----------------------------------------------------------------

@app.get("/api/models")
@app.get("/api/model-intelligence")
def model_intelligence():
    registry = get_model_registry()
    events   = get_recent_events(limit=50)
    
    # Get CURRENT anomaly rate (not just training-time rate)
    perf_stats = get_model_performance_stats(days=7)
    
    return {
        "registry": [dict(r) for r in registry],
        "events":   [dict(e) for e in events],
        "current_anomaly_rate": perf_stats.get("anomaly_rate", 0),  # Current rate
        "total_predictions": perf_stats.get("total_predictions", 0),
    }


# ----------------------------------------------------------------
# DECISION LAYER
# ----------------------------------------------------------------

@app.get("/api/trading-signal")
def trading_signal(strategy: str = Query(default=None), coin: str = Query(default="BTC")):
    """Get buy/sell/hold signal using the cached risk score — no redundant ML inference."""
    coin = coin.upper()
    # Use live cached price for accuracy (same source as the header ticker)
    live_data = fetch_latest_price(coin)
    if not live_data:
        raise HTTPException(status_code=503, detail="Price data unavailable.")

    current_price = live_data.get("close")
    # Use the score computed by the last /api/price/live call; default to HOLD (50) if not yet scored
    cached = _last_score_result.get(coin)
    risk_score = cached["risk_score"] if cached else 50

    signal = generate_trading_signal(risk_score, strategy=strategy)
    return {
        "signal":        signal,
        "current_price": current_price,
        "risk_score":    risk_score,
    }


@app.get("/api/trading-strategies")
def get_trading_strategies():
    """Get all available trading strategy profiles."""
    from service_layer.trading_signals import get_available_strategies
    strategies = get_available_strategies()
    
    # Format strategies for frontend
    formatted = []
    for key, strat in strategies.items():
        formatted.append({
            "id": key,
            "name": strat["name"],
            "description": strat["description"],
            "emoji": strat["emoji"],
            "risk_level": strat["risk_level"],
            "best_for": strat["best_for"],
            "parameters": {
                "buy_threshold": strat["buy_threshold"],
                "sell_threshold": strat["sell_threshold"]
            }
        })
    
    return {"strategies": formatted}


@app.post("/api/trading-strategy/set")
def set_trading_strategy(strategy: str = Query(...)):
    """Set the current trading strategy."""
    from service_layer.trading_signals import get_available_strategies
    
    strategies = get_available_strategies()
    if strategy not in strategies:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
    
    # Update global config
    import config
    config.CURRENT_TRADING_STRATEGY = strategy
    
    # Update the old config variables for backward compatibility
    strat = strategies[strategy]
    config.SIGNAL_BUY_THRESHOLD = strat["buy_threshold"]
    config.SIGNAL_SELL_THRESHOLD = strat["sell_threshold"]
    config.SIGNAL_HOLD_RANGE = (strat["buy_threshold"] + 1, strat["sell_threshold"] - 1)
    
    return {
        "status": "success",
        "message": f"Trading strategy changed to {strat['name']}",
        "current_strategy": strategy,
        "parameters": strat
    }


@app.get("/api/trading-strategy/current")
def get_current_trading_strategy():
    """Get the currently active trading strategy."""
    from service_layer.trading_signals import get_strategy_info
    from config import CURRENT_TRADING_STRATEGY
    
    current = CURRENT_TRADING_STRATEGY
    strat_info = get_strategy_info(current)
    
    return {
        "current_strategy": current,
        "info": strat_info
    }


# ----------------------------------------------------------------
# MODEL PERFORMANCE
# ----------------------------------------------------------------

@app.get("/api/performance")
def performance_metrics(days: int = Query(default=7, ge=1, le=90)):
    """Get model performance metrics over N days."""
    stats = get_model_performance_stats(days)
    distribution = get_anomaly_distribution()
    return {
        "performance": stats,
        "distribution": distribution
    }


# ----------------------------------------------------------------
# EXPORT / DOWNLOAD
# ----------------------------------------------------------------

@app.get("/api/export/anomalies")
def export_anomalies(limit: int = Query(default=1000, ge=1, le=10000)):
    """Export anomalies as CSV file."""
    import tempfile
    import os
    from starlette.background import BackgroundTask
    
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()
        
        export_anomalies_to_csv(temp_file.name, limit)
        
        return FileResponse(
            path=temp_file.name,
            filename="btc_anomalies.csv",
            media_type="text/csv",
            background=BackgroundTask(os.unlink, temp_file.name),
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/export/price-data")
def export_price_data(
    days: int = Query(default=30, ge=1, le=90)
):
    """Export historical price data as CSV file."""
    import tempfile
    import os
    from starlette.background import BackgroundTask
    
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()
        
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        export_price_data_to_csv(temp_file.name, start, end)
        
        return FileResponse(
            path=temp_file.name,
            filename=f"btc_price_data_{days}d.csv",
            media_type="text/csv",
            background=BackgroundTask(os.unlink, temp_file.name),
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ----------------------------------------------------------------
# DOCUMENTATION & EXPLANATIONS
# ----------------------------------------------------------------

@app.get("/api/explain/models")
def explain_models():
    """Get detailed explanation of how the ML models work."""
    return {
        "ensemble": {
            "description": "Risk score is calculated from 3 different machine learning models, each contributing a weighted portion of the final score.",
            "final_score": "Combined risk score (0-100)"
        },
        "models": [
            {
                "name": "Isolation Forest",
                "weight": "25%",
                "description": "Detects anomalies by identifying price and volume patterns that deviate significantly from normal clusters.",
                "how_it_works": "Isolates outliers by randomly selecting features and split values. Anomalies require fewer splits to isolate.",
                "good_for": "Detecting complex multivariate anomalies in OHLCV data.",
                "weakness": "May miss gradual trend changes"
            },
            {
                "name": "Z-Score (Statistical)",
                "weight": "25%",
                "description": "Measures how many standard deviations the current price is from the 24-hour mean.",
                "how_it_works": "Z = (price - mean) / std_dev. Higher deviation = higher anomaly score.",
                "good_for": "Detecting sudden price shocks and extreme values.",
                "weakness": "Assumes normal distribution, may react too strongly to outliers"
            },
            {
                "name": "LSTM Neural Network",
                "weight": "50%",
                "description": "Predicts the next price based on a 60-candle sequence. Anomalies are flagged when actual price differs significantly from prediction.",
                "how_it_works": "LSTM learns temporal patterns. Large prediction errors indicate unusual market behavior.",
                "good_for": "Capturing time-series patterns and trend breaks.",
                "weakness": "Requires historical data, may lag during regime shifts"
            }
        ],
        "consensus": {
            "description": "When 2+ models flag the same data as anomalous, confidence is higher.",
            "high_confidence": "3/3 models agree",
            "medium_confidence": "2/3 models agree",
            "low_confidence": "1/3 models agree"
        }
    }


@app.get("/api/explain/risk-levels")
def explain_risk_levels():
    """Get explanation of risk level categories and recommendations."""
    return {
        "low": {
            "range": "0-30",
            "emoji": "🟢",
            "meaning": "Market is calm and stable",
            "recommendation": "Market conditions are favorable. Good entry point for positions.",
            "anomaly_count": "Few to no anomalies detected"
        },
        "medium": {
            "range": "31-60",
            "emoji": "🟡",
            "meaning": "Market showing mixed signals",
            "recommendation": "Exercise caution. Wait for clearer signals before large positions.",
            "anomaly_count": "Some anomalies detected"
        },
        "high": {
            "range": "61-80",
            "emoji": "🟠",
            "meaning": "Market is volatile with elevated risk",
            "recommendation": "Reduce exposure or tighten stops. Multiple anomaly signals.",
            "anomaly_count": "Frequent anomalies detected"
        },
        "critical": {
            "range": "81-100",
            "emoji": "🔴",
            "meaning": "Market is highly unstable",
            "recommendation": "Consider exiting positions. Severe anomalies detected.",
            "anomaly_count": "Extreme anomalies frequently detected"
        }
    }


# ----------------------------------------------------------------
# AUTH PLACEHOLDER
# ----------------------------------------------------------------

@app.post("/api/auth/login")
def login(payload: dict = Body(default={})):
    return {"message": "Auth not implemented yet.", "token": None}


@app.post("/api/auth/logout")
def logout():
    return {"message": "Logged out."}


# ----------------------------------------------------------------
# SIGNAL BACKTEST ACCURACY
# ----------------------------------------------------------------

@app.get("/api/backtest")
def backtest(days: int = Query(default=30, ge=1, le=90), coin: str = Query(default="BTC")):
    return get_signal_backtest(days=days, coin=coin.upper())


# ----------------------------------------------------------------
# FEATURE SNAPSHOT  (v1 — reusable by downstream Series 1 systems)
# ----------------------------------------------------------------

@app.get("/api/v1/features/{coin}")
def get_features_v1(coin: str, limit: int = Query(default=200, ge=20, le=1000)):
    """
    Returns a full suite of technical features for the latest candle.
    Reusable by 1B (Regime), 1C (Volatility), 1D (Contagion), 1G (Execution).
    """
    coin = coin.upper()
    rows = get_price_history(limit=limit, coin=coin)
    if not rows:
        raise HTTPException(status_code=503, detail=f"No price history available for {coin}.")
    df      = pd.DataFrame([dict(r) for r in rows])
    snapshot = get_feature_snapshot(df)
    return {
        "coin":     coin,
        "snapshot": snapshot,
        "rows_used": len(df),
        "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
    }


# ----------------------------------------------------------------
# STANDARDIZED RISK SCORE  (v1 — canonical output for Series 1)
# ----------------------------------------------------------------

@app.get("/api/v1/risk/{coin}")
def get_risk_score_v1(coin: str):
    """
    Returns the canonical RiskScore for a coin.
    This is the standard output consumed by downstream Series 1 systems
    (1B Regime, 1C Volatility, 1D Contagion, 1E Master Risk, …).
    """
    coin = coin.upper()
    cached = _last_score_result.get(coin)
    if not cached:
        raise HTTPException(status_code=503, detail=f"No risk score available for {coin} yet.")
    live_data = fetch_latest_price(coin) or {}
    return build_risk_response(cached, live_data).model_dump()


@app.get("/api/v1/explain/{coin}")
def get_explain_v1(coin: str):
    """
    Step 6 — Explainability endpoint.
    Returns the current risk score WITH per-model reasoning:
      - model_breakdown: flat scores per model
      - reasoning:       human-readable 'why' per model
      - ensemble_weights: dynamic weights used
    Consumed by downstream 1E, 1F, 1G systems.
    """
    coin = coin.upper()
    cached = _last_score_result.get(coin)
    if not cached:
        raise HTTPException(status_code=503, detail=f"No explain data available for {coin} yet.")
    live_data = fetch_latest_price(coin) or {}
    return build_explain_response(cached, live_data).model_dump()


# ----------------------------------------------------------------
# LSTM PRICE FORECAST
# ----------------------------------------------------------------

@app.get("/api/forecast")
def price_forecast(steps: int = Query(default=12, ge=1, le=48), coin: str = Query(default="BTC")):
    """Return LSTM iterative price forecast for the next `steps` hours."""
    coin = coin.upper()
    predictions = forecast_prices(steps=steps, coin=coin)
    return {"steps": steps, "coin": coin, "forecast": predictions}


# ----------------------------------------------------------------
# ALERTS TEST
# ----------------------------------------------------------------

@app.post("/api/admin/alerts/test")
def test_alerts(coin: str = Query(default="BTC")):
    """Send a test alert through all configured channels."""
    from service_layer.alerts import send_email_alert, send_slack_alert
    coin = coin.upper()
    cached = _last_score_result.get(coin)
    test_data = {
        "close":      cached["close_price"] if cached else 0,
        "if_score":   cached["isolation_forest_score"] if cached else 0,
        "z_score":    cached["zscore_score"] if cached else 0,
        "lstm_score": cached["lstm_score"] if cached else 0,
        "summary":    cached["plain_english_summary"] if cached else "Test alert",
        "timestamp":  datetime.utcnow().isoformat(),
    }
    email_sent = send_email_alert(99.0, "Critical", test_data)
    slack_sent = send_slack_alert(99.0, "Critical", test_data)
    from config import ALERT_EMAIL_ENABLED, ALERT_SLACK_ENABLED
    return {
        "email_enabled": ALERT_EMAIL_ENABLED,
        "slack_enabled": ALERT_SLACK_ENABLED,
        "email_sent":    email_sent,
        "slack_sent":    slack_sent,
    }


@app.get("/api/admin/alerts/status")
def alerts_status():
    from config import (ALERT_EMAIL_ENABLED, ALERT_SLACK_ENABLED,
                        ALERT_RISK_THRESHOLD, ALERT_EMAIL_TO)
    return {
        "email_enabled":    ALERT_EMAIL_ENABLED,
        "slack_enabled":    ALERT_SLACK_ENABLED,
        "risk_threshold":   ALERT_RISK_THRESHOLD,
        "email_to":         ALERT_EMAIL_TO if ALERT_EMAIL_ENABLED else None,
    }