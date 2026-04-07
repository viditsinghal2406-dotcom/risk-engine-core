# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# data_pipeline.py -- Fetching, caching, seeding, storing price data
# ============================================================

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from config import (
    COINGECKO_URL, BINANCE_URL,
    COIN_ID, SYMBOL, COIN_CONFIG,
    CACHE_TTL_SECONDS, API_MAX_RETRIES, API_RETRY_DELAY,
    SEED_CSV_PATH, LSTM_FEATURES,
    POLLING_INTERVAL_SECONDS
)
from utils import retry_get

# How many polling intervals fit in one day. Live fetch stores the 24h rolling
# volume total, so we normalize it to a per-interval amount so that resampling
# daily candles with `sum` gives back the correct 24h volume.
_INTERVALS_PER_DAY = 86400 / POLLING_INTERVAL_SECONDS  # 2880 for 30s interval
from data_layer.database import (
    insert_price_row, insert_price_rows_bulk,
    get_latest_price, get_price_history, log_event
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# IN-MEMORY CACHE  (per-coin)
# ----------------------------------------------------------------
# _price_cache[coin] = {"data": ..., "fetched_at": ..., "source": ...}
_price_cache: dict = {}


def _is_cache_valid(coin: str = "BTC") -> bool:
    coin = coin.upper()
    c = _price_cache.get(coin)
    if not c or c["data"] is None or c["fetched_at"] is None:
        return False
    age = (datetime.utcnow() - c["fetched_at"]).total_seconds()
    return age < CACHE_TTL_SECONDS


def _set_cache(data: dict, source: str, coin: str = "BTC"):
    coin = coin.upper()
    _price_cache[coin] = {"data": data, "fetched_at": datetime.utcnow(), "source": source}


def clear_price_cache(coin: str = "BTC"):
    """Invalidate the in-memory price cache so next fetch hits the API."""
    coin = coin.upper()
    if coin in _price_cache:
        _price_cache[coin] = {"data": None, "fetched_at": None, "source": None}


def _get_cache(coin: str = "BTC") -> Optional[dict]:
    coin = coin.upper()
    if _is_cache_valid(coin):
        return {**_price_cache[coin]["data"], "source": "cache"}
    return None


# ----------------------------------------------------------------
# COMPUTE 24H ROLLING VOLATILITY
# ----------------------------------------------------------------

def compute_volatility(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(window=1440, min_periods=1).std().fillna(0)


# ----------------------------------------------------------------
# COINGECKO -- PRIMARY SOURCE
# ----------------------------------------------------------------

def _fetch_coingecko(coin: str = "BTC") -> Optional[dict]:
    cfg     = COIN_CONFIG.get(coin.upper(), COIN_CONFIG["BTC"])
    coin_id = cfg["coin_id"]
    try:
        price_resp = retry_get(
            f"{COINGECKO_URL}/simple/price",
            params={
                "ids":                 coin_id,
                "vs_currencies":       "usd",
                "include_24hr_vol":    "true",
                "include_24hr_change": "true",
            },
            label=f"CoinGecko price {coin}",
        )
        ohlc_resp = retry_get(
            f"{COINGECKO_URL}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": "1"},
            label=f"CoinGecko OHLC {coin}",
        )

        price_data  = price_resp.json()[coin_id]
        ohlc_data   = ohlc_resp.json()
        latest_ohlc = ohlc_data[-1]
        spot_price  = price_data["usd"]

        # Use Binance quoteVolume (USDT) so live ticks are consistent with
        # historical candles (seeded from Binance klines c[7] = USDT volume).
        # The frontend converts USDT → base-asset units by dividing by close price.
        cfg    = COIN_CONFIG.get(coin.upper(), COIN_CONFIG["BTC"])
        symbol = cfg["symbol"]
        quote_vol_24h = None
        try:
            vol_resp = retry_get(
                f"{BINANCE_URL}/ticker/24hr",
                params={"symbol": symbol},
                label=f"Binance volume {coin}",
            )
            quote_vol_24h = float(vol_resp.json()["quoteVolume"])  # USDT volume
        except Exception as ve:
            logger.warning(f"Binance volume fetch failed for {coin}, using CoinGecko vol: {ve}")

        vol_24h = quote_vol_24h if quote_vol_24h is not None else price_data["usd_24h_vol"]

        result = {
            "timestamp":      datetime.utcfromtimestamp(latest_ohlc[0] / 1000).strftime("%Y-%m-%dT%H:%M:%S"),
            "open":           latest_ohlc[1],
            "high":           max(latest_ohlc[2], spot_price),
            "low":            min(latest_ohlc[3], spot_price),
            "close":          spot_price,
            "volume":         vol_24h / _INTERVALS_PER_DAY,
            "volume_24h":     vol_24h,
            "change_24h_pct": price_data.get("usd_24h_change"),
            "source":         "coingecko",
        }
        logger.info(f"CoinGecko fetch OK for {coin}")
        return result

    except Exception as e:
        logger.error(f"CoinGecko failed for {coin}: {e}")
        return None


# ----------------------------------------------------------------
# BINANCE -- BACKUP SOURCE
# ----------------------------------------------------------------

def _fetch_binance(coin: str = "BTC") -> Optional[dict]:
    cfg    = COIN_CONFIG.get(coin.upper(), COIN_CONFIG["BTC"])
    symbol = cfg["symbol"]
    try:
        ticker_resp = retry_get(
            f"{BINANCE_URL}/ticker/24hr",
            params={"symbol": symbol},
            label=f"Binance ticker {coin}",
        )
        kline_resp = retry_get(
            f"{BINANCE_URL}/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 1},
            label=f"Binance klines {coin}",
        )

        ticker        = ticker_resp.json()
        candle        = kline_resp.json()[0]
        quote_vol_24h = float(ticker["quoteVolume"])  # USDT volume, consistent with klines c[7]

        result = {
            "timestamp":      datetime.utcfromtimestamp(candle[0] / 1000).strftime("%Y-%m-%dT%H:%M:%S"),
            "open":           float(candle[1]),
            "high":           float(candle[2]),
            "low":            float(candle[3]),
            "close":          float(candle[4]),
            "volume":         quote_vol_24h / _INTERVALS_PER_DAY,
            "volume_24h":     quote_vol_24h,
            "change_24h_pct": float(ticker.get("priceChangePercent", 0) or 0),
            "source":         "binance",
        }
        logger.info(f"Binance fallback fetch OK for {coin}")
        log_event("api_fallback", f"Fell back to Binance for live {coin} price.")
        return result

    except Exception as e:
        logger.error(f"Binance failed for {coin}: {e}")
        return None


# ----------------------------------------------------------------
# LIVE VOLATILITY HELPER
# ----------------------------------------------------------------

def _compute_live_volatility(coin: str = "BTC") -> float:
    """
    Compute a rolling standard deviation of recent close prices as the live
    volatility proxy.  Uses the last 50 stored rows (enough to give a stable
    std while being fast).  Falls back to 0.0 on any error.
    """
    try:
        recent = get_price_history(limit=50, coin=coin)
        if len(recent) >= 5:
            closes = [r["close"] for r in recent]
            return float(pd.Series(closes).std())
    except Exception:
        pass
    return 0.0


# ----------------------------------------------------------------
# MAIN FETCH
# ----------------------------------------------------------------

def fetch_latest_price(coin: str = "BTC") -> Optional[dict]:
    coin = coin.upper()
    cached = _get_cache(coin)
    if cached:
        return cached

    data = _fetch_coingecko(coin)

    if data is None:
        data = _fetch_binance(coin)

    if data is None:
        logger.error(f"All sources failed for {coin}. Using last DB row as fallback.")
        last_row = get_latest_price(coin=coin)
        if last_row:
            data = {
                "timestamp":      last_row["timestamp"],
                "open":           last_row["open"],
                "high":           last_row["high"],
                "low":            last_row["low"],
                "close":          last_row["close"],
                "volume":         last_row["volume"],
                "volatility_24h": last_row["volatility_24h"],
                "source":         "cache",
                "coin":           coin,
            }
            log_event("error", f"All APIs failed for {coin}. Serving last known DB price.")
            _set_cache(data, "cache", coin=coin)
        return data

    data["volatility_24h"] = _compute_live_volatility(coin)
    data["coin"] = coin.upper()

    insert_price_row(data)
    _set_cache(data, data["source"], coin=coin)

    return data


# ----------------------------------------------------------------
# BINANCE HISTORICAL SEED
# ----------------------------------------------------------------

def _fetch_binance_historical(interval: str = "1d", days: int = 365,
                              start_dt: datetime = None, coin: str = "BTC") -> list[dict]:
    cfg      = COIN_CONFIG.get(coin.upper(), COIN_CONFIG["BTC"])
    symbol   = cfg["symbol"]
    url      = f"{BINANCE_URL}/klines"
    origin   = start_dt if start_dt else (datetime.utcnow() - timedelta(days=days))
    start_ms = int(origin.timestamp() * 1000)
    all_candles = []
    limit       = 1000

    logger.info(f"Starting Binance historical fetch ({coin.upper()}, from {origin.strftime('%Y-%m-%d %H:%M')}, interval={interval})...")

    while True:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "limit":     limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            logger.error(f"Historical seed fetch failed: {e}")
            break

        if not candles:
            break

        for c in candles:
            all_candles.append({
                "timestamp":      datetime.utcfromtimestamp(c[0] / 1000).strftime("%Y-%m-%dT%H:%M:%S"),
                "open":           float(c[1]),
                "high":           float(c[2]),
                "low":            float(c[3]),
                "close":          float(c[4]),
                "volume":         float(c[7]),   # c[7] = quote asset (USDT) volume; frontend converts to base asset
                "volatility_24h": 0.0,
                "source":         "binance",
                "coin":           coin.upper(),
            })

        if len(candles) < limit:
            break

        start_ms = candles[-1][6] + 1

    logger.info(f"Fetched {len(all_candles)} historical candles from Binance ({coin.upper()}).")
    return all_candles


def _compute_and_attach_volatility(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    df["timestamp"]      = pd.to_datetime(df["timestamp"])
    df                   = df.sort_values("timestamp").reset_index(drop=True)
    df["volatility_24h"] = compute_volatility(df)
    # Convert timestamp back to string so SQLite can handle it
    df["timestamp"]      = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df.to_dict(orient="records")


def seed_database(coin: str = "BTC"):
    logger.info(f"Seeding database with historical {coin} data...")
    log_event("seed", f"Starting historical data seed for {coin}.")

    rows = _fetch_binance_historical(interval="1d", days=365, coin=coin)
    if not rows:
        logger.error("Seed failed -- no data returned from Binance.")
        log_event("error", "Historical seed returned no data.")
        return

    rows = _compute_and_attach_volatility(rows)

    try:
        import os
        os.makedirs(os.path.dirname(SEED_CSV_PATH), exist_ok=True)
        pd.DataFrame(rows).to_csv(SEED_CSV_PATH, index=False)
        logger.info(f"Seed CSV saved to {SEED_CSV_PATH}")
    except Exception as e:
        logger.warning(f"Could not save seed CSV: {e}")

    insert_price_rows_bulk(rows)
    log_event("seed", f"Seed complete for {coin}. Inserted {len(rows)} rows.", {"coin": coin, "rows": len(rows)})
    logger.info(f"Seeding complete for {coin}. {len(rows)} rows inserted.")


def backfill_missing_candles(coin: str = "BTC"):
    """
    On startup, detect any gap between the last stored price row and now.
    If the gap is > 1 hour, fetch 1h Binance candles to fill it so the
    chart never shows blank periods just because the app was offline.
    """
    coin = coin.upper()
    last_row = get_latest_price(coin=coin)
    if not last_row:
        return  # nothing in DB yet, seeding will handle it

    last_ts = datetime.fromisoformat(last_row["timestamp"])
    now_utc = datetime.utcnow()
    gap_hours = (now_utc - last_ts).total_seconds() / 3600

    if gap_hours < 1.0:
        logger.info(f"No gap detected for {coin}. Backfill not needed.")
        return

    logger.info(f"Gap of {gap_hours:.1f}h detected for {coin} since last data point ({last_ts}). Backfilling...")
    log_event("backfill", f"Backfilling {coin} {gap_hours:.1f}h gap from {last_ts}.")

    # Fetch 1h candles starting from the last stored timestamp
    rows = _fetch_binance_historical(interval="1h", start_dt=last_ts, coin=coin)
    if not rows:
        logger.warning(f"Backfill: no candles returned from Binance for {coin}.")
        return

    # Drop the first candle — it's the same timestamp as last_ts already in DB
    rows = [r for r in rows if r["timestamp"] > last_row["timestamp"]]
    if not rows:
        logger.info(f"Backfill: all returned candles for {coin} already in DB.")
        return

    rows = _compute_and_attach_volatility(rows)
    insert_price_rows_bulk(rows)
    log_event("backfill", f"Backfill complete for {coin}. Inserted {len(rows)} rows.", {"coin": coin, "rows": len(rows)})
    logger.info(f"Backfill complete for {coin}. {len(rows)} rows inserted covering the gap.")


def needs_seeding(coin: str = "BTC") -> bool:
    return get_latest_price(coin=coin.upper()) is None


def needs_deep_history(coin: str = "BTC", min_days: int = 700) -> bool:
    """
    Return True when we have less than `min_days` days of data for `coin`.
    This triggers a 2-year daily backfill so 1W/1M chart ranges are usable.
    """
    from data_layer.database import get_oldest_price_timestamp
    coin = coin.upper()
    oldest = get_oldest_price_timestamp(coin=coin)
    if oldest is None:
        return True
    try:
        oldest_dt = datetime.fromisoformat(oldest)
    except ValueError:
        return True
    age_days = (datetime.utcnow() - oldest_dt).days
    return age_days < min_days


def needs_hourly_seed(min_rows: int = 100, coin: str = "BTC") -> bool:
    """
    Return True when we have fewer than `min_rows` price rows in the last 7 days.
    A daily-only database has at most 7 rows in that window; an hourly-seeded
    one has ~168 rows.  100 is a safe threshold that triggers re-seeding only
    when hourly data is genuinely missing.
    """
    from data_layer.database import count_price_rows_since
    cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
    return count_price_rows_since(cutoff, coin=coin.upper()) < min_rows


def seed_recent_hourly(days: int = 14, coin: str = "BTC"):
    """
    Fetch the last `days` days of 1-hour Binance candles and insert them.
    Uses INSERT OR IGNORE so running it on an already-seeded DB is safe.
    """
    logger.info(f"Seeding last {days} days of 1-hour OHLCV data for {coin} chart wicks...")
    log_event("seed", f"Starting {days}-day hourly data seed for {coin}.")

    coin = coin.upper()
    rows = _fetch_binance_historical(interval="1h", days=days, coin=coin)
    if not rows:
        logger.warning(f"Hourly seed returned no data from Binance for {coin}.")
        log_event("error", f"Hourly seed returned no data for {coin}.")
        return

    rows = _compute_and_attach_volatility(rows)
    insert_price_rows_bulk(rows)
    log_event("seed", f"Hourly seed complete for {coin}. {len(rows)} rows inserted.", {"coin": coin, "rows": len(rows)})
    logger.info(f"Hourly seed complete for {coin}. {len(rows)} rows inserted.")