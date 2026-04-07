# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# config.py -- All constants and configuration in one place
# ============================================================

import os

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
DB_PATH         = os.path.join(BASE_DIR, "crypto_risk.db")
SEED_CSV_PATH   = os.path.join(DATA_DIR, "btc_seed.csv")
LOG_FILE        = os.path.join(LOGS_DIR, "app.log")

# ----------------------------------------------------------------
# SERVERS
# ----------------------------------------------------------------
PORT            = int(os.environ.get("PORT", 8000))  # Railway sets $PORT at runtime
FASTAPI_PORT    = PORT   # kept for backward compat
FLASK_PORT      = 5000   # legacy — Flask removed; kept so old imports don't break

# ----------------------------------------------------------------
# DATA SOURCES
# ----------------------------------------------------------------
COINGECKO_URL   = "https://api.coingecko.com/api/v3"
BINANCE_URL     = "https://api.binance.com/api/v3"
COIN_ID         = "bitcoin"
SYMBOL          = "BTCUSDT"

# Per-coin config used by the multi-coin fetch pipeline
COIN_CONFIG = {
    "BTC": {"coin_id": "bitcoin",  "symbol": "BTCUSDT"},
    "ETH": {"coin_id": "ethereum", "symbol": "ETHUSDT"},
}

POLLING_INTERVAL_SECONDS = 10
CACHE_TTL_SECONDS        = 12
API_MAX_RETRIES          = 3
API_RETRY_DELAY          = 2

# ----------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------
LOG_MAX_BYTES    = 10_485_760   # 10 MB per log file before rotation
LOG_BACKUP_COUNT = 5            # keep last 5 rotated files


def setup_logging():
    """
    Configure the root logger with a rotating file handler + console handler.
    Call once at process startup (e.g. from api_backend.py or main.py).
    Idempotent — safe to call multiple times.
    """
    import logging
    import logging.handlers

    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    os.makedirs(LOGS_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    # Rotating file
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        pass  # non-writable filesystem (e.g. read-only deploy) — console only

    root.addHandler(console)
    root.setLevel(logging.INFO)

# ----------------------------------------------------------------
# DATA RETENTION
# ----------------------------------------------------------------
RETENTION_DAYS = 90

# ----------------------------------------------------------------
# ML ENSEMBLE WEIGHTS
# ----------------------------------------------------------------
WEIGHT_ISOLATION_FOREST = 0.25
WEIGHT_ZSCORE           = 0.25
WEIGHT_LSTM             = 0.50

# ----------------------------------------------------------------
# RISK SCORE THRESHOLDS
# ----------------------------------------------------------------
RISK_LOW_MAX          = 30
RISK_MEDIUM_MAX       = 60
RISK_HIGH_MAX         = 80
ANOMALY_LOG_THRESHOLD = 61

RISK_LEVELS = {
    "low":      (0,  30),
    "medium":   (31, 60),
    "high":     (61, 80),
    "critical": (81, 100),
}

# ----------------------------------------------------------------
# LSTM CONFIGURATION
# ----------------------------------------------------------------
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS          = 20
LSTM_BATCH_SIZE      = 32
LSTM_HIDDEN_SIZE     = 64
LSTM_NUM_LAYERS      = 2
LSTM_FEATURES        = ["open", "high", "low", "close", "volume", "volatility_24h"]

# ----------------------------------------------------------------
# SCHEDULER
# ----------------------------------------------------------------
RETRAIN_HOUR   = 0
RETRAIN_MINUTE = 0
PURGE_HOUR     = 1
PURGE_MINUTE   = 0

# ----------------------------------------------------------------
# CHART RANGE CONFIGURATION  (candle frequency + look-back window)
# ----------------------------------------------------------------
CHART_RANGE_CONFIG = {
    "1H":  {"freq": "1h",  "days": 7},     # hourly candles, last 7 days
    "1D":  {"freq": "1D",  "days": 90},    # daily candles, last 90 days
    "1W":  {"freq": "1W",  "days": 365},   # weekly candles, last year
    "1M":  {"freq": "1ME", "days": 730},   # monthly candles, last 2 years
    "ALL": {"freq": "1D",  "days": 3650},  # daily candles, all data
}

# ----------------------------------------------------------------
# BROWSER NOTIFICATION
# ----------------------------------------------------------------
NOTIFICATION_TITLE = "BTC Risk Alert"
NOTIFICATION_BODY  = "Score: {score}/100 ({level}). Possible anomaly detected."

# ----------------------------------------------------------------
# PAGINATION DEFAULTS
# ----------------------------------------------------------------
DEFAULT_PAGE       = 1
DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT     = 100

# ----------------------------------------------------------------
# TRADING STRATEGY
# ----------------------------------------------------------------
CURRENT_TRADING_STRATEGY = "conservative"  # Active strategy: conservative | balanced | aggressive | asymmetric

# ----------------------------------------------------------------
# ALERTING CONFIGURATION
# ----------------------------------------------------------------
# Email Settings (set to None or empty to disable)
ALERT_EMAIL_ENABLED  = False  # Set to True to enable email alerts
ALERT_EMAIL_FROM     = os.getenv("ALERT_EMAIL_FROM", "your-email@example.com")
ALERT_EMAIL_TO       = os.getenv("ALERT_EMAIL_TO", "recipient@example.com")
ALERT_EMAIL_SMTP_URL = os.getenv("ALERT_EMAIL_SMTP_URL", "smtp.gmail.com:587")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")

# Slack Settings (set to None or empty to disable)
ALERT_SLACK_ENABLED = False  # Set to True to enable Slack alerts
ALERT_SLACK_WEBHOOK = os.getenv("ALERT_SLACK_WEBHOOK", "")  # Your Slack webhook URL

# Alert thresholds
ALERT_RISK_THRESHOLD = 80  # Send alert when score >= this (Critical level)

# ----------------------------------------------------------------
# TRADING SIGNALS - STRATEGY PROFILES
# ----------------------------------------------------------------
TRADING_SIGNAL_ENABLED  = True

# Define multiple trading strategies
TRADING_STRATEGIES = {
    "conservative": {
        "name": "Conservative",
        "description": "Safe approach for risk-averse traders. Only buy in very calm markets.",
        "emoji": "🛡️",
        "buy_threshold": 20,
        "sell_threshold": 75,
        "risk_level": "Low Risk",
        "best_for": "Long-term hodlers, risk-averse traders"
    },
    "balanced": {
        "name": "Balanced",
        "description": "Good risk/reward balance. Captures ~70% of rally opportunities.",
        "emoji": "⚖️",
        "buy_threshold": 30,
        "sell_threshold": 78,
        "risk_level": "Medium Risk",
        "best_for": "Day/swing traders, moderate risk seekers"
    },
    "aggressive": {
        "name": "Aggressive",
        "description": "Buys into volatility. Catches rallies in high-volatility markets.",
        "emoji": "🚀",
        "buy_threshold": 45,
        "sell_threshold": 82,
        "risk_level": "High Risk",
        "best_for": "Volatile market hunters, experienced traders"
    },
    "asymmetric": {
        "name": "Asymmetric",
        "description": "Extreme upside bias. Very early signals, requires tight stops.",
        "emoji": "🔥",
        "buy_threshold": 50,
        "sell_threshold": 85,
        "risk_level": "Extreme Risk",
        "best_for": "Risk-takers, momentum followers, scalpers"
    }
}

# Default strategy (user can change this via API)
DEFAULT_TRADING_STRATEGY = "conservative"
CURRENT_TRADING_STRATEGY = DEFAULT_TRADING_STRATEGY

# Legacy config for backward compatibility
SIGNAL_BUY_THRESHOLD  = TRADING_STRATEGIES[DEFAULT_TRADING_STRATEGY]["buy_threshold"]
SIGNAL_SELL_THRESHOLD = TRADING_STRATEGIES[DEFAULT_TRADING_STRATEGY]["sell_threshold"]
SIGNAL_HOLD_RANGE     = (SIGNAL_BUY_THRESHOLD + 1, SIGNAL_SELL_THRESHOLD - 1)

# ----------------------------------------------------------------
# MODEL PERFORMANCE TRACKING
# ----------------------------------------------------------------
PERFORMANCE_WINDOW_DAYS = 7  # Track accuracy over last N days
PERFORMANCE_UPDATE_FREQ = 100  # Update metrics every N predictions

# ----------------------------------------------------------------
# AUTH (placeholder -- not wired yet)
# ----------------------------------------------------------------
SECRET_KEY         = "change-this-before-production"
TOKEN_EXPIRE_HOURS = 24