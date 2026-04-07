# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# utils.py -- Shared utilities (retry, helpers)
# STEP 8 — Performance & Reliability
# ============================================================

import time
import logging
import requests
from typing import Optional

from config import API_MAX_RETRIES, API_RETRY_DELAY

logger = logging.getLogger(__name__)


def retry_get(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 10,
    label: str = "",
) -> requests.Response:
    """
    Perform an HTTP GET with automatic retry on failure.

    Retries up to API_MAX_RETRIES times, sleeping API_RETRY_DELAY seconds
    between attempts.  Raises RuntimeError if all attempts fail.

    Parameters
    ----------
    url     : Full URL to GET.
    params  : Optional query parameters dict.
    timeout : Per-request timeout in seconds.
    label   : Human-readable name for log messages (e.g. "CoinGecko price BTC").

    Returns
    -------
    requests.Response on success (status checked via raise_for_status).

    Raises
    ------
    RuntimeError — if every attempt fails.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            if attempt > 1:
                logger.info(f"{label} succeeded on attempt {attempt}.")
            return resp
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"{label} attempt {attempt}/{API_MAX_RETRIES} failed: {exc}"
            )
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_DELAY)

    raise RuntimeError(
        f"{label} failed after {API_MAX_RETRIES} retries. Last error: {last_exc}"
    )
