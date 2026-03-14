import asyncio
import logging

import ccxt

logger = logging.getLogger("futu.errors")

RETRY_ERRORS = [
    "network", "timeout", "rate limit", "too many requests",
    "service unavailable", "internal error", "502", "503", "504",
    "ETIMEDOUT", "ECONNRESET", "temporarily unavailable",
]

NO_RETRY_ERRORS = [
    "insufficient", "not enough", "invalid api", "authentication",
    "permission denied", "forbidden", "invalid symbol", "invalid parameter",
    "order not found", "position not found",
]

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]


def is_retryable(error: Exception) -> bool:
    msg = str(error).lower()
    if any(k in msg for k in NO_RETRY_ERRORS):
        return False
    if any(k in msg for k in RETRY_ERRORS):
        return True
    if isinstance(error, (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable)):
        return True
    return False


async def retry_api_call(func, *args, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if not is_retryable(e):
                logger.warning("Non-retryable error: %s", e)
                raise
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            logger.warning(
                "Retryable error (attempt %d/%d, retry in %ds): %s",
                attempt + 1, MAX_RETRIES, delay, e,
            )
            await asyncio.sleep(delay)
    raise last_error
