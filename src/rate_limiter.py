import asyncio
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("futu.ratelimit")


@dataclass
class TokenBucket:
    capacity: int
    refill_per_second: float
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    async def acquire(self, timeout: float = 30.0):
        start = time.monotonic()
        while True:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            if time.monotonic() - start > timeout:
                logger.warning("Rate limit timeout after %.1fs", timeout)
                return
            wait = (1 - self.tokens) / self.refill_per_second
            await asyncio.sleep(min(wait, 1.0))

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)


class RateLimiter:
    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._default = TokenBucket(capacity=10, refill_per_second=5)
        self._limits = {
            "fetch_ohlcv": TokenBucket(capacity=20, refill_per_second=5),
            "fetch_ticker": TokenBucket(capacity=20, refill_per_second=10),
            "fetch_tickers": TokenBucket(capacity=5, refill_per_second=1),
            "fetch_balance": TokenBucket(capacity=10, refill_per_second=5),
            "fetch_positions": TokenBucket(capacity=10, refill_per_second=5),
            "create_order": TokenBucket(capacity=10, refill_per_second=5),
            "cancel_all_orders": TokenBucket(capacity=5, refill_per_second=2),
            "fetch_funding_rate": TokenBucket(capacity=10, refill_per_second=3),
            "fetch_my_trades": TokenBucket(capacity=5, refill_per_second=2),
        }

    async def acquire(self, method: str):
        bucket = self._limits.get(method, self._default)
        await bucket.acquire()
