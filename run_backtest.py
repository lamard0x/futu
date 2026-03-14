"""
FUTU Scalper v1 — Backtest Runner

Usage:
    python run_backtest.py
    python run_backtest.py --symbol ETH/USDT:USDT --days 90
"""

import asyncio
import argparse
import sys

import ccxt.async_support as ccxt

from src.config import Config
from src.backtest import BacktestEngine, print_result


async def fetch_historical(symbol: str, timeframe: str, days: int) -> list[dict]:
    exchange = ccxt.bybit({"options": {"defaultType": "swap"}})
    try:
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_candles = []
        while since < exchange.milliseconds():
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            await asyncio.sleep(0.1)

        candles = [
            {
                "timestamp": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in all_candles
        ]
        print(f"Fetched {len(candles)} candles ({days} days, {timeframe})")
        return candles
    finally:
        await exchange.close()


async def main():
    parser = argparse.ArgumentParser(description="FUTU Backtest Runner")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=60, help="Days of history")
    parser.add_argument("--balance", type=float, default=300, help="Starting balance")
    args = parser.parse_args()

    config = Config()
    config.risk.account_balance = args.balance
    config.exchange.symbol = args.symbol

    print(f"Fetching {args.symbol} data...")
    candles_15m = await fetch_historical(args.symbol, "15m", args.days)
    candles_htf = await fetch_historical(args.symbol, "4h", args.days + 30)

    if len(candles_15m) < 100:
        print("Not enough data for backtest")
        sys.exit(1)

    engine = BacktestEngine(config)
    result = engine.run(candles_15m, candles_htf=candles_htf)
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
