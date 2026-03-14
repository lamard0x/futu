"""
FUTU Auto-Optimizer — Grid search best strategy parameters

Usage:
    python -m src.optimizer
    python -m src.optimizer --symbol ETH/USDT:USDT --days 90 --max-combos 100
"""

import asyncio
import argparse
import copy
import itertools
import random
import sys
import logging

from src.config import Config
from src.backtest import BacktestEngine, BacktestResult, print_result

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("futu.optimizer")

PARAM_GRID = {
    "strategy": {
        "adx_trending": [20, 25, 30],
        "rsi_oversold": [30, 35, 40],
        "rsi_overbought": [60, 65, 70],
        "main_tp1_atr_mult": [1.0, 1.5, 2.0],
        "main_tp2_atr_mult": [2.0, 2.5, 3.0],
        "main_sl_trending_atr_mult": [1.0, 1.5, 2.0],
        "main_sl_ranging_atr_mult": [0.8, 1.0, 1.5],
    },
    "indicators": {
        "ema_fast": [5, 9, 13],
        "rsi_period": [7, 10, 14],
        "chandelier_mult": [2.5, 3.0, 3.5],
    },
}


def generate_combinations(grid: dict, max_combos: int) -> list[dict]:
    sections = []
    keys = []

    for section, params in grid.items():
        for param, values in params.items():
            sections.append((section, param))
            keys.append(values)

    all_combos = list(itertools.product(*keys))
    total = len(all_combos)

    if total > max_combos:
        print(f"Total combos: {total}, sampling {max_combos}")
        all_combos = random.sample(all_combos, max_combos)
    else:
        print(f"Total combos: {total}")

    results = []
    for combo in all_combos:
        override = {}
        for (section, param), value in zip(sections, combo):
            if section not in override:
                override[section] = {}
            override[section][param] = value
        results.append(override)

    return results


def score_result(r: BacktestResult) -> float:
    if r.total_trades < 10:
        return -999
    if r.max_drawdown_pct > 30:
        return -999
    if r.win_rate < 35:
        return -999

    return (
        r.profit_factor * 0.3
        + r.sharpe_ratio * 0.2
        + (r.win_rate / 100) * 0.2
        + (r.total_pnl_pct / 100) * 0.2
        - (r.max_drawdown_pct / 100) * 0.1
    )


def apply_overrides(config: Config, overrides: dict):
    for section_name, values in overrides.items():
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for key, val in values.items():
            if hasattr(section, key):
                setattr(section, key, type(getattr(section, key))(val))


async def fetch_historical(symbol: str, timeframe: str, days: int) -> list[dict]:
    import ccxt.async_support as ccxt
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
            {"timestamp": c[0], "open": c[1], "high": c[2],
             "low": c[3], "close": c[4], "volume": c[5]}
            for c in all_candles
        ]
        print(f"Fetched {len(candles)} candles ({days}d, {timeframe})")
        return candles
    finally:
        await exchange.close()


async def run_optimization(
    symbol: str, days: int, balance: float, max_combos: int, save: bool
):
    print(f"\nFetching {symbol} data...")
    candles_15m = await fetch_historical(symbol, "15m", days)
    candles_htf = await fetch_historical(symbol, "4h", days + 30)

    if len(candles_15m) < 100:
        print("Not enough data")
        sys.exit(1)

    combos = generate_combinations(PARAM_GRID, max_combos)
    results: list[tuple[dict, BacktestResult, float]] = []

    print(f"\nRunning {len(combos)} backtests...\n")

    for i, override in enumerate(combos):
        config = Config()
        config.risk.account_balance = balance
        apply_overrides(config, override)

        engine = BacktestEngine(config)
        result = engine.run(candles_15m, candles_htf=candles_htf)
        s = score_result(result)
        results.append((override, result, s))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    results.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 80)
    print(f"{'#':>3} {'Score':>7} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL%':>8} {'DD%':>6} {'Sharpe':>7}")
    print("-" * 80)

    for i, (override, r, s) in enumerate(results[:10], 1):
        print(
            f"{i:>3} {s:>7.3f} {r.total_trades:>7} {r.win_rate:>5.1f}% "
            f"{r.profit_factor:>5.2f} {r.total_pnl_pct:>+7.1f}% "
            f"{r.max_drawdown_pct:>5.1f}% {r.sharpe_ratio:>6.2f}"
        )

    best_override, best_result, best_score = results[0]

    print(f"\n{'='*80}")
    print("BEST PARAMETERS:")
    print(f"{'='*80}")
    for section, params in best_override.items():
        for k, v in params.items():
            print(f"  {section}.{k} = {v}")

    print_result(best_result)

    if save:
        config = Config()
        config.save_overrides(best_override)
        print(f"\nSaved to config_overrides.json")
    else:
        print(f"\nRun with --save to apply these params")


async def main():
    parser = argparse.ArgumentParser(description="FUTU Auto-Optimizer")
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--balance", type=float, default=300)
    parser.add_argument("--max-combos", type=int, default=200)
    parser.add_argument("--save", action="store_true", help="Save best params")
    args = parser.parse_args()

    await run_optimization(args.symbol, args.days, args.balance, args.max_combos, args.save)


if __name__ == "__main__":
    asyncio.run(main())
