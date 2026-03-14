"""Quick debug: count trending vs ranging signals in backtest"""
import asyncio
import ccxt.async_support as ccxt
from src.config import Config
from src.indicators import compute_all
from src.strategy import scan_main, detect_regime, Regime


async def main():
    exchange = ccxt.bybit({"options": {"defaultType": "swap"}})
    try:
        since = exchange.milliseconds() - 180 * 24 * 60 * 60 * 1000
        all_candles = []
        while since < exchange.milliseconds():
            ohlcv = await exchange.fetch_ohlcv("BTC/USDT:USDT", "15m", since=since, limit=1000)
            if not ohlcv:
                break
            all_candles.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            await asyncio.sleep(0.1)

        candles = [{"timestamp": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in all_candles]
        cfg = Config()
        df = compute_all(candles, cfg.indicators)

        trending_signals = 0
        ranging_signals = 0
        trending_regime = 0
        for i in range(50, len(df)):
            window = df.iloc[i-50:i+1]
            regime = detect_regime(window, cfg.strategy)
            if regime == Regime.TRENDING:
                trending_regime += 1
            signal = scan_main(window, cfg.strategy)
            if signal:
                if signal.regime == Regime.TRENDING:
                    trending_signals += 1
                    if trending_signals <= 3:
                        print(f"  TRENDING: {signal.reason} @ {signal.entry_price:.0f}")
                else:
                    ranging_signals += 1

        print(f"\nTrending regime candles: {trending_regime}")
        print(f"Trending signals: {trending_signals}")
        print(f"Ranging signals: {ranging_signals}")
    finally:
        await exchange.close()


asyncio.run(main())
