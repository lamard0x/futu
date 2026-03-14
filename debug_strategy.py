"""Debug: check why trending signals never fire"""
import asyncio
import ccxt.async_support as ccxt
from src.config import Config
from src.indicators import compute_all
from src.strategy import detect_regime, Regime


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

        # Check regime distribution
        trending = 0
        ranging = 0
        uncertain = 0
        for i in range(50, len(df)):
            window = df.iloc[i-50:i+1]
            regime = detect_regime(window, cfg.strategy)
            if regime == Regime.TRENDING:
                trending += 1
            elif regime == Regime.RANGING:
                ranging += 1
            else:
                uncertain += 1

        total = trending + ranging + uncertain
        print(f"Regime distribution ({total} candles):")
        print(f"  TRENDING:  {trending} ({trending/total*100:.1f}%)")
        print(f"  RANGING:   {ranging} ({ranging/total*100:.1f}%)")
        print(f"  UNCERTAIN: {uncertain} ({uncertain/total*100:.1f}%)")

        # ADX stats
        adx = df["adx"].dropna()
        print(f"\nADX stats:")
        print(f"  Min:  {adx.min():.1f}")
        print(f"  Max:  {adx.max():.1f}")
        print(f"  Mean: {adx.mean():.1f}")
        print(f"  > 22: {(adx > 22).sum()} ({(adx > 22).mean()*100:.1f}%)")
        print(f"  > 25: {(adx > 25).sum()} ({(adx > 25).mean()*100:.1f}%)")

        # Check trending signal conditions
        print(f"\nTrending signal debug (last 100 trending candles):")
        trend_count = 0
        for i in range(50, len(df)):
            window = df.iloc[i-50:i+1]
            regime = detect_regime(window, cfg.strategy)
            if regime != Regime.TRENDING:
                continue
            trend_count += 1
            if trend_count > 5:
                break
            row = df.iloc[i]
            ema_f, ema_m, ema_s = row["ema_9"], row["ema_21"], row["ema_50"]
            stack_bull = ema_f > ema_m > ema_s
            stack_bear = ema_f < ema_m < ema_s
            above_vwap = row["close"] > row["vwap"]
            rsi = row["rsi"]
            vol_ok = row["volume"] > row["volume_sma"] * 1.1
            pullback_ema21 = row["low"] <= ema_m * 1.005
            pullback_vwap = row["low"] <= row["vwap"] * 1.005
            near_ema9 = abs(row["close"] - ema_f) / row["close"] < 0.003

            print(f"  [{row.name}] close={row['close']:.0f} ADX={row['adx']:.1f}")
            print(f"    EMA stack bull={stack_bull} bear={stack_bear}")
            print(f"    above_vwap={above_vwap} RSI={rsi:.1f} vol_ok={vol_ok}")
            print(f"    pullback_ema21={pullback_ema21} pullback_vwap={pullback_vwap} near_ema9={near_ema9}")
    finally:
        await exchange.close()


asyncio.run(main())
