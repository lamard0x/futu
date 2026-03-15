"""
FUTU Backtest Optimizer — sweep params to find best frequency + PnL
"""
import asyncio
import logging
import math
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv

from src.indicators import compute_all
from src.config import IndicatorConfig, StrategyConfig, RiskConfig

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("opt")
log.setLevel(logging.INFO)

TAKER_FEE = 0.0005
ENTRY_FEE = TAKER_FEE
EXIT_FEE = TAKER_FEE
BALANCE_START = 300.0
LEVERAGE = 10
IND_CFG = IndicatorConfig()

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
           "BNB/USDT:USDT", "XAU/USDT:USDT", "AXS/USDT:USDT",
           "FIL/USDT:USDT", "ICP/USDT:USDT", "BCH/USDT:USDT",
           "YFI/USDT:USDT"]

# ═══ Parameter Sets to Test ═══
PARAM_SETS = {
    "CURRENT": {
        "rsi_oversold": 40, "rsi_overbought": 60,
        "bb_touch_pct": 0.5, "volume_range_mult": 0.8,
        "volume_trend_mult": 1.2, "rsi_trend_bull": 50,
        "trending_neutral_bias": False,  # custom: allow neutral bias for trending
        "add_5m_scan": False,
    },
    "A_RSI_WIDER": {
        "rsi_oversold": 45, "rsi_overbought": 55,
        "bb_touch_pct": 0.5, "volume_range_mult": 0.8,
        "volume_trend_mult": 1.2, "rsi_trend_bull": 50,
        "trending_neutral_bias": False,
        "add_5m_scan": False,
    },
    "B_BB_WIDER": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 1.0, "volume_range_mult": 0.8,
        "volume_trend_mult": 1.2, "rsi_trend_bull": 50,
        "trending_neutral_bias": False,
        "add_5m_scan": False,
    },
    "C_VOL_LOWER": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "volume_trend_mult": 1.0, "rsi_trend_bull": 50,
        "trending_neutral_bias": False,
        "add_5m_scan": False,
    },
    "D_TREND_NEUTRAL": {
        "rsi_oversold": 40, "rsi_overbought": 60,
        "bb_touch_pct": 0.5, "volume_range_mult": 0.8,
        "volume_trend_mult": 1.0, "rsi_trend_bull": 45,
        "trending_neutral_bias": True,  # allow trending on neutral HTF
        "add_5m_scan": False,
    },
    "E_COMBO_MILD": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.7,
        "volume_trend_mult": 1.0, "rsi_trend_bull": 45,
        "trending_neutral_bias": True,
        "add_5m_scan": False,
    },
    "F_COMBO_AGG": {
        "rsi_oversold": 45, "rsi_overbought": 55,
        "bb_touch_pct": 1.2, "volume_range_mult": 0.6,
        "volume_trend_mult": 0.9, "rsi_trend_bull": 45,
        "trending_neutral_bias": True,
        "add_5m_scan": False,
    },
    "G_5M_SCAN": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.7,
        "volume_trend_mult": 1.0, "rsi_trend_bull": 45,
        "trending_neutral_bias": True,
        "add_5m_scan": True,
    },
}


@dataclass
class Trade:
    symbol: str
    side: str
    regime: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    notional: float
    entry_time: str
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    candles_held: int = 0
    tf: str = "15m"


def detect_htf_bias(df_htf, ind_cfg):
    if len(df_htf) < ind_cfg.ema_slow + 5:
        return "neutral"
    last = df_htf.iloc[-1]
    ef = last.get(f"ema_{ind_cfg.ema_fast}", 0)
    em = last.get(f"ema_{ind_cfg.ema_mid}", 0)
    es = last.get(f"ema_{ind_cfg.ema_slow}", 0)
    adx = last.get("adx", 0) or 0
    if ef > em > es and adx > 20:
        return "bullish"
    if ef < em < es and adx > 20:
        return "bearish"
    return "neutral"


def calc_size(balance, entry, sl, leverage, max_pos):
    sl_dist = abs(entry - sl) / entry
    if sl_dist == 0:
        return 0, 0
    risk = balance * 0.02
    pv = risk / sl_dist
    mx = (balance * leverage) / max(max_pos, 1)
    pv = min(pv, mx)
    return pv / entry, pv


def scan_bar(row, prev, bias, p, ind_cfg):
    close = row["close"]
    high = row["high"]
    low = row["low"]
    rsi = row.get("rsi") or 0
    prev_rsi = prev.get("rsi") or 0
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    vol = row.get("volume") or 0
    vsma = row.get("volume_sma") or 0
    bbu = row.get("bb_upper") or 0
    bbl = row.get("bb_lower") or 0
    bbm = row.get("bb_mid") or 0
    vwap = row.get("vwap") or 0
    ema21 = row.get(f"ema_{ind_cfg.ema_mid}") or 0

    if atr <= 0:
        return None

    trending = adx >= 25

    # RANGING LONG
    if not trending and bias != "bearish":
        if (bbl > 0 and close <= bbl * (1 + p["bb_touch_pct"] / 100)
                and rsi < p["rsi_oversold"]
                and vsma > 0 and vol > vsma * p["volume_range_mult"]):
            sl = close - 1.0 * atr
            tp = bbm
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= 1.2:
                return {"side": "long", "regime": "ranging", "entry": close, "sl": sl, "tp": tp}

    # RANGING SHORT
    if not trending and bias != "bullish":
        if (bbu > 0 and close >= bbu * (1 - p["bb_touch_pct"] / 100)
                and rsi > p["rsi_overbought"]
                and vsma > 0 and vol > vsma * p["volume_range_mult"]):
            sl = close + 1.0 * atr
            tp = bbm
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= 1.2:
                return {"side": "short", "regime": "ranging", "entry": close, "sl": sl, "tp": tp}

    # TRENDING LONG
    bias_ok = bias == "bullish" or (p["trending_neutral_bias"] and bias == "neutral")
    if trending and bias_ok:
        pullback = (low <= ema21 * 1.005) or (vwap > 0 and low <= vwap * 1.005)
        if (vwap > 0 and close > vwap
                and rsi > p["rsi_trend_bull"] and rsi > prev_rsi
                and vsma > 0 and vol > vsma * p["volume_trend_mult"]
                and pullback):
            sl = close - 1.5 * atr
            tp = close + 2.5 * atr
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= 1.3:
                return {"side": "long", "regime": "trending", "entry": close, "sl": sl, "tp": tp}

    # TRENDING SHORT
    bias_ok_s = bias == "bearish" or (p["trending_neutral_bias"] and bias == "neutral")
    if trending and bias_ok_s:
        pullback = (high >= ema21 * 0.995) or (vwap > 0 and high >= vwap * 0.995)
        if (vwap > 0 and close < vwap
                and rsi < (100 - p["rsi_trend_bull"]) and rsi < prev_rsi
                and vsma > 0 and vol > vsma * p["volume_trend_mult"]
                and pullback):
            sl = close + 1.5 * atr
            tp = close - 2.5 * atr
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= 1.3:
                return {"side": "short", "regime": "trending", "entry": close, "sl": sl, "tp": tp}

    return None


def check_exit(t, row, candles, is_ranging):
    h, l = row["high"], row["low"]
    if t.side == "long" and l <= t.sl_price:
        return t.sl_price, "SL"
    if t.side == "short" and h >= t.sl_price:
        return t.sl_price, "SL"
    if t.side == "long" and h >= t.tp_price:
        return t.tp_price, "TP"
    if t.side == "short" and l <= t.tp_price:
        return t.tp_price, "TP"
    if is_ranging and candles >= 15:
        return row["close"], "TIME"
    return None


def update_trail(t, row):
    cl = row.get("chandelier_long")
    cs = row.get("chandelier_short")
    if t.side == "long" and cl and cl > t.entry_price and cl > t.sl_price:
        t.sl_price = cl
    elif t.side == "short" and cs and cs < t.entry_price and cs < t.sl_price:
        t.sl_price = cs


async def fetch_all(ex, symbol, tf, since_ms):
    all_c = []
    cur = since_ms
    while True:
        try:
            data = await ex.fetch_ohlcv(symbol, tf, since=cur, limit=1000)
        except Exception:
            break
        if not data:
            break
        all_c.extend(data)
        if len(data) < 1000:
            break
        cur = data[-1][0] + 1
        await asyncio.sleep(0.12)
    return all_c


def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    return df


def run_sim(params, data_cache, backtest_start):
    """Run simulation with given params on cached data."""
    balance = BALANCE_START
    trades = []
    open_trades = []
    daily_loss = 0.0
    daily_date = ""
    cooldown = 0
    wins = 0
    losses = 0
    peak = BALANCE_START
    max_dd = 0.0

    timeframes_to_scan = ["15m"]
    if params.get("add_5m_scan"):
        timeframes_to_scan.append("5m")

    for symbol in SYMBOLS:
        for tf in timeframes_to_scan:
            key = f"{symbol}:{tf}"
            if key not in data_cache:
                continue

            df_ind = data_cache[key]
            htf_biases = data_cache.get(f"{symbol}:htf_bias", {})

            for i in range(IND_CFG.ema_slow + 5, len(df_ind)):
                bar_time = df_ind.index[i]
                if bar_time.replace(tzinfo=timezone.utc) < backtest_start:
                    continue

                row = df_ind.iloc[i]
                prev = df_ind.iloc[i - 1]

                day_str = bar_time.strftime("%Y-%m-%d")
                if daily_date != day_str:
                    daily_loss = 0.0
                    daily_date = day_str

                # Check exits
                closed = []
                for t in open_trades:
                    if t.symbol != symbol or t.tf != tf:
                        continue
                    t.candles_held += 1
                    update_trail(t, row)
                    r = check_exit(t, row, t.candles_held, t.regime == "ranging")
                    if r:
                        ep, reason = r
                        t.exit_price = ep
                        t.exit_time = str(bar_time)
                        t.exit_reason = reason
                        if t.side == "long":
                            t.pnl_gross = (ep - t.entry_price) * t.size
                        else:
                            t.pnl_gross = (t.entry_price - ep) * t.size
                        fee = t.notional * ENTRY_FEE + abs(ep * t.size) * EXIT_FEE
                        t.pnl_net = t.pnl_gross - fee
                        balance += t.pnl_net
                        if t.pnl_net >= 0:
                            wins += 1
                        else:
                            losses += 1
                            daily_loss += abs(t.pnl_net)
                            cooldown = 2
                        trades.append(t)
                        closed.append(t)
                        if balance > peak:
                            peak = balance
                        dd = (peak - balance) / peak if peak > 0 else 0
                        if dd > max_dd:
                            max_dd = dd

                for t in closed:
                    open_trades.remove(t)

                if cooldown > 0:
                    cooldown -= 1
                    continue

                if daily_loss >= balance * 0.06:
                    continue
                if len(open_trades) >= 3:
                    continue
                if any(t.symbol == symbol and t.tf == tf for t in open_trades):
                    continue

                bias = "neutral"
                for ht in sorted(htf_biases.keys(), reverse=True):
                    if bar_time >= ht:
                        bias = htf_biases[ht]
                        break

                sig = scan_bar(row, prev, bias, params, IND_CFG)
                if not sig:
                    continue

                amt, notional = calc_size(balance, sig["entry"], sig["sl"], LEVERAGE, 3)
                if amt <= 0:
                    continue

                trade = Trade(symbol=symbol, side=sig["side"], regime=sig["regime"],
                              entry_price=sig["entry"], sl_price=sig["sl"], tp_price=sig["tp"],
                              size=amt, notional=notional, entry_time=str(bar_time), tf=tf)
                open_trades.append(trade)

    total = len(trades)
    if total == 0:
        return {"trades": 0, "pnl": 0, "wr": 0, "dd": 0, "avg_day": 0,
                "fees": 0, "pf": 0, "balance": BALANCE_START}

    total_pnl = sum(t.pnl_net for t in trades)
    total_fees = sum(t.notional * ENTRY_FEE + abs(t.exit_price * t.size) * EXIT_FEE for t in trades)
    wr = wins / total * 100
    win_pnl = sum(t.pnl_net for t in trades if t.pnl_net >= 0)
    loss_pnl = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))
    pf = win_pnl / loss_pnl if loss_pnl > 0 else 999

    days_with_trades = len(set(t.exit_time[:10] for t in trades))
    trending_count = sum(1 for t in trades if t.regime == "trending")
    ranging_count = sum(1 for t in trades if t.regime == "ranging")

    return {
        "trades": total, "wins": wins, "losses": losses,
        "pnl": total_pnl, "wr": wr, "dd": max_dd * 100,
        "fees": total_fees, "pf": pf, "balance": balance,
        "avg_day": total / 30, "days_active": days_with_trades,
        "trending": trending_count, "ranging": ranging_count,
    }


async def main():
    log.info("=" * 70)
    log.info("FUTU Parameter Optimizer — 30 Days")
    log.info("=" * 70)

    ex = ccxt.okx({"options": {"defaultType": "swap"}})
    await ex.load_markets()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=32)
    since_ms = int(since.timestamp() * 1000)
    backtest_start = now - timedelta(days=30)

    # Fetch all data once
    data_cache = {}
    for symbol in SYMBOLS:
        if symbol not in ex.markets:
            continue
        short = symbol.split("/")[0]
        log.info("Fetching %s...", short)

        for tf in ["15m", "5m", "4h"]:
            raw = await fetch_all(ex, symbol, tf, since_ms)
            if len(raw) < 50:
                continue
            df = to_df(raw)
            candles = [{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                        "low": r["low"], "close": r["close"], "volume": r["volume"]}
                       for ts, r in df.iterrows()]
            df_ind = compute_all(candles, IND_CFG)
            data_cache[f"{symbol}:{tf}"] = df_ind

            if tf == "4h":
                htf_biases = {}
                for i in range(IND_CFG.ema_slow + 5, len(df_ind)):
                    chunk = df_ind.iloc[:i + 1]
                    bias = detect_htf_bias(chunk, IND_CFG)
                    htf_biases[df_ind.index[i]] = bias
                data_cache[f"{symbol}:htf_bias"] = htf_biases

        await asyncio.sleep(0.2)

    await ex.close()

    # Run all parameter sets
    log.info("\n" + "=" * 70)
    log.info("%-18s %6s %5s %5s %8s %6s %6s %5s %5s %5s",
             "SET", "TRADES", "W", "L", "PNL", "WR%", "DD%", "PF", "T/DAY", "TREND")
    log.info("-" * 70)

    results = {}
    for name, params in PARAM_SETS.items():
        r = run_sim(params, data_cache, backtest_start)
        results[name] = r
        log.info("%-18s %6d %5d %5d %+8.1f %5.1f%% %5.1f%% %5.1f %5.1f %5d",
                 name, r["trades"], r.get("wins", 0), r.get("losses", 0),
                 r["pnl"], r["wr"], r["dd"], r["pf"], r["avg_day"], r.get("trending", 0))

    # Find best
    log.info("-" * 70)
    best = max(results.items(), key=lambda x: x[1]["pnl"] * (x[1]["wr"] / 100))
    log.info("\nBEST: %s", best[0])
    r = best[1]
    log.info("  Trades: %d (%.1f/day) | PnL: $%+.2f | WR: %.1f%% | DD: %.1f%% | PF: %.1f",
             r["trades"], r["avg_day"], r["pnl"], r["wr"], r["dd"], r["pf"])
    log.info("  Fees: $%.2f | Trending: %d | Ranging: %d",
             r["fees"], r.get("trending", 0), r.get("ranging", 0))
    log.info("  Final Balance: $%.2f ($%.0f start)", r["balance"], BALANCE_START)

    # Print the winning params
    bp = PARAM_SETS[best[0]]
    log.info("\n  RECOMMENDED PARAMS:")
    log.info("    rsi_oversold:      %.0f", bp["rsi_oversold"])
    log.info("    rsi_overbought:    %.0f", bp["rsi_overbought"])
    log.info("    bb_touch_pct:      %.1f", bp["bb_touch_pct"])
    log.info("    volume_range_mult: %.1f", bp["volume_range_mult"])
    log.info("    volume_trend_mult: %.1f", bp["volume_trend_mult"])
    log.info("    rsi_trend_bull:    %.0f", bp["rsi_trend_bull"])
    log.info("    trending_neutral:  %s", bp["trending_neutral_bias"])
    log.info("    add_5m_scan:       %s", bp.get("add_5m_scan", False))


if __name__ == "__main__":
    asyncio.run(main())
