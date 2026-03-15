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
    # ═══ Baseline: ranging only ═══
    "RANGE_ONLY": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 0,
    },
    # ═══ Parallel: ranging (3) + breakout trending (2) ═══
    # Trending params: t_adx_min, t_vol_mult, t_lookback, t_body_pct, t_sl_atr, t_min_rr
    "BRK_STRICT": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 2,
        "t_adx_min": 28, "t_vol_mult": 2.0, "t_lookback": 20,
        "t_body_pct": 0.6, "t_sl_atr": 2.0, "t_min_rr": 1.5,
    },
    "BRK_MEDIUM": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 2,
        "t_adx_min": 25, "t_vol_mult": 1.5, "t_lookback": 20,
        "t_body_pct": 0.5, "t_sl_atr": 2.0, "t_min_rr": 1.5,
    },
    "BRK_LOOSE": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 2,
        "t_adx_min": 22, "t_vol_mult": 1.3, "t_lookback": 15,
        "t_body_pct": 0.4, "t_sl_atr": 2.0, "t_min_rr": 1.3,
    },
    "BRK_WIDESLR": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 2,
        "t_adx_min": 25, "t_vol_mult": 1.5, "t_lookback": 20,
        "t_body_pct": 0.5, "t_sl_atr": 2.5, "t_min_rr": 1.5,
    },
    "BRK_SHORT_LB": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 2,
        "t_adx_min": 25, "t_vol_mult": 1.5, "t_lookback": 10,
        "t_body_pct": 0.5, "t_sl_atr": 2.0, "t_min_rr": 1.5,
    },
    "BRK_3SLOTS": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 3,
        "t_adx_min": 25, "t_vol_mult": 1.5, "t_lookback": 20,
        "t_body_pct": 0.5, "t_sl_atr": 2.0, "t_min_rr": 1.5,
    },
    "BRK_HIGHVOL": {
        "rsi_oversold": 42, "rsi_overbought": 58,
        "bb_touch_pct": 0.8, "volume_range_mult": 0.6,
        "max_ranging_pos": 3, "max_trending_pos": 2,
        "t_adx_min": 25, "t_vol_mult": 2.0, "t_lookback": 20,
        "t_body_pct": 0.5, "t_sl_atr": 2.0, "t_min_rr": 1.5,
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


def scan_ranging(row, bias, p, ind_cfg):
    """Mean-reversion: buy BB lower, sell BB upper."""
    close = row["close"]
    rsi = row.get("rsi") or 0
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    vol = row.get("volume") or 0
    vsma = row.get("volume_sma") or 0
    bbu = row.get("bb_upper") or 0
    bbl = row.get("bb_lower") or 0
    bbm = row.get("bb_mid") or 0

    if atr <= 0 or adx >= 25:
        return None

    # RANGING LONG
    if bias != "bearish":
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
    if bias != "bullish":
        if (bbu > 0 and close >= bbu * (1 - p["bb_touch_pct"] / 100)
                and rsi > p["rsi_overbought"]
                and vsma > 0 and vol > vsma * p["volume_range_mult"]):
            sl = close + 1.0 * atr
            tp = bbm
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= 1.2:
                return {"side": "short", "regime": "ranging", "entry": close, "sl": sl, "tp": tp}

    return None


def scan_trending(row, prev, df_slice, bias, tp, ind_cfg):
    """Breakout + momentum: ride the wave with trailing stop, no fixed TP.

    Trending logic:
    - Breakout: close breaks above/below recent N-bar high/low
    - Momentum: strong body candle + volume surge + DI alignment
    - ADX confirms trend strength
    - Trailing SL only (chandelier), NO fixed TP — let winners run
    """
    close = row["close"]
    high = row["high"]
    low = row["low"]
    opn = row["open"]
    rsi = row.get("rsi") or 0
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    vol = row.get("volume") or 0
    vsma = row.get("volume_sma") or 0
    plus_di = row.get("plus_di") or 0
    minus_di = row.get("minus_di") or 0
    ema_f = row.get(f"ema_{ind_cfg.ema_fast}") or 0
    ema_m = row.get(f"ema_{ind_cfg.ema_mid}") or 0
    ema_s = row.get(f"ema_{ind_cfg.ema_slow}") or 0

    if atr <= 0:
        return None

    adx_min = tp.get("t_adx_min", 25)
    vol_mult = tp.get("t_vol_mult", 1.5)
    lookback = tp.get("t_lookback", 20)
    body_pct = tp.get("t_body_pct", 0.5)
    sl_atr = tp.get("t_sl_atr", 2.0)
    min_rr = tp.get("t_min_rr", 1.5)

    if adx < adx_min:
        return None
    if vsma <= 0 or vol < vsma * vol_mult:
        return None

    # Body strength: candle body > X% of total range
    candle_range = high - low
    if candle_range <= 0:
        return None
    body = abs(close - opn)
    if body / candle_range < body_pct:
        return None

    # Recent high/low for breakout detection
    if len(df_slice) < lookback:
        return None
    recent = df_slice.iloc[-lookback:]
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    # ═══ TRENDING LONG ═══
    if (plus_di > minus_di
            and close > recent_high  # breakout above recent high
            and close > opn  # bullish candle
            and ema_f > ema_m  # short-term momentum
            and rsi > 50 and rsi < 80  # momentum but not overbought
            and (bias == "bullish" or bias == "neutral")):
        sl = close - sl_atr * atr
        tp_price = close + sl_atr * min_rr * atr  # used for R:R check only
        return {"side": "long", "regime": "trending", "entry": close,
                "sl": sl, "tp": tp_price}

    # ═══ TRENDING SHORT ═══
    if (minus_di > plus_di
            and close < recent_low  # breakout below recent low
            and close < opn  # bearish candle
            and ema_f < ema_m  # short-term momentum
            and rsi < 50 and rsi > 20  # momentum but not oversold
            and (bias == "bearish" or bias == "neutral")):
        sl = close + sl_atr * atr
        tp_price = close - sl_atr * min_rr * atr
        return {"side": "short", "regime": "trending", "entry": close,
                "sl": sl, "tp": tp_price}

    return None


def check_exit(t, row, candles, is_ranging):
    h, l = row["high"], row["low"]
    # SL always checked
    if t.side == "long" and l <= t.sl_price:
        return t.sl_price, "SL"
    if t.side == "short" and h >= t.sl_price:
        return t.sl_price, "SL"

    if is_ranging:
        # Ranging: fixed TP + time exit
        if t.side == "long" and h >= t.tp_price:
            return t.tp_price, "TP"
        if t.side == "short" and l <= t.tp_price:
            return t.tp_price, "TP"
        if candles >= 15:
            return row["close"], "TIME"
    else:
        # Trending: NO fixed TP — rely on trailing SL (chandelier)
        # Only have a max hold time to prevent stuck positions
        if candles >= 60:  # 60 bars = 15h on 15m
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
    """Run simulation with given params on cached data.
    Parallel mode: ranging and trending have SEPARATE position pools.
    """
    balance = BALANCE_START
    trades = []
    open_ranging = []   # separate pool
    open_trending = []  # separate pool
    daily_loss = 0.0
    daily_date = ""
    cooldown = 0
    wins = 0
    losses = 0
    peak = BALANCE_START
    max_dd = 0.0

    max_ranging = params.get("max_ranging_pos", 3)
    max_trending = params.get("max_trending_pos", 2)

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

                # Check exits for BOTH pools
                all_open = open_ranging + open_trending
                closed = []
                for t in all_open:
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
                    if t in open_ranging:
                        open_ranging.remove(t)
                    elif t in open_trending:
                        open_trending.remove(t)

                if cooldown > 0:
                    cooldown -= 1
                    continue

                if daily_loss >= balance * 0.06:
                    continue

                bias = "neutral"
                for ht in sorted(htf_biases.keys(), reverse=True):
                    if bar_time >= ht:
                        bias = htf_biases[ht]
                        break

                # ═══ RANGING: independent scan ═══
                if (max_ranging > 0
                        and len(open_ranging) < max_ranging
                        and not any(t.symbol == symbol and t.tf == tf for t in open_ranging)):
                    rsig = scan_ranging(row, bias, params, IND_CFG)
                    if rsig:
                        amt, notional = calc_size(balance, rsig["entry"], rsig["sl"], LEVERAGE, max_ranging)
                        if amt > 0:
                            trade = Trade(symbol=symbol, side=rsig["side"], regime="ranging",
                                          entry_price=rsig["entry"], sl_price=rsig["sl"], tp_price=rsig["tp"],
                                          size=amt, notional=notional, entry_time=str(bar_time), tf=tf)
                            open_ranging.append(trade)

                # ═══ TRENDING: independent scan (different logic) ═══
                if (max_trending > 0
                        and len(open_trending) < max_trending
                        and not any(t.symbol == symbol and t.tf == tf for t in open_trending)):
                    df_slice = df_ind.iloc[max(0, i - 30):i]
                    tsig = scan_trending(row, prev, df_slice, bias, params, IND_CFG)
                    if tsig:
                        amt, notional = calc_size(balance, tsig["entry"], tsig["sl"], LEVERAGE, max_trending)
                        if amt > 0:
                            trade = Trade(symbol=symbol, side=tsig["side"], regime="trending",
                                          entry_price=tsig["entry"], sl_price=tsig["sl"], tp_price=tsig["tp"],
                                          size=amt, notional=notional, entry_time=str(bar_time), tf=tf)
                            open_trending.append(trade)

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
    trending_trades = [t for t in trades if t.regime == "trending"]
    ranging_trades = [t for t in trades if t.regime == "ranging"]
    trending_wins = sum(1 for t in trending_trades if t.pnl_net >= 0)
    ranging_wins = sum(1 for t in ranging_trades if t.pnl_net >= 0)
    trending_pnl = sum(t.pnl_net for t in trending_trades)
    ranging_pnl = sum(t.pnl_net for t in ranging_trades)

    return {
        "trades": total, "wins": wins, "losses": losses,
        "pnl": total_pnl, "wr": wr, "dd": max_dd * 100,
        "fees": total_fees, "pf": pf, "balance": balance,
        "avg_day": total / 90, "days_active": days_with_trades,
        "trending": len(trending_trades), "ranging": len(ranging_trades),
        "trending_wins": trending_wins, "ranging_wins": ranging_wins,
        "trending_pnl": trending_pnl, "ranging_pnl": ranging_pnl,
    }


async def main():
    log.info("=" * 70)
    log.info("FUTU Parameter Optimizer — 90 Days")
    log.info("=" * 70)

    ex = ccxt.okx({"options": {"defaultType": "swap"}})
    await ex.load_markets()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=92)
    since_ms = int(since.timestamp() * 1000)
    backtest_start = now - timedelta(days=90)

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
    log.info("\n" + "=" * 85)
    log.info("%-16s %5s %4s %4s %8s %5s %5s %4s %5s  %-12s %-12s",
             "SET", "TOT", "W", "L", "PNL", "WR%", "DD%", "PF", "$/DAY",
             "RANG(w/l)", "TREND(w/l)")
    log.info("-" * 85)

    results = {}
    for name, params in PARAM_SETS.items():
        r = run_sim(params, data_cache, backtest_start)
        results[name] = r
        rng = r.get("ranging", 0)
        trn = r.get("trending", 0)
        rng_w = r.get("ranging_wins", 0)
        trn_w = r.get("trending_wins", 0)
        rng_l = rng - rng_w if rng >= rng_w else 0
        trn_l = trn - trn_w if trn >= trn_w else 0
        days = r.get("days_active", 1) or 1
        pnl_day = r["pnl"] / 30
        log.info("%-16s %5d %4d %4d %+8.1f %5.1f %5.1f %4.1f %+5.1f  %3d(%d/%d)    %3d(%d/%d)",
                 name, r["trades"], r.get("wins", 0), r.get("losses", 0),
                 r["pnl"], r["wr"], r["dd"], r["pf"], pnl_day,
                 rng, rng_w, rng_l, trn, trn_w, trn_l)

    # Find best by score: PnL * WR * (1 / max(DD, 1))
    log.info("-" * 85)
    best = max(results.items(),
               key=lambda x: x[1]["pnl"] * (x[1]["wr"] / 100) / max(x[1]["dd"], 1))
    log.info("\nBEST: %s", best[0])
    r = best[1]
    log.info("  Trades: %d (%.1f/day) | PnL: $%+.2f | WR: %.1f%% | DD: %.1f%% | PF: %.1f",
             r["trades"], r["avg_day"], r["pnl"], r["wr"], r["dd"], r["pf"])
    log.info("  Fees: $%.2f | Ranging: %d | Trending: %d",
             r["fees"], r.get("ranging", 0), r.get("trending", 0))
    log.info("  Final Balance: $%.2f ($%.0f start)", r["balance"], BALANCE_START)

    bp = PARAM_SETS[best[0]]
    log.info("\n  RANGING PARAMS:")
    for k in ["rsi_oversold", "rsi_overbought", "bb_touch_pct", "volume_range_mult",
              "max_ranging_pos"]:
        log.info("    %-22s %s", k + ":", bp.get(k, "-"))
    log.info("  TRENDING PARAMS:")
    for k in ["t_adx_min", "t_vol_mult", "t_lookback", "t_body_pct",
              "t_sl_atr", "t_min_rr", "max_trending_pos"]:
        log.info("    %-22s %s", k + ":", bp.get(k, "-"))

    # Show trending PnL breakdown for all sets
    log.info("\n" + "=" * 60)
    log.info("TRENDING PnL BREAKDOWN")
    log.info("%-16s %8s %8s %8s", "SET", "RANG PnL", "TREND PnL", "TOTAL")
    log.info("-" * 60)
    for name, r in results.items():
        log.info("%-16s %+8.1f %+8.1f %+8.1f",
                 name, r.get("ranging_pnl", 0), r.get("trending_pnl", 0), r["pnl"])


if __name__ == "__main__":
    asyncio.run(main())
