"""
FUTU Dual Strategy Backtest — 60 days, OKX fees
Ranging (mean-reversion on 15m) + Trending (breakout on 1H) in parallel
Usage: python backtest.py
"""
import asyncio
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv

from src.indicators import compute_all
from src.config import IndicatorConfig, StrategyConfig, RiskConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backtest")

# ═══ OKX Fee Schedule (Tier 1) ═══
TAKER_FEE = 0.0005
ENTRY_FEE = TAKER_FEE
EXIT_FEE = TAKER_FEE

# ═══ Config ═══
IND_CFG = IndicatorConfig()
STRAT_CFG = StrategyConfig()
RISK_CFG = RiskConfig()
BALANCE_START = 300.0
LEVERAGE = 10
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAIN_TF = sys.argv[2] if len(sys.argv) > 2 else "5m"
MAX_RANGING_POS = 999   # unlimited — chỉ giới hạn 1 per symbol
MAX_TRENDING_POS = 999  # unlimited — chỉ giới hạn 1 per symbol

# Trending breakout params (BRK_WIDESLR winner)
T_ADX_MIN = 25
T_VOL_MULT = 1.5
T_LOOKBACK = 20
T_BODY_PCT = 0.5
T_SL_ATR = 2.5

SYMBOLS_TO_TEST = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "BNB/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "SUI/USDT:USDT",
]

# Trending only on top 5 volume (major, less chop)
TRENDING_SYMBOLS = {
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "BNB/USDT:USDT", "XRP/USDT:USDT",
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
    fee_entry: float = 0.0
    fee_exit: float = 0.0
    pnl_net: float = 0.0
    candles_held: int = 0


@dataclass
class BacktestState:
    balance: float = BALANCE_START
    trades: list = field(default_factory=list)
    open_ranging: list = field(default_factory=list)
    open_trending: list = field(default_factory=list)
    daily_loss: float = 0.0
    daily_date: str = ""
    cooldown: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    peak_balance: float = BALANCE_START


def detect_htf_bias(df_htf, ind_cfg):
    if len(df_htf) < ind_cfg.ema_slow + 5:
        return "neutral"
    last = df_htf.iloc[-1]
    ema_f = last.get(f"ema_{ind_cfg.ema_fast}", 0)
    ema_m = last.get(f"ema_{ind_cfg.ema_mid}", 0)
    # Simplified: EMA9 > EMA21 = bullish, EMA9 < EMA21 = bearish
    if ema_f > ema_m:
        return "bullish"
    if ema_f < ema_m:
        return "bearish"
    return "neutral"


def calc_position_size(balance, entry, sl, risk_pct, leverage):
    sl_dist = abs(entry - sl) / entry
    if sl_dist == 0:
        return 0.0, 0.0
    risk_amount = BALANCE_START * risk_pct  # Always size on STARTING balance, not compound
    pos_value = risk_amount / sl_dist
    max_notional = BALANCE_START * leverage  # Cap at starting balance * leverage
    pos_value = min(pos_value, max_notional)
    amount = pos_value / entry
    return amount, pos_value


# ═══════════════════════════════════════════════════════════════
# RANGING STRATEGY — mean-reversion at BB bands
# ═══════════════════════════════════════════════════════════════
def scan_ranging(row, bias, cfg):
    close = row["close"]
    rsi = row.get("rsi") or 0
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    vol = row.get("volume") or 0
    vsma = row.get("volume_sma") or 0
    bbu = row.get("bb_upper") or 0
    bbl = row.get("bb_lower") or 0
    bbm = row.get("bb_mid") or 0
    ema21 = row.get(f"ema_{IND_CFG.ema_mid}") or 0

    if atr <= 0 or adx >= cfg.adx_trending:
        return None

    # LONG: price touches lower BB + RSI oversold
    if bias != "bearish":
        if (bbl > 0 and close <= bbl * (1 + cfg.bb_touch_pct / 100)
                and rsi < cfg.rsi_oversold
                and vsma > 0 and vol > vsma * cfg.volume_range_mult):
            sl = close - cfg.main_sl_ranging_atr_mult * atr
            tp = bbm
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= RISK_CFG.min_rr_ranging:
                return {"side": "long", "entry": close, "sl": sl, "tp": tp}

    # SHORT: price touches upper BB + RSI overbought
    if bias != "bullish":
        if (bbu > 0 and close >= bbu * (1 - cfg.bb_touch_pct / 100)
                and rsi > cfg.rsi_overbought
                and vsma > 0 and vol > vsma * cfg.volume_range_mult):
            sl = close + cfg.main_sl_ranging_atr_mult * atr
            tp = bbm
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= RISK_CFG.min_rr_ranging:
                return {"side": "short", "entry": close, "sl": sl, "tp": tp}

    return None


# ═══════════════════════════════════════════════════════════════
# TRENDING STRATEGY — breakout + momentum, trailing SL only
# ═══════════════════════════════════════════════════════════════
def scan_trending(row, prev_row, df_slice, bias):
    """Breakout + momentum on 1H:
    - ADX >= 25 and not falling (trend building)
    - Volume surge >= 1.5x SMA
    - Strong body candle (>50% of range)
    - Close breaks above/below recent 20-bar high/low
    - DI alignment + EMA momentum
    - SL = 2.5 ATR, trailing via chandelier, no fixed TP
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
    ema_f = row.get(f"ema_{IND_CFG.ema_fast}") or 0
    ema_m = row.get(f"ema_{IND_CFG.ema_mid}") or 0

    prev_adx = prev_row.get("adx") or 0

    if atr <= 0 or adx < T_ADX_MIN:
        return None
    if vsma <= 0 or vol < vsma * T_VOL_MULT:
        return None

    # ADX must not be falling (trend not fading)
    if adx < prev_adx - 1:
        return None

    # Body strength check
    candle_range = high - low
    if candle_range <= 0:
        return None
    body = abs(close - opn)
    if body / candle_range < T_BODY_PCT:
        return None

    # Recent high/low for breakout
    if len(df_slice) < T_LOOKBACK:
        return None
    recent = df_slice.iloc[-T_LOOKBACK:]
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    # BREAKOUT LONG
    if (plus_di > minus_di
            and close > recent_high          # breaking above recent range
            and close > opn                  # bullish candle
            and ema_f > ema_m
            and rsi > 50 and rsi < 80
            and bias in ("bullish", "neutral")):
        sl = close - T_SL_ATR * atr
        tp = close + T_SL_ATR * 1.5 * atr
        return {"side": "long", "entry": close, "sl": sl, "tp": tp}

    # BREAKOUT SHORT
    if (minus_di > plus_di
            and close < recent_low           # breaking below recent range
            and close < opn                  # bearish candle
            and ema_f < ema_m
            and rsi < 50 and rsi > 20
            and bias in ("bearish", "neutral")):
        sl = close + T_SL_ATR * atr
        tp = close - T_SL_ATR * 1.5 * atr
        return {"side": "short", "entry": close, "sl": sl, "tp": tp}

    return None


def check_exit(trade, row, candles_held):
    high = row["high"]
    low = row["low"]

    # SL always
    if trade.side == "long" and low <= trade.sl_price:
        return trade.sl_price, "SL"
    if trade.side == "short" and high >= trade.sl_price:
        return trade.sl_price, "SL"

    if trade.regime == "ranging":
        # Fixed TP + time exit
        if trade.side == "long" and high >= trade.tp_price:
            return trade.tp_price, "TP"
        if trade.side == "short" and low <= trade.tp_price:
            return trade.tp_price, "TP"
        if candles_held >= STRAT_CFG.ranging_max_candles:
            return row["close"], "TIME"
    else:
        # Trending on 1H: trailing SL only, max hold 48 bars (48h = 2 days)
        if candles_held >= 48:
            return row["close"], "TIME"

    return None


def update_trailing_sl(trade, row):
    chand_long = row.get("chandelier_long")
    chand_short = row.get("chandelier_short")
    if trade.side == "long" and chand_long:
        if chand_long > trade.entry_price and chand_long > trade.sl_price:
            trade.sl_price = chand_long
    elif trade.side == "short" and chand_short:
        if chand_short < trade.entry_price and chand_short < trade.sl_price:
            trade.sl_price = chand_short


async def fetch_ohlcv(ex, symbol, tf, since_ms, limit=300):
    all_candles = []
    current = since_ms
    while True:
        try:
            data = await ex.fetch_ohlcv(symbol, tf, since=current, limit=limit)
        except Exception as e:
            log.warning("Fetch error %s %s: %s", symbol, tf, e)
            break
        if not data:
            break
        all_candles.extend(data)
        if len(data) < limit:
            break
        current = data[-1][0] + 1
        await asyncio.sleep(0.15)
    return all_candles


def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    return df


async def run_backtest():
    log.info("=" * 60)
    log.info("FUTU Dual Strategy Backtest — %d Days — OKX Fees", DAYS)
    log.info("Balance: $%.0f | Leverage: %dx | Fee: %.2f%%",
             BALANCE_START, LEVERAGE, TAKER_FEE * 100)
    log.info("Ranging TF: %s | Trending: 1H", MAIN_TF)
    log.info("=" * 60)

    ex = ccxt.okx({"options": {"defaultType": "swap"}})
    await ex.load_markets()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=DAYS + 5)
    since_ms = int(since.timestamp() * 1000)
    backtest_start = now - timedelta(days=DAYS)

    state = BacktestState()

    for symbol in SYMBOLS_TO_TEST:
        if symbol not in ex.markets:
            log.info("SKIP %s — not on OKX", symbol.split("/")[0])
            continue

        short = symbol.split("/")[0]
        log.info("\n--- %s ---", short)

        raw_15m = await fetch_ohlcv(ex, symbol, "15m", since_ms)
        raw_5m = await fetch_ohlcv(ex, symbol, "5m", since_ms)
        raw_4h = await fetch_ohlcv(ex, symbol, "4h", since_ms)
        is_trending_sym = symbol in TRENDING_SYMBOLS
        raw_1h = await fetch_ohlcv(ex, symbol, "1h", since_ms) if is_trending_sym else []

        if len(raw_15m) < 200 or len(raw_5m) < 200 or len(raw_4h) < 20:
            log.info("SKIP %s — not enough data", short)
            continue
        if is_trending_sym and len(raw_1h) < 50:
            log.info("SKIP %s trending — not enough 1H data", short)
            is_trending_sym = False

        df_15m = ohlcv_to_df(raw_15m)
        df_5m = ohlcv_to_df(raw_5m)
        df_1h = ohlcv_to_df(raw_1h) if is_trending_sym else None
        df_4h = ohlcv_to_df(raw_4h)

        candles_15m = [{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                        "low": r["low"], "close": r["close"], "volume": r["volume"]}
                       for ts, r in df_15m.iterrows()]
        candles_5m = [{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                       "low": r["low"], "close": r["close"], "volume": r["volume"]}
                      for ts, r in df_5m.iterrows()]
        candles_1h = ([{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                        "low": r["low"], "close": r["close"], "volume": r["volume"]}
                       for ts, r in df_1h.iterrows()] if is_trending_sym else [])
        candles_1h = list(candles_1h)
        candles_4h = [{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                       "low": r["low"], "close": r["close"], "volume": r["volume"]}
                      for ts, r in df_4h.iterrows()]

        df_15m_ind = compute_all(candles_15m, IND_CFG)
        df_5m_ind = compute_all(candles_5m, IND_CFG)
        df_1h_ind = compute_all(candles_1h, IND_CFG) if is_trending_sym else None
        df_4h_ind = compute_all(candles_4h, IND_CFG)

        # H4 bias lookup
        htf_biases = {}
        for i in range(IND_CFG.ema_slow + 5, len(df_4h_ind)):
            chunk = df_4h_ind.iloc[:i + 1]
            bias = detect_htf_bias(chunk, IND_CFG)
            htf_biases[df_4h_ind.index[i]] = bias

        sym_ranging = 0
        sym_trending = 0

        def get_bias(bar_time):
            bias = "neutral"
            for htf_time in sorted(htf_biases.keys(), reverse=True):
                if bar_time >= htf_time:
                    bias = htf_biases[htf_time]
                    break
            return bias

        def close_trade(t, exit_price, reason, bar_time):
            t.exit_price = exit_price
            t.exit_time = str(bar_time)
            t.exit_reason = reason
            if t.side == "long":
                t.pnl_gross = (t.exit_price - t.entry_price) * t.size
            else:
                t.pnl_gross = (t.entry_price - t.exit_price) * t.size
            t.fee_entry = t.notional * ENTRY_FEE
            t.fee_exit = abs(t.exit_price * t.size) * EXIT_FEE
            t.pnl_net = t.pnl_gross - t.fee_entry - t.fee_exit
            state.balance += t.pnl_net
            if t.pnl_net >= 0:
                state.wins += 1
            else:
                state.losses += 1
                state.daily_loss += abs(t.pnl_net)
                state.cooldown = RISK_CFG.cooldown_candles
            state.trades.append(t)
            if state.balance > state.peak_balance:
                state.peak_balance = state.balance
            dd = (state.peak_balance - state.balance) / state.peak_balance
            if dd > state.max_drawdown:
                state.max_drawdown = dd

        # ═══ Ranging: 15m detect + 5m confirm ═══
        # Build 15m signals with entry params: {timestamp -> signal_dict}
        flags_15m = {}
        for i in range(IND_CFG.ema_slow + 5, len(df_15m_ind)):
            bar_time = df_15m_ind.index[i]
            if bar_time.replace(tzinfo=timezone.utc) < backtest_start:
                continue
            row = df_15m_ind.iloc[i]
            bias = get_bias(bar_time)
            rsig = scan_ranging(row, bias, STRAT_CFG)
            if rsig:
                flags_15m[bar_time] = rsig  # includes side, entry, sl, tp

        # Track which flags have been used (1 trade per flag)
        used_flags = set()

        # Iterate 5m bars — exits on 5m, entry only when 15m flagged + 5m confirms
        for i in range(IND_CFG.ema_slow + 5, len(df_5m_ind)):
            bar_time = df_5m_ind.index[i]
            if bar_time.replace(tzinfo=timezone.utc) < backtest_start:
                continue
            row = df_5m_ind.iloc[i]

            day_str = bar_time.strftime("%Y-%m-%d")
            if state.daily_date != day_str:
                state.daily_loss = 0.0
                state.daily_date = day_str

            # Exit check on 5m bars
            closed = []
            for t in state.open_ranging:
                if t.symbol != symbol:
                    continue
                t.candles_held += 1
                update_trailing_sl(t, row)
                result = check_exit(t, row, t.candles_held)
                if result:
                    close_trade(t, result[0], result[1], bar_time)
                    closed.append(t)
            for t in closed:
                state.open_ranging.remove(t)

            if state.daily_loss >= state.balance * RISK_CFG.max_daily_loss_pct:
                continue
            if any(t.symbol == symbol for t in state.open_ranging):
                continue

            # Find active 15m flag (not yet used)
            active_flag = None
            active_flag_time = None
            for flag_time, sig in flags_15m.items():
                if flag_time in used_flags:
                    continue
                diff = (bar_time - flag_time).total_seconds()
                if 0 <= diff < 900:  # within 15 min window
                    active_flag = sig
                    active_flag_time = flag_time
                    break

            if active_flag is None:
                continue

            # 5m confirm: must agree with 15m direction
            rsi_5m = row.get("rsi") or 50
            close_5m = row["close"]
            bb_mid_5m = row.get("bb_mid") or close_5m
            confirmed = False

            if active_flag["side"] == "long" and (close_5m <= bb_mid_5m or rsi_5m < 45):
                confirmed = True
            elif active_flag["side"] == "short" and (close_5m >= bb_mid_5m or rsi_5m > 55):
                confirmed = True

            if not confirmed:
                continue

            # Use 5m price for entry, 15m signal for SL/TP
            entry = close_5m
            sl = active_flag["sl"]
            tp = active_flag["tp"]
            amt, notional = calc_position_size(
                state.balance, entry, sl, RISK_CFG.risk_per_trade_main, LEVERAGE)
            if amt > 0:
                trade = Trade(symbol=symbol, side=active_flag["side"], regime="ranging",
                              entry_price=entry, sl_price=sl, tp_price=tp,
                              size=amt, notional=notional, entry_time=str(bar_time))
                state.open_ranging.append(trade)
                sym_ranging += 1
                used_flags.add(active_flag_time)  # mark flag as used

        # ═══ Trending on 1H (top volume symbols only) ═══
        if symbol not in TRENDING_SYMBOLS:
            log.info("%s: %d ranging + 0 trending (skip)", short, sym_ranging)
            await asyncio.sleep(0.2)
            continue

        for i in range(IND_CFG.ema_slow + 5, len(df_1h_ind)):
            bar_time = df_1h_ind.index[i]
            if bar_time.replace(tzinfo=timezone.utc) < backtest_start:
                continue
            row = df_1h_ind.iloc[i]
            prev_row = df_1h_ind.iloc[i - 1]

            day_str = bar_time.strftime("%Y-%m-%d")
            if state.daily_date != day_str:
                state.daily_loss = 0.0
                state.daily_date = day_str

            # Exit check for trending positions (on 1H bars)
            closed = []
            for t in state.open_trending:
                if t.symbol != symbol:
                    continue
                t.candles_held += 1
                update_trailing_sl(t, row)
                result = check_exit(t, row, t.candles_held)
                if result:
                    close_trade(t, result[0], result[1], bar_time)
                    closed.append(t)
            for t in closed:
                state.open_trending.remove(t)

            if state.daily_loss >= state.balance * RISK_CFG.max_daily_loss_pct:
                continue

            bias = get_bias(bar_time)
            if not any(t.symbol == symbol for t in state.open_trending):
                df_slice = df_1h_ind.iloc[max(0, i - T_LOOKBACK - 5):i]
                tsig = scan_trending(row, prev_row, df_slice, bias)
                if tsig:
                    amt, notional = calc_position_size(
                        state.balance, tsig["entry"], tsig["sl"],
                        RISK_CFG.risk_per_trade_main, LEVERAGE)
                    if amt > 0:
                        trade = Trade(symbol=symbol, side=tsig["side"], regime="trending",
                                      entry_price=tsig["entry"], sl_price=tsig["sl"],
                                      tp_price=tsig["tp"], size=amt, notional=notional,
                                      entry_time=str(bar_time))
                        state.open_trending.append(trade)
                        sym_trending += 1

        log.info("%s: %d ranging + %d trending (1H)", short, sym_ranging, sym_trending)
        await asyncio.sleep(0.2)

    await ex.close()
    print_results(state)


def print_results(state):
    trades = state.trades
    total = len(trades)

    log.info("\n" + "=" * 60)
    log.info("BACKTEST RESULTS — %d DAYS (Dual Strategy)", DAYS)
    log.info("=" * 60)

    if total == 0:
        log.info("No trades executed.")
        return

    total_pnl_gross = sum(t.pnl_gross for t in trades)
    total_pnl_net = sum(t.pnl_net for t in trades)
    total_fees = sum(t.fee_entry + t.fee_exit for t in trades)
    win_rate = state.wins / total * 100

    winners = [t for t in trades if t.pnl_net >= 0]
    losers = [t for t in trades if t.pnl_net < 0]
    avg_win = sum(t.pnl_net for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.pnl_net for t in losers) / len(losers) if losers else 0
    avg_rr = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    profit_factor = sum(t.pnl_net for t in winners) / abs(sum(t.pnl_net for t in losers)) if losers else float('inf')
    avg_hold = sum(t.candles_held for t in trades) / total

    trending_trades = [t for t in trades if t.regime == "trending"]
    ranging_trades = [t for t in trades if t.regime == "ranging"]

    log.info("")
    log.info("OVERVIEW")
    log.info("  Starting Balance:  $%.2f", BALANCE_START)
    log.info("  Final Balance:     $%.2f", state.balance)
    log.info("  Net PnL:           $%+.2f (%.1f%%)", total_pnl_net, total_pnl_net / BALANCE_START * 100)
    log.info("  Gross PnL:         $%.2f", total_pnl_gross)
    log.info("  Total Fees:        -$%.2f", total_fees)
    log.info("  Max Drawdown:      %.1f%%", state.max_drawdown * 100)
    log.info("")
    log.info("TRADES")
    log.info("  Total:             %d (%.1f/day)", total, total / DAYS)
    log.info("  Winners:           %d (%.1f%%)", state.wins, win_rate)
    log.info("  Losers:            %d", state.losses)
    log.info("  Avg Win:           $%.2f", avg_win)
    log.info("  Avg Loss:          $%.2f", avg_loss)
    log.info("  Avg R:R:           %.2f", avg_rr)
    log.info("  Profit Factor:     %.2f", profit_factor)
    log.info("  Best Trade:        $%.2f", max((t.pnl_net for t in trades), default=0))
    log.info("  Worst Trade:       $%.2f", min((t.pnl_net for t in trades), default=0))
    log.info("  Avg Hold:          %.1f candles", avg_hold)

    log.info("")
    log.info("BY REGIME")
    if ranging_trades:
        rw = sum(1 for t in ranging_trades if t.pnl_net >= 0)
        rp = sum(t.pnl_net for t in ranging_trades)
        log.info("  Ranging:   %d trades | %d wins (%.0f%%) | PnL $%+.2f",
                 len(ranging_trades), rw, rw / len(ranging_trades) * 100, rp)
    if trending_trades:
        tw = sum(1 for t in trending_trades if t.pnl_net >= 0)
        tp = sum(t.pnl_net for t in trending_trades)
        log.info("  Trending:  %d trades | %d wins (%.0f%%) | PnL $%+.2f",
                 len(trending_trades), tw, tw / len(trending_trades) * 100, tp)
        # Trending detail
        for t in trending_trades:
            emoji = "W" if t.pnl_net >= 0 else "L"
            log.info("    [%s] %s %-5s entry=%.2f exit=%.2f pnl=$%+.2f (%s) [%d bars]",
                     emoji, t.symbol.split("/")[0], t.side.upper(),
                     t.entry_price, t.exit_price, t.pnl_net, t.exit_reason, t.candles_held)

    sym_pnl = {}
    for t in trades:
        s = t.symbol.split("/")[0]
        sym_pnl[s] = sym_pnl.get(s, 0) + t.pnl_net
    log.info("")
    log.info("BY SYMBOL")
    for s, pnl in sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True):
        cnt = sum(1 for t in trades if t.symbol.split("/")[0] == s)
        r_cnt = sum(1 for t in ranging_trades if t.symbol.split("/")[0] == s)
        t_cnt = sum(1 for t in trending_trades if t.symbol.split("/")[0] == s)
        log.info("  %-6s %3d trades (R:%d T:%d) | PnL $%+.2f", s, cnt, r_cnt, t_cnt, pnl)

    log.info("")
    log.info("FEE BREAKDOWN")
    log.info("  Entry fees:  $%.2f", sum(t.fee_entry for t in trades))
    log.info("  Exit fees:   $%.2f", sum(t.fee_exit for t in trades))
    log.info("  Total:       $%.2f (%.1f%% of gross PnL)", total_fees,
             abs(total_fees / total_pnl_gross * 100) if total_pnl_gross != 0 else 0)

    log.info("")
    log.info("TOP 5 TRADES")
    for t in sorted(trades, key=lambda x: x.pnl_net, reverse=True)[:5]:
        log.info("  %s %-5s %-8s entry=%.2f exit=%.2f pnl=$%+.2f (%s) [%d bars]",
                 t.symbol.split("/")[0], t.side.upper(), t.regime,
                 t.entry_price, t.exit_price, t.pnl_net, t.exit_reason, t.candles_held)

    log.info("")
    log.info("WORST 5 TRADES")
    for t in sorted(trades, key=lambda x: x.pnl_net)[:5]:
        log.info("  %s %-5s %-8s entry=%.2f exit=%.2f pnl=$%+.2f (%s) [%d bars]",
                 t.symbol.split("/")[0], t.side.upper(), t.regime,
                 t.entry_price, t.exit_price, t.pnl_net, t.exit_reason, t.candles_held)

    log.info("")
    log.info("DAILY PNL")
    daily = {}
    for t in trades:
        day = t.exit_time[:10]
        daily[day] = daily.get(day, 0) + t.pnl_net
    for day in sorted(daily.keys()):
        bar = "+" * int(abs(daily[day]) / 0.5) if daily[day] >= 0 else "-" * int(abs(daily[day]) / 0.5)
        log.info("  %s  $%+7.2f  %s", day, daily[day], bar[:30])


if __name__ == "__main__":
    asyncio.run(run_backtest())
