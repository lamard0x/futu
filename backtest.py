"""
FUTU Strategy Backtest — 30 days, OKX demo fees
Usage: python backtest.py
"""
import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv
import os

from src.indicators import compute_all
from src.config import IndicatorConfig, StrategyConfig, RiskConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backtest")

# ═══ OKX Fee Schedule (Tier 1) ═══
TAKER_FEE = 0.0005   # 0.05%
MAKER_FEE = 0.0002   # 0.02%
# Market orders = taker, TP/SL triggers = taker
ENTRY_FEE = TAKER_FEE
EXIT_FEE = TAKER_FEE

# ═══ Config ═══
IND_CFG = IndicatorConfig()
STRAT_CFG = StrategyConfig()
RISK_CFG = RiskConfig()
BALANCE_START = 300.0
LEVERAGE = 10
SYMBOLS_TO_TEST = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                   "BNB/USDT:USDT", "XAU/USDT:USDT", "AXS/USDT:USDT",
                   "FIL/USDT:USDT", "ICP/USDT:USDT", "BCH/USDT:USDT",
                   "YFI/USDT:USDT"]


@dataclass
class Trade:
    symbol: str
    side: str  # "long" or "short"
    regime: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float  # in coins
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
    open_trades: list = field(default_factory=list)
    daily_loss: float = 0.0
    daily_date: str = ""
    cooldown: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    peak_balance: float = BALANCE_START


def detect_htf_bias(df_htf, ind_cfg):
    """Compute H4 bias from H4 dataframe."""
    if len(df_htf) < ind_cfg.ema_slow + 5:
        return "neutral"
    last = df_htf.iloc[-1]
    ema_f = last.get(f"ema_{ind_cfg.ema_fast}", 0)
    ema_m = last.get(f"ema_{ind_cfg.ema_mid}", 0)
    ema_s = last.get(f"ema_{ind_cfg.ema_slow}", 0)
    adx = last.get("adx", 0) or 0
    if ema_f > ema_m > ema_s and adx > 20:
        return "bullish"
    if ema_f < ema_m < ema_s and adx > 20:
        return "bearish"
    return "neutral"


def calc_position_size(balance, entry, sl, risk_pct, leverage, max_positions):
    sl_dist = abs(entry - sl) / entry
    if sl_dist == 0:
        return 0.0, 0.0
    risk_amount = balance * risk_pct
    pos_value = risk_amount / sl_dist
    max_notional = (balance * leverage) / max(max_positions, 1)
    pos_value = min(pos_value, max_notional)
    amount = pos_value / entry
    return amount, pos_value


def check_rr(entry, sl, tp, min_rr):
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return False, 0
    rr = reward / risk
    return rr >= min_rr, rr


def scan_bar(row, prev_row, bias, cfg):
    """Check for entry signals on a single 15m bar. Returns signal dict or None."""
    close = row["close"]
    high = row["high"]
    low = row["low"]
    rsi = row.get("rsi") or 0
    prev_rsi = prev_row.get("rsi") or 0
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    volume = row.get("volume") or 0
    vol_sma = row.get("volume_sma") or 0
    bb_upper = row.get("bb_upper") or 0
    bb_lower = row.get("bb_lower") or 0
    bb_mid = row.get("bb_mid") or 0
    vwap = row.get("vwap") or 0
    ema21 = row.get(f"ema_{IND_CFG.ema_mid}") or 0

    if atr <= 0:
        return None

    trending = adx >= cfg.adx_trending

    # ═══ RANGING LONG ═══
    if not trending and bias != "bearish":
        if (bb_lower > 0 and close <= bb_lower * (1 + cfg.bb_touch_pct / 100)
                and rsi < cfg.rsi_oversold
                and vol_sma > 0 and volume > vol_sma * cfg.volume_range_mult):
            entry = close
            sl = entry - cfg.main_sl_ranging_atr_mult * atr
            tp = bb_mid
            ok, rr = check_rr(entry, sl, tp, RISK_CFG.min_rr_ranging)
            if ok:
                return {"side": "long", "regime": "ranging", "entry": entry,
                        "sl": sl, "tp": tp, "rr": rr, "source": "main"}

    # ═══ RANGING SHORT ═══
    if not trending and bias != "bullish":
        if (bb_upper > 0 and close >= bb_upper * (1 - cfg.bb_touch_pct / 100)
                and rsi > cfg.rsi_overbought
                and vol_sma > 0 and volume > vol_sma * cfg.volume_range_mult):
            entry = close
            sl = entry + cfg.main_sl_ranging_atr_mult * atr
            tp = bb_mid
            ok, rr = check_rr(entry, sl, tp, RISK_CFG.min_rr_ranging)
            if ok:
                return {"side": "short", "regime": "ranging", "entry": entry,
                        "sl": sl, "tp": tp, "rr": rr, "source": "main"}

    # ═══ TRENDING LONG ═══
    if trending and bias == "bullish":
        pullback = (low <= ema21 * 1.005) or (vwap > 0 and low <= vwap * 1.005)
        if (vwap > 0 and close > vwap
                and rsi > 45 and rsi > prev_rsi
                and vol_sma > 0 and volume > vol_sma * cfg.volume_trend_mult
                and pullback):
            entry = close
            sl = entry - cfg.main_sl_trending_atr_mult * atr
            tp1 = entry + cfg.main_tp1_atr_mult * atr
            tp2 = entry + cfg.main_tp2_atr_mult * atr
            tp = tp2  # bot uses TP2 for trending
            ok, rr = check_rr(entry, sl, tp, RISK_CFG.min_rr_trending)
            if ok:
                return {"side": "long", "regime": "trending", "entry": entry,
                        "sl": sl, "tp": tp, "rr": rr, "source": "main"}

    # ═══ TRENDING SHORT ═══
    if trending and bias == "bearish":
        pullback = (high >= ema21 * 0.995) or (vwap > 0 and high >= vwap * 0.995)
        if (vwap > 0 and close < vwap
                and rsi < 55 and rsi < prev_rsi
                and vol_sma > 0 and volume > vol_sma * cfg.volume_trend_mult
                and pullback):
            entry = close
            sl = entry + cfg.main_sl_trending_atr_mult * atr
            tp1 = entry - cfg.main_tp1_atr_mult * atr
            tp2 = entry - cfg.main_tp2_atr_mult * atr
            tp = tp2
            ok, rr = check_rr(entry, sl, tp, RISK_CFG.min_rr_trending)
            if ok:
                return {"side": "short", "regime": "trending", "entry": entry,
                        "sl": sl, "tp": tp, "rr": rr, "source": "main"}

    return None


def check_exit(trade, row, candles_held, cfg):
    """Check if trade should exit. Returns (exit_price, reason) or None."""
    high = row["high"]
    low = row["low"]

    # SL hit
    if trade.side == "long" and low <= trade.sl_price:
        return trade.sl_price, "SL"
    if trade.side == "short" and high >= trade.sl_price:
        return trade.sl_price, "SL"

    # TP hit
    if trade.side == "long" and high >= trade.tp_price:
        return trade.tp_price, "TP"
    if trade.side == "short" and low <= trade.tp_price:
        return trade.tp_price, "TP"

    # Time exit (ranging only)
    if trade.regime == "ranging" and candles_held >= cfg.ranging_max_candles:
        return row["close"], "TIME"

    return None


def update_trailing_sl(trade, row):
    """Update trailing SL using chandelier exit."""
    chand_long = row.get("chandelier_long")
    chand_short = row.get("chandelier_short")

    if trade.side == "long" and chand_long:
        if chand_long > trade.entry_price and chand_long > trade.sl_price:
            trade.sl_price = chand_long
    elif trade.side == "short" and chand_short:
        if chand_short < trade.entry_price and chand_short < trade.sl_price:
            trade.sl_price = chand_short


async def fetch_ohlcv(ex, symbol, tf, since_ms, limit=1000):
    """Fetch candles in chunks."""
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
    log.info("FUTU Backtest — 30 Days — OKX Fees")
    log.info("Balance: $%.0f | Leverage: %dx | Fee: %.2f%%/%.2f%%",
             BALANCE_START, LEVERAGE, TAKER_FEE * 100, MAKER_FEE * 100)
    log.info("=" * 60)

    # Connect to OKX (public data, no auth needed)
    ex = ccxt.okx({"options": {"defaultType": "swap"}})
    await ex.load_markets()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=32)  # extra 2 days for indicator warmup
    since_ms = int(since.timestamp() * 1000)
    backtest_start = now - timedelta(days=30)

    state = BacktestState()

    for symbol in SYMBOLS_TO_TEST:
        if symbol not in ex.markets:
            log.info("SKIP %s — not on OKX", symbol.split("/")[0])
            continue

        short = symbol.split("/")[0]
        log.info("\n--- %s ---", short)

        # Fetch 15m and 4h data
        raw_15m = await fetch_ohlcv(ex, symbol, "15m", since_ms)
        raw_4h = await fetch_ohlcv(ex, symbol, "4h", since_ms)

        if len(raw_15m) < 100 or len(raw_4h) < 20:
            log.info("SKIP %s — not enough data (15m=%d, 4h=%d)", short, len(raw_15m), len(raw_4h))
            continue

        df_15m = ohlcv_to_df(raw_15m)
        df_4h = ohlcv_to_df(raw_4h)

        # Compute indicators
        candles_15m = [{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                        "low": r["low"], "close": r["close"], "volume": r["volume"]}
                       for ts, r in df_15m.iterrows()]
        candles_4h = [{"timestamp": int(ts.timestamp() * 1000), "open": r["open"], "high": r["high"],
                       "low": r["low"], "close": r["close"], "volume": r["volume"]}
                      for ts, r in df_4h.iterrows()]

        df_15m_ind = compute_all(candles_15m, IND_CFG)
        df_4h_ind = compute_all(candles_4h, IND_CFG)

        # Build H4 bias lookup (timestamp -> bias)
        htf_biases = {}
        for i in range(IND_CFG.ema_slow + 5, len(df_4h_ind)):
            chunk = df_4h_ind.iloc[:i + 1]
            bias = detect_htf_bias(chunk, IND_CFG)
            htf_biases[df_4h_ind.index[i]] = bias

        # Simulate bar-by-bar on 15m
        sym_trades = 0
        for i in range(IND_CFG.ema_slow + 5, len(df_15m_ind)):
            bar_time = df_15m_ind.index[i]
            if bar_time.replace(tzinfo=timezone.utc) < backtest_start:
                continue

            row = df_15m_ind.iloc[i]
            prev_row = df_15m_ind.iloc[i - 1]

            # Check daily loss reset
            day_str = bar_time.strftime("%Y-%m-%d")
            if state.daily_date != day_str:
                state.daily_loss = 0.0
                state.daily_date = day_str

            # Check exits for open trades
            closed_trades = []
            for t in state.open_trades:
                if t.symbol != symbol:
                    continue
                t.candles_held += 1
                update_trailing_sl(t, row)
                result = check_exit(t, row, t.candles_held, STRAT_CFG)
                if result:
                    exit_price, reason = result
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
                    closed_trades.append(t)
                    # Track drawdown
                    if state.balance > state.peak_balance:
                        state.peak_balance = state.balance
                    dd = (state.peak_balance - state.balance) / state.peak_balance
                    if dd > state.max_drawdown:
                        state.max_drawdown = dd

            for t in closed_trades:
                state.open_trades.remove(t)

            # Cooldown tick
            if state.cooldown > 0:
                state.cooldown -= 1
                continue

            # Daily loss cap
            max_daily = state.balance * RISK_CFG.max_daily_loss_pct
            if state.daily_loss >= max_daily:
                continue

            # Max positions check
            open_count = len(state.open_trades)
            if open_count >= RISK_CFG.max_positions:
                continue

            # Already has position in this symbol?
            if any(t.symbol == symbol for t in state.open_trades):
                continue

            # Get H4 bias for current bar
            bias = "neutral"
            for htf_time in sorted(htf_biases.keys(), reverse=True):
                if bar_time >= htf_time:
                    bias = htf_biases[htf_time]
                    break

            # Check for signal
            signal = scan_bar(row, prev_row, bias, STRAT_CFG)
            if signal is None:
                continue

            # Position sizing
            amount, notional = calc_position_size(
                state.balance, signal["entry"], signal["sl"],
                RISK_CFG.risk_per_trade_main, LEVERAGE, RISK_CFG.max_positions,
            )
            if amount <= 0:
                continue

            trade = Trade(
                symbol=symbol,
                side=signal["side"],
                regime=signal["regime"],
                entry_price=signal["entry"],
                sl_price=signal["sl"],
                tp_price=signal["tp"],
                size=amount,
                notional=notional,
                entry_time=str(bar_time),
            )
            state.open_trades.append(trade)
            sym_trades += 1

        log.info("%s: %d signals found", short, sym_trades)
        await asyncio.sleep(0.2)

    await ex.close()

    # ═══ RESULTS ═══
    print_results(state)


def print_results(state):
    trades = state.trades
    total = len(trades)

    log.info("\n" + "=" * 60)
    log.info("BACKTEST RESULTS — 30 DAYS")
    log.info("=" * 60)

    if total == 0:
        log.info("No trades executed.")
        return

    total_pnl_gross = sum(t.pnl_gross for t in trades)
    total_pnl_net = sum(t.pnl_net for t in trades)
    total_fees = sum(t.fee_entry + t.fee_exit for t in trades)
    win_rate = state.wins / total * 100 if total > 0 else 0

    avg_win = 0
    avg_loss = 0
    winners = [t for t in trades if t.pnl_net >= 0]
    losers = [t for t in trades if t.pnl_net < 0]
    if winners:
        avg_win = sum(t.pnl_net for t in winners) / len(winners)
    if losers:
        avg_loss = sum(t.pnl_net for t in losers) / len(losers)

    avg_rr = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    profit_factor = sum(t.pnl_net for t in winners) / abs(sum(t.pnl_net for t in losers)) if losers else float('inf')

    avg_hold = sum(t.candles_held for t in trades) / total
    max_win = max((t.pnl_net for t in trades), default=0)
    max_loss = min((t.pnl_net for t in trades), default=0)

    # By regime
    trending_trades = [t for t in trades if t.regime == "trending"]
    ranging_trades = [t for t in trades if t.regime == "ranging"]

    # By symbol
    sym_pnl = {}
    for t in trades:
        s = t.symbol.split("/")[0]
        sym_pnl[s] = sym_pnl.get(s, 0) + t.pnl_net

    log.info("")
    log.info("OVERVIEW")
    log.info("  Starting Balance:  $%.2f", BALANCE_START)
    log.info("  Final Balance:     $%.2f", state.balance)
    log.info("  Net PnL:           $%.2f (%.1f%%)", total_pnl_net, total_pnl_net / BALANCE_START * 100)
    log.info("  Gross PnL:         $%.2f", total_pnl_gross)
    log.info("  Total Fees:        -$%.2f", total_fees)
    log.info("  Max Drawdown:      %.1f%%", state.max_drawdown * 100)
    log.info("")
    log.info("TRADES")
    log.info("  Total:             %d", total)
    log.info("  Winners:           %d (%.1f%%)", state.wins, win_rate)
    log.info("  Losers:            %d", state.losses)
    log.info("  Avg Win:           $%.2f", avg_win)
    log.info("  Avg Loss:          $%.2f", avg_loss)
    log.info("  Avg R:R:           %.2f", avg_rr)
    log.info("  Profit Factor:     %.2f", profit_factor)
    log.info("  Best Trade:        $%.2f", max_win)
    log.info("  Worst Trade:       $%.2f", max_loss)
    log.info("  Avg Hold:          %.1f candles (%.0f min)", avg_hold, avg_hold * 15)
    log.info("")
    log.info("BY REGIME")
    if trending_trades:
        tw = sum(1 for t in trending_trades if t.pnl_net >= 0)
        log.info("  Trending:  %d trades | %d wins (%.0f%%) | PnL $%.2f",
                 len(trending_trades), tw, tw / len(trending_trades) * 100,
                 sum(t.pnl_net for t in trending_trades))
    if ranging_trades:
        rw = sum(1 for t in ranging_trades if t.pnl_net >= 0)
        log.info("  Ranging:   %d trades | %d wins (%.0f%%) | PnL $%.2f",
                 len(ranging_trades), rw, rw / len(ranging_trades) * 100,
                 sum(t.pnl_net for t in ranging_trades))
    log.info("")
    log.info("BY SYMBOL")
    for s, pnl in sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True):
        count = sum(1 for t in trades if t.symbol.split("/")[0] == s)
        log.info("  %-6s %3d trades | PnL $%+.2f", s, count, pnl)
    log.info("")
    log.info("FEE BREAKDOWN")
    log.info("  Entry fees:  $%.2f", sum(t.fee_entry for t in trades))
    log.info("  Exit fees:   $%.2f", sum(t.fee_exit for t in trades))
    log.info("  Total:       $%.2f (%.1f%% of gross PnL)", total_fees,
             abs(total_fees / total_pnl_gross * 100) if total_pnl_gross != 0 else 0)

    # Top 5 trades
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

    # Daily PnL
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
