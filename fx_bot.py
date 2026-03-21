"""
FUTU FX Live Trading Bot — Session-based strategy on Exness MT5
Asian session (00:00-09:00 UTC): BB ranging on 5m
London/NY (09:00-21:00 UTC): Trending breakout on 1H
Off hours (21:00-00:00 UTC): Monitor exits only

Usage: python fx_bot.py
"""
import asyncio
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

from src.indicators import compute_all
from src.config import IndicatorConfig, StrategyConfig
from src.telegram import (
    send_message, notify_error, notify_daily_summary,
)

# ═══ Logging ═══
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_DIR / "fx_bot.log", encoding="utf-8")
file_handler.setFormatter(log_fmt)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_fmt)

log = logging.getLogger("fx_bot")
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(stream_handler)

# ═══ MT5 Credentials ═══
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# ═══ Constants ═══
MAGIC_NUMBER = 202603
COMMENT = "FUTU FX"

IND_CFG = IndicatorConfig()
STRAT_CFG = StrategyConfig()

BALANCE_START = float(os.getenv("FX_BALANCE_START", "0"))

# Sessions
SESSION_ASIAN = "asian"
SESSION_LONDON = "london"
SESSION_NY = "ny"
SESSION_OFF = "off"

# Symbols
ASIAN_SYMBOLS = ["EURUSDm", "GBPUSDm", "AUDUSDm", "USDJPYm"]
TRENDING_SYMBOLS = ["GBPUSDm", "GBPJPYm"]
ALL_SYMBOLS = list(set(ASIAN_SYMBOLS + TRENDING_SYMBOLS))

PIP_SIZE = {
    "EURUSDm": 0.0001, "GBPUSDm": 0.0001, "USDJPYm": 0.01,
    "AUDUSDm": 0.0001, "GBPJPYm": 0.01,
}
PIP_VALUE = {
    "EURUSDm": 10.0, "GBPUSDm": 10.0, "USDJPYm": 6.5,
    "AUDUSDm": 10.0, "GBPJPYm": 6.5,
}


# ═══ Trading Profiles ═══

@dataclass
class FXProfile:
    name: str
    risk_ranging: float
    risk_trending: float
    ranging_sl_atr: float
    ranging_min_rr: float
    trending_sl_atr: float
    trending_min_rr: float
    trailing_be_pct: float
    partial_close_pct: float
    t_adx_min: float
    t_lookback: int
    t_body_pct: float
    t_vol_mult: float
    max_daily_loss_pct: float  # safety cap: stop trading if daily loss exceeds this
    max_positions: int


PROFILES = {
    "normal": FXProfile(
        name="normal",
        risk_ranging=0.02,
        risk_trending=0.02,
        ranging_sl_atr=0.5,
        ranging_min_rr=1.2,
        trending_sl_atr=1.5,
        trending_min_rr=1.5,
        trailing_be_pct=0.50,
        partial_close_pct=0.50,
        t_adx_min=30,
        t_lookback=20,
        t_body_pct=0.5,
        t_vol_mult=1.2,
        max_daily_loss_pct=0.06,   # 6% daily loss cap
        max_positions=10,
    ),
    "challenge": FXProfile(
        name="challenge",
        risk_ranging=0.015,        # conservative: protect drawdown
        risk_trending=0.015,
        ranging_sl_atr=0.4,        # tighter SL
        ranging_min_rr=1.3,        # higher R:R for better expectancy
        trending_sl_atr=1.2,       # tighter SL
        trending_min_rr=1.5,
        trailing_be_pct=0.40,      # lock profit earlier
        partial_close_pct=0.50,
        t_adx_min=30,
        t_lookback=20,
        t_body_pct=0.5,
        t_vol_mult=1.2,
        max_daily_loss_pct=0.04,   # 4% daily cap (FundedNext = 5%, buffer 1%)
        max_positions=4,           # limit exposure
    ),
}

PROFILE_NAME = os.getenv("FX_PROFILE", "normal")
PROFILE = PROFILES.get(PROFILE_NAME, PROFILES["normal"])

# Trending detection
T_ADX_MIN = PROFILE.t_adx_min
T_LOOKBACK = PROFILE.t_lookback
T_BODY_PCT = PROFILE.t_body_pct
T_VOL_MULT = PROFILE.t_vol_mult

# Scan intervals (seconds)
TICK_INTERVAL = 30
SCAN_5M_INTERVAL = 300
SCAN_1H_INTERVAL = 3600
BIAS_H4_INTERVAL = 900

# MT5 timeframes
TF_MAP = {
    "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


# ═══ Data Classes ═══

@dataclass
class LiveTrade:
    """Tracks a live position opened by this bot."""
    ticket: int
    symbol: str
    side: str  # "long" or "short"
    regime: str  # "ranging" or "trending"
    session: str
    entry_price: float
    sl_price: float
    tp_price: float
    lots: float
    lots_remaining: float
    entry_time: datetime
    partial_closed: bool = False


@dataclass
class DailyStats:
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0

    def reset(self, date_str: str):
        self.date = date_str
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0


@dataclass
class BotState:
    running: bool = True
    connected: bool = False
    balance_start: float = 0.0
    balance: float = 0.0
    tracked_trades: list = field(default_factory=list)
    h4_biases: dict = field(default_factory=dict)
    daily: DailyStats = field(default_factory=DailyStats)
    last_scan_5m: float = 0.0
    last_scan_1h: float = 0.0
    last_bias_update: float = 0.0
    last_daily_summary: str = ""


# ═══ MT5 Helpers ═══

def mt5_connect() -> bool:
    """Initialize MT5 connection."""
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        err = mt5.last_error()
        log.error("MT5 init failed: %s", err)
        return False
    info = mt5.account_info()
    if info is None:
        log.error("MT5 account_info() returned None")
        return False
    log.info("MT5 connected: login=%d server=%s balance=%.2f",
             info.login, info.server, info.balance)
    return True


def mt5_ensure_connected() -> bool:
    """Check connection, reconnect if needed."""
    info = mt5.account_info()
    if info is not None:
        return True
    log.warning("MT5 disconnected, reconnecting...")
    mt5.shutdown()
    return mt5_connect()


def fetch_candles(symbol: str, tf_str: str, count: int = 200) -> list[dict]:
    """Fetch latest candles from MT5."""
    tf = TF_MAP.get(tf_str)
    if tf is None:
        return []
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        log.warning("No data for %s %s", symbol, tf_str)
        return []
    return [
        {
            "timestamp": int(r[0]) * 1000,
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        }
        for r in rates
    ]


def get_account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


def get_open_positions(symbol: str = None) -> list:
    """Get open positions, optionally filtered by symbol."""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER]


def has_open_position(symbol: str) -> bool:
    return len(get_open_positions(symbol)) > 0


# ═══ Order Execution ═══

def place_order(symbol: str, side: str, lots: float,
                sl: float, tp: float) -> int | None:
    """Place a market order. Returns ticket or None."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("No tick data for %s", symbol)
        return None

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        log.error("No symbol info for %s", symbol)
        return None

    if not sym_info.visible:
        mt5.symbol_select(symbol, True)

    order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == "long" else tick.bid

    digits = sym_info.digits
    sl = round(sl, digits)
    tp = round(tp, digits)
    price = round(price, digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error("order_send returned None for %s", symbol)
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Order failed %s: retcode=%d comment=%s",
                  symbol, result.retcode, result.comment)
        return None

    log.info("ORDER FILLED %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f ticket=%d",
             symbol, side.upper(), lots, result.price, sl, tp, result.order)
    return result.order


def modify_sl(ticket: int, symbol: str, new_sl: float, tp: float) -> bool:
    """Modify SL of an existing position."""
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return False

    digits = sym_info.digits
    new_sl = round(new_sl, digits)
    tp = round(tp, digits)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": tp,
        "magic": MAGIC_NUMBER,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning("Modify SL failed ticket=%d: %s",
                    ticket, result.comment if result else "None")
        return False
    return True


def close_partial(ticket: int, symbol: str, lots_to_close: float) -> bool:
    """Partially close a position."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False

    pos = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    sym_info = mt5.symbol_info(symbol)
    digits = sym_info.digits if sym_info else 5

    lots_to_close = round(lots_to_close, 2)
    if lots_to_close < 0.01:
        return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots_to_close,
        "type": close_type,
        "position": ticket,
        "price": round(price, digits),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "FUTU FX partial",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.warning("Partial close failed ticket=%d: %s",
                    ticket, result.comment if result else "None")
        return False

    log.info("PARTIAL CLOSE ticket=%d %.2f lots @ %.5f", ticket, lots_to_close, price)
    return True


# ═══ Session Detection ═══

def get_session(dt_utc: datetime) -> str:
    if dt_utc.weekday() >= 5:
        return SESSION_OFF
    hour = dt_utc.hour
    if 0 <= hour < 9:
        return SESSION_ASIAN
    if 9 <= hour < 21:
        return SESSION_LONDON  # London + NY overlap
    return SESSION_OFF


def is_weekend() -> bool:
    now = datetime.now(timezone.utc)
    wd = now.weekday()
    if wd == 5:
        return True
    if wd == 6:
        return True
    if wd == 4 and now.hour >= 22:
        return True
    return False


# ═══ Strategy — Ranging (same as backtest) ═══

def get_rsi_thresholds(cfg, bias):
    if bias == "bullish":
        return cfg.rsi_bull_oversold, cfg.rsi_bull_overbought
    if bias == "bearish":
        return cfg.rsi_bear_oversold, cfg.rsi_bear_overbought
    return cfg.rsi_oversold, cfg.rsi_overbought


def detect_htf_bias(df_htf):
    if len(df_htf) < IND_CFG.ema_slow + 5:
        return "neutral"
    last = df_htf.iloc[-1]
    ema_f = last.get(f"ema_{IND_CFG.ema_fast}", 0)
    ema_m = last.get(f"ema_{IND_CFG.ema_mid}", 0)
    if ema_f > ema_m:
        return "bullish"
    if ema_f < ema_m:
        return "bearish"
    return "neutral"


def scan_ranging(row, prev_row, bias, cfg):
    """BB mean reversion — Asian session on 5m."""
    close = row["close"]
    opn = row["open"]
    high = row["high"]
    low = row["low"]
    rsi = row.get("rsi") or 0
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    vol = row.get("volume") or 0
    vsma = row.get("volume_sma") or 0
    bbu = row.get("bb_upper") or 0
    bbl = row.get("bb_lower") or 0
    bbm = row.get("bb_mid") or 0

    if atr <= 0 or adx >= cfg.adx_trending:
        return None

    bb_width = bbu - bbl
    if bb_width <= 0:
        return None
    candle_range = high - low
    if candle_range <= 0:
        return None

    p_close = prev_row["close"]
    p_bbl = prev_row.get("bb_lower") or 0
    p_bbu = prev_row.get("bb_upper") or 0

    oversold, overbought = get_rsi_thresholds(cfg, bias)

    # LONG
    touch_lower = low <= bbl * (1 + cfg.bb_touch_pct / 100)
    close_inside_long = close > bbl + bb_width * 0.25
    prev_above_bb = p_close > p_bbl if p_bbl > 0 else True
    mid_room_long = bbm > close

    if touch_lower and close_inside_long and prev_above_bb and mid_room_long:
        lower_wick = min(close, opn) - low
        wick_pct = lower_wick / candle_range

        opt_wick = wick_pct >= 0.15
        opt_bullish = close > opn
        opt_rsi = rsi <= oversold
        opt_vol = vol > vsma * cfg.volume_range_mult if vsma > 0 else True
        opt_count = sum([opt_wick, opt_bullish, opt_rsi, opt_vol])

        if opt_count >= 3:
            sl = close - PROFILE.ranging_sl_atr * atr
            tp = close + (bbm - close) * 0.50
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= PROFILE.ranging_min_rr:
                return {
                    "side": "long", "entry": close, "sl": sl, "tp": tp,
                    "reason": f"RANGING LONG | RSI={rsi:.0f} ADX={adx:.0f}",
                }

    # SHORT
    touch_upper = high >= bbu * (1 - cfg.bb_touch_pct / 100)
    close_inside_short = close < bbu - bb_width * 0.25
    prev_below_bb = p_close < p_bbu if p_bbu > 0 else True
    mid_room_short = bbm < close

    if touch_upper and close_inside_short and prev_below_bb and mid_room_short:
        upper_wick = high - max(close, opn)
        wick_pct = upper_wick / candle_range

        opt_wick = wick_pct >= 0.15
        opt_bearish = close < opn
        opt_rsi = rsi >= overbought
        opt_vol = vol > vsma * cfg.volume_range_mult if vsma > 0 else True
        opt_count = sum([opt_wick, opt_bearish, opt_rsi, opt_vol])

        if opt_count >= 3:
            sl = close + PROFILE.ranging_sl_atr * atr
            tp = close - (close - bbm) * 0.50
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= PROFILE.ranging_min_rr:
                return {
                    "side": "short", "entry": close, "sl": sl, "tp": tp,
                    "reason": f"RANGING SHORT | RSI={rsi:.0f} ADX={adx:.0f}",
                }

    return None


# ═══ Strategy — Trending (same as backtest) ═══

def scan_trending(row, prev_row, df_slice, bias):
    """Trending breakout on 1H — London/NY sessions."""
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
    if adx < prev_adx - 1:
        return None

    candle_range = high - low
    if candle_range <= 0:
        return None
    body = abs(close - opn)
    if body / candle_range < T_BODY_PCT:
        return None

    if len(df_slice) < T_LOOKBACK:
        return None
    recent = df_slice.iloc[-T_LOOKBACK:]
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    sl_dist = PROFILE.trending_sl_atr * atr

    # LONG breakout
    if (plus_di > minus_di and close > recent_high and close > opn
            and ema_f > ema_m and 50 < rsi < 80
            and bias in ("bullish", "neutral")):
        sl = ema_f - sl_dist if ema_f > 0 else close - sl_dist
        sl = min(sl, close - 0.5 * atr)
        risk = abs(close - sl)
        tp = close + 2.0 * risk
        if risk > 0 and (tp - close) / risk >= PROFILE.trending_min_rr:
            return {
                "side": "long", "entry": close, "sl": sl, "tp": tp,
                "reason": f"TREND LONG | ADX={adx:.0f} RSI={rsi:.0f}",
            }

    # SHORT breakout
    if (minus_di > plus_di and close < recent_low and close < opn
            and ema_f < ema_m and 20 < rsi < 50
            and bias in ("bearish", "neutral")):
        sl = ema_f + sl_dist if ema_f > 0 else close + sl_dist
        sl = max(sl, close + 0.5 * atr)
        risk = abs(close - sl)
        tp = close - 2.0 * risk
        if risk > 0 and (close - tp) / risk >= PROFILE.trending_min_rr:
            return {
                "side": "short", "entry": close, "sl": sl, "tp": tp,
                "reason": f"TREND SHORT | ADX={adx:.0f} RSI={rsi:.0f}",
            }

    return None


# ═══ Position Sizing ═══

def calc_lot_size(entry: float, sl: float, risk_pct: float,
                  symbol: str, balance_start: float) -> float:
    """Risk-based lot calculation. Fixed sizing on BALANCE_START."""
    pip_size = PIP_SIZE.get(symbol, 0.0001)
    pip_val = PIP_VALUE.get(symbol, 10.0)
    sl_pips = abs(entry - sl) / pip_size
    if sl_pips <= 0:
        return 0.0
    risk_amount = balance_start * risk_pct
    lots = risk_amount / (sl_pips * pip_val)
    lots = round(lots, 2)
    if lots < 0.01:
        return 0.0
    return lots


# ═══ Trailing / Partial Logic ═══

def check_trailing_and_partial(trade: LiveTrade, state: BotState):
    """Check trailing SL and partial close for a tracked trade."""
    positions = mt5.positions_get(ticket=trade.ticket)
    if not positions:
        return

    pos = positions[0]
    current_price = pos.price_current
    tp_dist = abs(trade.tp_price - trade.entry_price)
    half_tp_dist = tp_dist * PROFILE.trailing_be_pct

    if trade.side == "long":
        half_tp_level = trade.entry_price + half_tp_dist
        reached_half = current_price >= half_tp_level
    else:
        half_tp_level = trade.entry_price - half_tp_dist
        reached_half = current_price <= half_tp_level

    if not reached_half:
        return

    # Partial close at 50% TP
    if not trade.partial_closed:
        close_lots = round(trade.lots * PROFILE.partial_close_pct, 2)
        if close_lots >= 0.01:
            ok = close_partial(trade.ticket, trade.symbol, close_lots)
            if ok:
                trade.partial_closed = True
                trade.lots_remaining = round(trade.lots - close_lots, 2)
                log.info("Partial close %s %.2f lots, remaining %.2f",
                         trade.symbol, close_lots, trade.lots_remaining)

    # Move SL to breakeven
    if trade.side == "long" and trade.sl_price < trade.entry_price:
        sym_info = mt5.symbol_info(trade.symbol)
        digits = sym_info.digits if sym_info else 5
        new_sl = round(trade.entry_price, digits)
        if modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
            trade.sl_price = new_sl
            log.info("SL moved to breakeven %s @ %.5f", trade.symbol, new_sl)
    elif trade.side == "short" and trade.sl_price > trade.entry_price:
        sym_info = mt5.symbol_info(trade.symbol)
        digits = sym_info.digits if sym_info else 5
        new_sl = round(trade.entry_price, digits)
        if modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
            trade.sl_price = new_sl
            log.info("SL moved to breakeven %s @ %.5f", trade.symbol, new_sl)

    # Chandelier trailing (use 5m data for ranging, 1h for trending)
    tf = "5m" if trade.regime == "ranging" else "1h"
    candles = fetch_candles(trade.symbol, tf, 50)
    if len(candles) < 30:
        return
    df = compute_all(candles, IND_CFG)
    last = df.iloc[-1]

    chand_long = last.get("chandelier_long")
    chand_short = last.get("chandelier_short")

    if trade.side == "long" and chand_long:
        if chand_long > trade.entry_price and chand_long > trade.sl_price:
            sym_info = mt5.symbol_info(trade.symbol)
            digits = sym_info.digits if sym_info else 5
            new_sl = round(chand_long, digits)
            if modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
                trade.sl_price = new_sl
                log.info("Chandelier trail %s SL -> %.5f", trade.symbol, new_sl)
    elif trade.side == "short" and chand_short:
        if chand_short < trade.entry_price and chand_short < trade.sl_price:
            sym_info = mt5.symbol_info(trade.symbol)
            digits = sym_info.digits if sym_info else 5
            new_sl = round(chand_short, digits)
            if modify_sl(trade.ticket, trade.symbol, new_sl, trade.tp_price):
                trade.sl_price = new_sl
                log.info("Chandelier trail %s SL -> %.5f", trade.symbol, new_sl)


# ═══ Trade Sync ═══

def sync_tracked_trades(state: BotState):
    """Remove tracked trades whose MT5 positions are closed."""
    closed = []
    for trade in state.tracked_trades:
        positions = mt5.positions_get(ticket=trade.ticket)
        if not positions:
            closed.append(trade)

    for trade in closed:
        state.tracked_trades.remove(trade)
        # Determine PnL from deal history
        pnl = _get_closed_pnl(trade.ticket)
        won = pnl >= 0
        state.daily.trades += 1
        state.daily.pnl += pnl
        if won:
            state.daily.wins += 1
        else:
            state.daily.losses += 1

        reason = "TP" if won else "SL"
        log.info("CLOSED %s %s %s pnl=%.2f %s",
                 trade.symbol, trade.side.upper(), trade.regime, pnl, reason)
        asyncio.get_event_loop().create_task(
            _notify_close(trade, pnl, reason, state)
        )


def _get_closed_pnl(ticket: int) -> float:
    """Get PnL of a closed position from deal history."""
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=1)
    deals = mt5.history_deals_get(from_dt, now)
    if deals is None:
        return 0.0
    total = 0.0
    for deal in deals:
        if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
            total += deal.profit + deal.swap + deal.commission
    return total


async def _notify_close(trade: LiveTrade, pnl: float,
                        reason: str, state: BotState):
    balance = get_account_balance()
    emoji = "+" if pnl >= 0 else ""
    text = (
        f"{'WIN' if pnl >= 0 else 'LOSS'} {trade.symbol} "
        f"{trade.side.upper()} {trade.regime}\n"
        f"PnL: ${emoji}{pnl:.2f} | {reason}\n"
        f"Balance: ${balance:.2f}"
    )
    await send_message(text)


# ═══ Telegram Notifications ═══

async def notify_startup(state: BotState):
    text = (
        f"FX BOT STARTED\n"
        f"Balance: ${state.balance:.2f}\n"
        f"Ranging: {', '.join(ASIAN_SYMBOLS)}\n"
        f"Trending: {', '.join(TRENDING_SYMBOLS)}\n"
        f"Profile: {PROFILE.name.upper()}\n"
        f"Risk: {PROFILE.risk_ranging*100:.1f}% ranging / {PROFILE.risk_trending*100:.1f}% trending\n"
        f"Daily loss cap: {PROFILE.max_daily_loss_pct*100:.0f}% | Max positions: {PROFILE.max_positions}"
    )
    await send_message(text)


async def notify_signal(symbol: str, side: str, regime: str,
                        entry: float, sl: float, tp: float,
                        lots: float, reason: str):
    pip_size = PIP_SIZE.get(symbol, 0.0001)
    sl_pips = abs(entry - sl) / pip_size
    tp_pips = abs(tp - entry) / pip_size
    rr = tp_pips / sl_pips if sl_pips > 0 else 0
    text = (
        f"NEW TRADE {symbol} {side.upper()} [{regime}]\n"
        f"Entry: {entry:.5f}\n"
        f"SL: {sl:.5f} ({sl_pips:.1f} pips)\n"
        f"TP: {tp:.5f} ({tp_pips:.1f} pips)\n"
        f"Lots: {lots:.2f} | R:R {rr:.2f}\n"
        f"{reason}"
    )
    await send_message(text)


# ═══ Main Bot Loop ═══

class FXBot:
    def __init__(self):
        self.state = BotState()
        self._shutdown = False

    async def run(self):
        log.info("=" * 50)
        log.info("FUTU FX Bot starting...")
        log.info("=" * 50)

        if not mt5_connect():
            log.error("Cannot connect to MT5. Exiting.")
            await notify_error("FX Bot: MT5 connection failed")
            return

        self.state.connected = True
        self.state.balance = get_account_balance()

        # Set BALANCE_START from account if not configured
        global BALANCE_START
        if BALANCE_START <= 0:
            BALANCE_START = self.state.balance
            log.info("BALANCE_START set from account: %.2f", BALANCE_START)
        self.state.balance_start = BALANCE_START

        log.info("Profile: %s | Balance: %.2f | BALANCE_START: %.2f",
                 PROFILE.name.upper(), self.state.balance, BALANCE_START)
        log.info("Risk: %.1f%% | Daily cap: %.0f%% | Max pos: %d",
                 PROFILE.risk_ranging * 100, PROFILE.max_daily_loss_pct * 100,
                 PROFILE.max_positions)

        await notify_startup(self.state)

        # Initial H4 bias
        await self._update_h4_bias()

        try:
            while not self._shutdown:
                try:
                    await self._tick()
                except Exception as e:
                    log.error("Tick error: %s", e, exc_info=True)
                    await notify_error(f"FX Bot tick error: {e}")
                await asyncio.sleep(TICK_INTERVAL)
        except asyncio.CancelledError:
            pass
        finally:
            log.info("Shutting down...")
            mt5.shutdown()
            log.info("MT5 disconnected. Bot stopped.")

    def shutdown(self):
        self._shutdown = True

    async def _tick(self):
        """Main tick — runs every 30 seconds."""
        if not mt5_ensure_connected():
            self.state.connected = False
            log.warning("MT5 not connected, skipping tick")
            return
        self.state.connected = True

        now = time.time()
        now_utc = datetime.now(timezone.utc)
        session = get_session(now_utc)
        today = now_utc.strftime("%Y-%m-%d")

        # Reset daily stats
        if self.state.daily.date != today:
            # Send daily summary for previous day
            if self.state.daily.date and self.state.daily.trades > 0:
                await self._send_daily_summary()
            self.state.daily.reset(today)

        # Update balance
        self.state.balance = get_account_balance()

        # Sync tracked trades (detect closed positions)
        sync_tracked_trades(self.state)

        # Monitor trailing SL / partial close on all tracked trades
        for trade in list(self.state.tracked_trades):
            try:
                check_trailing_and_partial(trade, self.state)
            except Exception as e:
                log.warning("Trailing check error %s: %s", trade.symbol, e)

        # Skip scanning on weekends
        if is_weekend():
            return

        # Daily loss cap — stop trading if exceeded
        if self.state.balance_start > 0:
            daily_loss = self.state.daily.pnl
            daily_loss_pct = abs(daily_loss) / self.state.balance_start if daily_loss < 0 else 0
            if daily_loss_pct >= PROFILE.max_daily_loss_pct:
                return  # stop scanning, only monitor exits

        # Max positions cap
        open_count = len(get_open_positions())
        if open_count >= PROFILE.max_positions:
            return  # only monitor exits, no new trades

        # H4 bias update every 15 min
        if now - self.state.last_bias_update >= BIAS_H4_INTERVAL:
            await self._update_h4_bias()
            self.state.last_bias_update = now

        # Asian ranging scan — every 30s (aligned with tick), on 5m data
        if session == SESSION_ASIAN and now - self.state.last_scan_5m >= SCAN_5M_INTERVAL:
            await self._scan_ranging()
            self.state.last_scan_5m = now

        # London/NY trending scan — every 5 min on 1H data
        if session in (SESSION_LONDON, SESSION_NY):
            if now - self.state.last_scan_1h >= SCAN_5M_INTERVAL:
                await self._scan_trending()
                self.state.last_scan_1h = now

        # Daily summary at 21:00 UTC
        if (now_utc.hour == 21 and now_utc.minute < 1
                and self.state.last_daily_summary != today):
            await self._send_daily_summary()
            self.state.last_daily_summary = today

    async def _update_h4_bias(self):
        """Update H4 bias for all symbols."""
        for symbol in ALL_SYMBOLS:
            try:
                candles = fetch_candles(symbol, "4h", 100)
                if len(candles) < IND_CFG.ema_slow + 10:
                    self.state.h4_biases[symbol] = "neutral"
                    continue
                df = compute_all(candles, IND_CFG)
                bias = detect_htf_bias(df)
                old_bias = self.state.h4_biases.get(symbol, "")
                self.state.h4_biases[symbol] = bias
                if old_bias and old_bias != bias:
                    log.info("H4 bias %s: %s -> %s", symbol, old_bias, bias)
            except Exception as e:
                log.warning("H4 bias error %s: %s", symbol, e)
                self.state.h4_biases[symbol] = "neutral"

    async def _scan_ranging(self):
        """Scan Asian symbols on 5m for BB ranging signals."""
        for symbol in ASIAN_SYMBOLS:
            try:
                if has_open_position(symbol):
                    continue

                candles = fetch_candles(symbol, "5m", 200)
                if len(candles) < IND_CFG.ema_slow + 10:
                    continue

                df = compute_all(candles, IND_CFG)
                row = df.iloc[-1]
                prev_row = df.iloc[-2]

                bias = self.state.h4_biases.get(symbol, "neutral")
                sig = scan_ranging(row, prev_row, bias, STRAT_CFG)
                if sig is None:
                    continue

                # Calculate lot size
                lots = calc_lot_size(
                    sig["entry"], sig["sl"], PROFILE.risk_ranging,
                    symbol, BALANCE_START,
                )
                if lots <= 0:
                    continue

                log.info("SIGNAL %s | %s", symbol, sig["reason"])
                await notify_signal(
                    symbol, sig["side"], "ranging",
                    sig["entry"], sig["sl"], sig["tp"],
                    lots, sig["reason"],
                )

                ticket = place_order(
                    symbol, sig["side"], lots, sig["sl"], sig["tp"],
                )
                if ticket:
                    trade = LiveTrade(
                        ticket=ticket, symbol=symbol,
                        side=sig["side"], regime="ranging",
                        session=SESSION_ASIAN,
                        entry_price=sig["entry"],
                        sl_price=sig["sl"], tp_price=sig["tp"],
                        lots=lots, lots_remaining=lots,
                        entry_time=datetime.now(timezone.utc),
                    )
                    self.state.tracked_trades.append(trade)

            except Exception as e:
                log.warning("Ranging scan error %s: %s", symbol, e)

    async def _scan_trending(self):
        """Scan trending symbols on 1H for breakout signals."""
        for symbol in TRENDING_SYMBOLS:
            try:
                if has_open_position(symbol):
                    continue

                candles = fetch_candles(symbol, "1h", 200)
                if len(candles) < IND_CFG.ema_slow + 10:
                    continue

                df = compute_all(candles, IND_CFG)
                if len(df) < T_LOOKBACK + 5:
                    continue

                row = df.iloc[-1]
                prev_row = df.iloc[-2]
                df_slice = df.iloc[-(T_LOOKBACK + 5):]

                bias = self.state.h4_biases.get(symbol, "neutral")
                sig = scan_trending(row, prev_row, df_slice, bias)
                if sig is None:
                    continue

                lots = calc_lot_size(
                    sig["entry"], sig["sl"], PROFILE.risk_trending,
                    symbol, BALANCE_START,
                )
                if lots <= 0:
                    continue

                log.info("SIGNAL %s | %s", symbol, sig["reason"])
                await notify_signal(
                    symbol, sig["side"], "trending",
                    sig["entry"], sig["sl"], sig["tp"],
                    lots, sig["reason"],
                )

                ticket = place_order(
                    symbol, sig["side"], lots, sig["sl"], sig["tp"],
                )
                if ticket:
                    trade = LiveTrade(
                        ticket=ticket, symbol=symbol,
                        side=sig["side"], regime="trending",
                        session=get_session(datetime.now(timezone.utc)),
                        entry_price=sig["entry"],
                        sl_price=sig["sl"], tp_price=sig["tp"],
                        lots=lots, lots_remaining=lots,
                        entry_time=datetime.now(timezone.utc),
                    )
                    self.state.tracked_trades.append(trade)

            except Exception as e:
                log.warning("Trending scan error %s: %s", symbol, e)

    async def _send_daily_summary(self):
        d = self.state.daily
        balance = get_account_balance()
        await notify_daily_summary(
            d.wins, d.losses, d.pnl, balance, d.trades,
        )
        log.info("Daily summary: %d trades, %d/%d W/L, PnL $%.2f, Balance $%.2f",
                 d.trades, d.wins, d.losses, d.pnl, balance)


# ═══ Entry Point ═══

def main():
    bot = FXBot()

    # Graceful shutdown on Ctrl+C
    def handle_signal(*_):
        log.info("Shutdown signal received")
        bot.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
