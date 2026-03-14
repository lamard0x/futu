import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.config import StrategyConfig, IndicatorConfig
from src.indicators import compute_all

logger = logging.getLogger("futu.strategy")


class HTFBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Regime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    UNCERTAIN = "uncertain"


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class SignalSource(Enum):
    MAIN = "main"
    ALERT = "alert"


@dataclass
class Signal:
    type: SignalType
    source: SignalSource
    regime: Regime
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float | None
    atr: float
    reason: str


def detect_htf_bias(df_htf: pd.DataFrame) -> HTFBias:
    """H4 trend direction — the boss that decides long or short."""
    row = df_htf.iloc[-1]
    ema_f = row["ema_9"]
    ema_m = row["ema_21"]
    ema_s = row["ema_50"]
    adx = row.get("adx", 0)

    if ema_f > ema_m > ema_s and adx > 20:
        return HTFBias.BULLISH
    if ema_f < ema_m < ema_s and adx > 20:
        return HTFBias.BEARISH
    return HTFBias.NEUTRAL


def detect_regime(df: pd.DataFrame, cfg: StrategyConfig) -> Regime:
    row = df.iloc[-1]
    adx = row.get("adx", 0)
    if adx > cfg.adx_trending:
        return Regime.TRENDING
    return Regime.RANGING


# ── RANGING MODE (mean reversion, filtered by HTF) ──────────────────

def check_ranging_long(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias) -> Signal | None:
    if bias == HTFBias.BEARISH:
        return None  # Don't buy against H4 downtrend

    row = df.iloc[-1]
    close = row["close"]
    rsi = row["rsi"]
    bb_lower = row["bb_lower"]
    bb_mid = row["bb_mid"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]
    bullish_candle = close > row["open"]

    touch_lower = close <= bb_lower * (1 + cfg.bb_touch_pct / 100)
    rsi_oversold = rsi < cfg.rsi_oversold
    volume_ok = volume > vol_sma * cfg.volume_range_mult

    if not (touch_lower and rsi_oversold and volume_ok):
        return None

    entry = close
    sl = entry - cfg.main_sl_ranging_atr_mult * atr
    tp1 = bb_mid

    return Signal(
        type=SignalType.LONG,
        source=SignalSource.MAIN,
        regime=Regime.RANGING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=None,
        atr=atr,
        reason=f"RANGE LONG | BB + RSI {rsi:.0f} + H4 {bias.value}",
    )


def check_ranging_short(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias) -> Signal | None:
    if bias == HTFBias.BULLISH:
        return None  # Don't short against H4 uptrend

    row = df.iloc[-1]
    close = row["close"]
    rsi = row["rsi"]
    bb_upper = row["bb_upper"]
    bb_mid = row["bb_mid"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]
    bearish_candle = close < row["open"]

    touch_upper = close >= bb_upper * (1 - cfg.bb_touch_pct / 100)
    rsi_overbought = rsi > cfg.rsi_overbought
    volume_ok = volume > vol_sma * cfg.volume_range_mult

    if not (touch_upper and rsi_overbought and volume_ok):
        return None

    entry = close
    sl = entry + cfg.main_sl_ranging_atr_mult * atr
    tp1 = bb_mid

    return Signal(
        type=SignalType.SHORT,
        source=SignalSource.MAIN,
        regime=Regime.RANGING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=None,
        atr=atr,
        reason=f"RANGE SHORT | BB + RSI {rsi:.0f} + H4 {bias.value}",
    )


# ── TRENDING MODE (trend-following on 15m, WITH H4 direction) ───────

def check_trending_long(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias) -> Signal | None:
    if bias != HTFBias.BULLISH:
        return None  # Only long when H4 is bullish

    row = df.iloc[-1]
    prev = df.iloc[-2]

    ema_f = row["ema_9"]
    ema_m = row["ema_21"]
    close = row["close"]
    rsi = row["rsi"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]
    vwap = row["vwap"]

    # 15m must also be bullish
    above_vwap = close > vwap
    rsi_bull = rsi > 45 and rsi > prev["rsi"]
    volume_ok = volume > vol_sma * cfg.volume_trend_mult

    # Pullback to EMA21 or VWAP (buy the dip)
    pullback = row["low"] <= ema_m * 1.005 or row["low"] <= vwap * 1.005

    if not (above_vwap and rsi_bull and volume_ok and pullback):
        return None

    entry = close
    sl = entry - cfg.main_sl_trending_atr_mult * atr
    tp1 = entry + cfg.main_tp1_atr_mult * atr
    tp2 = entry + cfg.main_tp2_atr_mult * atr

    return Signal(
        type=SignalType.LONG,
        source=SignalSource.MAIN,
        regime=Regime.TRENDING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=tp2,
        atr=atr,
        reason=f"TREND LONG | H4 bull + 15m pullback + RSI {rsi:.0f}",
    )


def check_trending_short(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias) -> Signal | None:
    if bias != HTFBias.BEARISH:
        return None  # Only short when H4 is bearish

    row = df.iloc[-1]
    prev = df.iloc[-2]

    ema_f = row["ema_9"]
    ema_m = row["ema_21"]
    close = row["close"]
    rsi = row["rsi"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]
    vwap = row["vwap"]

    below_vwap = close < vwap
    rsi_bear = rsi < 55 and rsi < prev["rsi"]
    volume_ok = volume > vol_sma * cfg.volume_trend_mult

    # Pullback to EMA21 or VWAP (sell the rally)
    pullback = row["high"] >= ema_m * 0.995 or row["high"] >= vwap * 0.995

    if not (below_vwap and rsi_bear and volume_ok and pullback):
        return None

    entry = close
    sl = entry + cfg.main_sl_trending_atr_mult * atr
    tp1 = entry - cfg.main_tp1_atr_mult * atr
    tp2 = entry - cfg.main_tp2_atr_mult * atr

    return Signal(
        type=SignalType.SHORT,
        source=SignalSource.MAIN,
        regime=Regime.TRENDING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=tp2,
        atr=atr,
        reason=f"TREND SHORT | H4 bear + 15m pullback + RSI {rsi:.0f}",
    )


# ── ALERT MODE (1-min volume spike, filtered by HTF) ────────────────

def check_alert_signal(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias) -> Signal | None:
    row = df.iloc[-1]
    close = row["close"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]
    bb_upper = row["bb_upper"]
    bb_lower = row["bb_lower"]

    volume_alert = volume > vol_sma * cfg.volume_alert_mult
    if not volume_alert:
        return None

    if close < bb_lower and bias != HTFBias.BEARISH:
        entry = close
        sl = entry - cfg.alert_sl_atr_mult * atr
        tp = entry + cfg.alert_target_pct / 100 * entry
        return Signal(
            type=SignalType.LONG,
            source=SignalSource.ALERT,
            regime=Regime.UNCERTAIN,
            entry_price=entry,
            sl_price=sl,
            tp1_price=tp,
            tp2_price=None,
            atr=atr,
            reason=f"ALERT LONG | vol {volume/vol_sma:.1f}x + H4 {bias.value}",
        )

    if close > bb_upper and bias != HTFBias.BULLISH:
        entry = close
        sl = entry + cfg.alert_sl_atr_mult * atr
        tp = entry - cfg.alert_target_pct / 100 * entry
        return Signal(
            type=SignalType.SHORT,
            source=SignalSource.ALERT,
            regime=Regime.UNCERTAIN,
            entry_price=entry,
            sl_price=sl,
            tp1_price=tp,
            tp2_price=None,
            atr=atr,
            reason=f"ALERT SHORT | vol {volume/vol_sma:.1f}x + H4 {bias.value}",
        )

    return None


# ── MAIN SCANNERS ────────────────────────────────────────────────────

def scan_main(df_15m: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias = HTFBias.NEUTRAL) -> Signal | None:
    regime = detect_regime(df_15m, cfg)
    logger.info(
        "Regime: %s | HTF: %s | ADX: %.1f",
        regime.value, bias.value, df_15m.iloc[-1].get("adx", 0),
    )

    if regime == Regime.TRENDING:
        signal = check_trending_long(df_15m, cfg, bias)
        if signal:
            return signal
        signal = check_trending_short(df_15m, cfg, bias)
        if signal:
            return signal

    # Always check ranging (mean reversion is our bread & butter)
    signal = check_ranging_long(df_15m, cfg, bias)
    if signal:
        return signal
    return check_ranging_short(df_15m, cfg, bias)


def scan_alert(df_1m: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias = HTFBias.NEUTRAL) -> Signal | None:
    return check_alert_signal(df_1m, cfg, bias)
