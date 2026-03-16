import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.config import StrategyConfig, IndicatorConfig, TrendingConfig
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
    ema_mid = row.get("ema_21") or 0

    # Don't long if price below EMA21 (downtrend)
    if close < ema_mid and ema_mid > 0:
        return None

    rsi = row["rsi"]
    bb_lower = row["bb_lower"]
    bb_mid = row["bb_mid"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]

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
    ema_mid = row.get("ema_21") or row.get(f"ema_{cfg.adx_trending}", 0)

    # Don't short if price above EMA21 (uptrend)
    if close > ema_mid and ema_mid > 0:
        return None
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


# ── TRENDING MODE (breakout on 1H, trailing SL only) ─────────────

def scan_trending_1h(df_1h: pd.DataFrame, cfg: TrendingConfig, bias: HTFBias) -> Signal | None:
    """Breakout momentum on 1H: ADX rising, volume surge, DI alignment, body strength."""
    if len(df_1h) < cfg.lookback + 5:
        return None

    row = df_1h.iloc[-1]
    prev = df_1h.iloc[-2]

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
    ema_f = row.get("ema_9") or 0
    ema_m = row.get("ema_21") or 0
    prev_adx = prev.get("adx") or 0

    if atr <= 0 or adx < cfg.adx_min:
        return None
    if vsma <= 0 or vol < vsma * cfg.vol_mult:
        return None
    # ADX not falling
    if adx < prev_adx - 1:
        return None

    # Body strength
    candle_range = high - low
    if candle_range <= 0:
        return None
    body = abs(close - opn)
    if body / candle_range < cfg.body_pct:
        return None

    # Recent high/low for breakout
    recent = df_1h.iloc[-(cfg.lookback + 1):-1]
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    # BREAKOUT LONG
    if (plus_di > minus_di
            and close > recent_high
            and close > opn
            and ema_f > ema_m
            and 50 < rsi < 80
            and bias in (HTFBias.BULLISH, HTFBias.NEUTRAL)):
        sl = close - cfg.sl_atr * atr
        tp = close + cfg.sl_atr * 1.5 * atr  # for R:R check only
        return Signal(
            type=SignalType.LONG, source=SignalSource.MAIN,
            regime=Regime.TRENDING, entry_price=close,
            sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
            reason=f"BRK LONG | ADX {adx:.0f} +DI>{minus_di:.0f} vol {vol/vsma:.1f}x",
        )

    # BREAKOUT SHORT
    if (minus_di > plus_di
            and close < recent_low
            and close < opn
            and ema_f < ema_m
            and 20 < rsi < 50
            and bias in (HTFBias.BEARISH, HTFBias.NEUTRAL)):
        sl = close + cfg.sl_atr * atr
        tp = close - cfg.sl_atr * 1.5 * atr
        return Signal(
            type=SignalType.SHORT, source=SignalSource.MAIN,
            regime=Regime.TRENDING, entry_price=close,
            sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
            reason=f"BRK SHORT | ADX {adx:.0f} -DI>{plus_di:.0f} vol {vol/vsma:.1f}x",
        )

    return None


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
    """Scan 15m for ranging signals only. Trending is handled separately on 1H."""
    regime = detect_regime(df_15m, cfg)
    logger.info(
        "Regime: %s | HTF: %s | ADX: %.1f",
        regime.value, bias.value, df_15m.iloc[-1].get("adx", 0),
    )

    # Only check ranging on 15m (ADX < 25)
    if regime != Regime.RANGING:
        return None

    signal = check_ranging_long(df_15m, cfg, bias)
    if signal:
        return signal
    return check_ranging_short(df_15m, cfg, bias)


def confirm_on_5m(df_5m: pd.DataFrame, signal: Signal) -> bool:
    """Check if 5m data confirms the 15m signal direction."""
    if len(df_5m) < 10:
        return False
    row = df_5m.iloc[-1]
    rsi = row.get("rsi") or 50
    close = row["close"]
    bb_lower = row.get("bb_lower") or 0
    bb_upper = row.get("bb_upper") or 0
    bb_mid = row.get("bb_mid") or 0

    if signal.type == SignalType.LONG:
        # 5m confirms: price near/below BB lower or RSI < 45
        return (bb_lower > 0 and close <= bb_mid) or rsi < 45
    elif signal.type == SignalType.SHORT:
        # 5m confirms: price near/above BB upper or RSI > 55
        return (bb_upper > 0 and close >= bb_mid) or rsi > 55
    return False


def scan_alert(df_1m: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias = HTFBias.NEUTRAL) -> Signal | None:
    return check_alert_signal(df_1m, cfg, bias)
