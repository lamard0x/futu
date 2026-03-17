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
    """H4 trend direction — simplified: EMA9 vs EMA21."""
    row = df_htf.iloc[-1]
    ema_f = row["ema_9"]
    ema_m = row["ema_21"]

    if ema_f > ema_m:
        return HTFBias.BULLISH
    if ema_f < ema_m:
        return HTFBias.BEARISH
    return HTFBias.NEUTRAL


def detect_regime(df: pd.DataFrame, cfg: StrategyConfig) -> Regime:
    row = df.iloc[-1]
    adx = row.get("adx", 0)
    if adx > cfg.adx_trending:
        return Regime.TRENDING
    return Regime.RANGING


# ── DEMAND / SUPPLY ZONES ────────────────────────────────────────────

def find_demand_zones(df: pd.DataFrame, lookback: int = 50, strength: int = 3) -> list[tuple[float, float]]:
    """Find demand zones from recent swing lows.
    Returns list of (zone_low, zone_high) tuples.
    """
    zones = []
    end = len(df) - 1
    start = max(0, end - lookback)
    for i in range(start + strength, end - strength):
        low_i = df.iloc[i]["low"]
        is_swing = all(
            low_i <= df.iloc[i - j]["low"] for j in range(1, strength + 1)
        ) and all(
            low_i <= df.iloc[i + j]["low"] for j in range(1, strength + 1)
        )
        if is_swing:
            body_low = min(df.iloc[i]["open"], df.iloc[i]["close"])
            zones.append((low_i, body_low))
    return zones


def find_supply_zones(df: pd.DataFrame, lookback: int = 50, strength: int = 3) -> list[tuple[float, float]]:
    """Find supply zones from recent swing highs.
    Returns list of (zone_low, zone_high) tuples.
    """
    zones = []
    end = len(df) - 1
    start = max(0, end - lookback)
    for i in range(start + strength, end - strength):
        high_i = df.iloc[i]["high"]
        is_swing = all(
            high_i >= df.iloc[i - j]["high"] for j in range(1, strength + 1)
        ) and all(
            high_i >= df.iloc[i + j]["high"] for j in range(1, strength + 1)
        )
        if is_swing:
            body_high = max(df.iloc[i]["open"], df.iloc[i]["close"])
            zones.append((body_high, high_i))
    return zones


def in_demand_zone(price: float, zones: list[tuple[float, float]], tolerance: float = 0.002) -> bool:
    """Check if price is within any demand zone (with small tolerance)."""
    for zone_low, zone_high in zones:
        margin = (zone_high - zone_low) * tolerance / 0.002 if zone_high > zone_low else zone_low * tolerance
        if zone_low - margin <= price <= zone_high + margin:
            return True
    return False


def in_supply_zone(price: float, zones: list[tuple[float, float]], tolerance: float = 0.002) -> bool:
    """Check if price is within any supply zone (with small tolerance)."""
    for zone_low, zone_high in zones:
        margin = (zone_high - zone_low) * tolerance / 0.002 if zone_high > zone_low else zone_high * tolerance
        if zone_low - margin <= price <= zone_high + margin:
            return True
    return False


# ── RANGING MODE (mean reversion, filtered by HTF) ──────────────────

def check_ranging_long(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias, symbol: str = "", extra_demand: list | None = None) -> Signal | None:
    if len(df) < 3:
        return None
    row = df.iloc[-2]  # Use closed candle — live candle wick/close unreliable
    close = row["close"]
    opn = row["open"]
    low = row["low"]
    high = row["high"]
    rsi = row["rsi"]
    bb_lower = row["bb_lower"]
    bb_mid = row["bb_mid"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]

    # Wick touches BB lower, close inside BB, wick >= 25%, bullish candle
    candle_range = high - low
    lower_wick = min(close, opn) - low
    wick_pct = lower_wick / candle_range if candle_range > 0 else 0
    touch_lower = low <= bb_lower * (1 + cfg.bb_touch_pct / 100)
    close_inside = close > bb_lower
    wick_ok = wick_pct >= 0.15
    bullish = close > opn

    rsi_oversold = rsi <= cfg.rsi_oversold
    volume_ok = volume > vol_sma * cfg.volume_range_mult

    if not (touch_lower and close_inside and wick_ok and bullish and rsi_oversold and volume_ok):
        reasons = []
        if not touch_lower:
            dist = (low - bb_lower) / bb_lower * 100
            reasons.append(f"BB {dist:.1f}% away")
        if touch_lower and not close_inside:
            reasons.append("close below BB")
        if touch_lower and not wick_ok:
            reasons.append(f"wick {wick_pct:.0%} < 15%")
        if touch_lower and not bullish:
            reasons.append("bearish candle")
        if not rsi_oversold:
            reasons.append(f"RSI {rsi:.0f} > {cfg.rsi_oversold}")
        if not volume_ok:
            ratio = volume / vol_sma if vol_sma > 0 else 0
            reasons.append(f"vol {ratio:.1f}x < {cfg.volume_range_mult}x")
        logger.debug("SKIP LONG %s: %s", symbol, " | ".join(reasons))
        return None

    # Demand zone filter — check 15m + H1/H4 zones
    demand_zones = find_demand_zones(df) + (extra_demand or [])
    if not in_demand_zone(low, demand_zones):
        logger.debug("SKIP LONG %s: no demand zone near %.2f", symbol, low)
        return None

    entry = close
    if bb_mid <= entry:
        logger.debug("SKIP LONG %s: price above BB mid (no room for TP)", symbol)
        return None
    sl = entry - cfg.main_sl_ranging_atr_mult * atr
    tp1 = entry + (bb_mid - entry) * 0.50  # 50% distance to BB mid

    return Signal(
        type=SignalType.LONG,
        source=SignalSource.MAIN,
        regime=Regime.RANGING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=None,
        atr=atr,
        reason=f"RANGE LONG | BB wick {wick_pct:.0%} + RSI {rsi:.0f} + demand",
    )


def check_ranging_short(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias, symbol: str = "", extra_supply: list | None = None) -> Signal | None:
    if len(df) < 3:
        return None
    row = df.iloc[-2]  # Use closed candle — live candle wick/close unreliable
    close = row["close"]
    opn = row["open"]
    low = row["low"]
    high = row["high"]
    rsi = row["rsi"]
    bb_upper = row["bb_upper"]
    bb_mid = row["bb_mid"]
    volume = row["volume"]
    vol_sma = row["volume_sma"]
    atr = row["atr"]

    # Wick touches BB upper, close inside BB, wick >= 25%, bearish candle
    candle_range = high - low
    upper_wick = high - max(close, opn)
    wick_pct = upper_wick / candle_range if candle_range > 0 else 0
    touch_upper = high >= bb_upper * (1 - cfg.bb_touch_pct / 100)
    close_inside = close < bb_upper
    wick_ok = wick_pct >= 0.15
    bearish = close < opn

    rsi_overbought = rsi >= cfg.rsi_overbought
    volume_ok = volume > vol_sma * cfg.volume_range_mult

    if not (touch_upper and close_inside and wick_ok and bearish and rsi_overbought and volume_ok):
        reasons = []
        if not touch_upper:
            dist = (bb_upper - high) / bb_upper * 100
            reasons.append(f"BB {dist:.1f}% away")
        if touch_upper and not close_inside:
            reasons.append("close above BB")
        if touch_upper and not wick_ok:
            reasons.append(f"wick {wick_pct:.0%} < 15%")
        if touch_upper and not bearish:
            reasons.append("bullish candle")
        if not rsi_overbought:
            reasons.append(f"RSI {rsi:.0f} < {cfg.rsi_overbought}")
        if not volume_ok:
            ratio = volume / vol_sma if vol_sma > 0 else 0
            reasons.append(f"vol {ratio:.1f}x < {cfg.volume_range_mult}x")
        logger.debug("SKIP SHORT %s: %s", symbol, " | ".join(reasons))
        return None

    # Supply zone filter — check 15m + H1/H4 zones
    supply_zones = find_supply_zones(df) + (extra_supply or [])
    if not in_supply_zone(high, supply_zones):
        logger.debug("SKIP SHORT %s: no supply zone near %.2f", symbol, high)
        return None

    entry = close
    if bb_mid >= entry:
        logger.debug("SKIP SHORT %s: price below BB mid (no room for TP)", symbol)
        return None
    sl = entry + cfg.main_sl_ranging_atr_mult * atr
    tp1 = entry - (entry - bb_mid) * 0.50  # 50% distance to BB mid

    return Signal(
        type=SignalType.SHORT,
        source=SignalSource.MAIN,
        regime=Regime.RANGING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=None,
        atr=atr,
        reason=f"RANGE SHORT | BB wick {wick_pct:.0%} + RSI {rsi:.0f} + supply",
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

    # BREAKOUT LONG — entry at breakout level (retest), not at close
    if (plus_di > minus_di
            and close > recent_high
            and close > opn
            and ema_f > ema_m
            and 50 < rsi < 80
            and bias in (HTFBias.BULLISH, HTFBias.NEUTRAL)):
        entry = recent_high  # limit buy at breakout level, wait for retest
        sl = entry - cfg.sl_atr * atr
        tp = entry + cfg.sl_atr * 1.5 * atr
        return Signal(
            type=SignalType.LONG, source=SignalSource.MAIN,
            regime=Regime.TRENDING, entry_price=entry,
            sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
            reason=f"BRK LONG | ADX {adx:.0f} +DI>{minus_di:.0f} vol {vol/vsma:.1f}x retest@{entry:.2f}",
        )

    # BREAKOUT SHORT — entry at breakout level (retest), not at close
    if (minus_di > plus_di
            and close < recent_low
            and close < opn
            and ema_f < ema_m
            and 20 < rsi < 50
            and bias in (HTFBias.BEARISH, HTFBias.NEUTRAL)):
        entry = recent_low  # limit sell at breakout level, wait for retest
        sl = entry + cfg.sl_atr * atr
        tp = entry - cfg.sl_atr * 1.5 * atr
        return Signal(
            type=SignalType.SHORT, source=SignalSource.MAIN,
            regime=Regime.TRENDING, entry_price=entry,
            sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
            reason=f"BRK SHORT | ADX {adx:.0f} -DI>{plus_di:.0f} vol {vol/vsma:.1f}x retest@{entry:.2f}",
        )

    return None


# ── TRENDING PULLBACK (EMA bounce in trend) ──────────────────────

def scan_trending_pullback(df: pd.DataFrame, cfg: TrendingConfig, bias: HTFBias, symbol: str = "") -> Signal | None:
    """Two-layer pullback entry:
    Layer 1: EMA21 touch + 40% wick rejection → entry at EMA21
    Layer 2: If EMA21 no wick → wait for EMA50 touch + any wick → entry at EMA50
    """
    if len(df) < 30:
        return None

    row = df.iloc[-1]

    close = row["close"]
    opn = row["open"]
    low = row["low"]
    high = row["high"]
    adx = row.get("adx") or 0
    atr = row.get("atr") or 0
    rsi = row.get("rsi") or 0
    plus_di = row.get("plus_di") or 0
    minus_di = row.get("minus_di") or 0
    ema_f = row.get("ema_9") or 0
    ema_m = row.get("ema_21") or 0
    ema_s = row.get("ema_50") or 0

    if atr <= 0 or adx < cfg.adx_min:
        logger.debug("SKIP PB %s: ADX %.0f < %s", symbol, adx, cfg.adx_min)
        return None

    candle_range = high - low
    if candle_range <= 0:
        return None

    # ── LONG ──
    long_bias_ok = bias in (HTFBias.BULLISH, HTFBias.NEUTRAL)
    long_di_ok = plus_di > minus_di
    long_ema_ok = ema_f > ema_m
    long_candle_ok = close > opn
    long_rsi_ok = 40 < rsi < 70

    if long_bias_ok and long_di_ok and long_ema_ok and long_candle_ok and long_rsi_ok:
        # Wick ratio for long: lower wick / total range
        lower_wick = min(close, opn) - low
        wick_pct = lower_wick / candle_range

        ema21_dist = (low - ema_m) / ema_m * 100 if ema_m > 0 else 99
        ema50_dist = (low - ema_s) / ema_s * 100 if ema_s > 0 else 99

        # Layer 1: EMA21 touch + 40% wick rejection
        if low <= ema_m * 1.002 and close > ema_m and wick_pct >= 0.4:
            entry = ema_m
            sl = ema_m - 1.5 * atr
            tp = entry + 2.0 * (entry - sl)
            return Signal(
                type=SignalType.LONG, source=SignalSource.MAIN,
                regime=Regime.TRENDING, entry_price=entry,
                sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
                reason=f"PB LONG | EMA21 wick {wick_pct:.0%} ADX {adx:.0f}",
            )

        # Layer 2: EMA50 touch + 40% wick rejection
        if ema_s > 0 and low <= ema_s * 1.002 and close > ema_s and wick_pct >= 0.4:
            entry = ema_s
            sl = ema_s - 1.5 * atr
            tp = entry + 2.0 * (entry - sl)
            return Signal(
                type=SignalType.LONG, source=SignalSource.MAIN,
                regime=Regime.TRENDING, entry_price=entry,
                sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
                reason=f"PB LONG | EMA50 wick {wick_pct:.0%} ADX {adx:.0f}",
            )

        # Log why pullback didn't trigger
        reasons = []
        if ema21_dist > 0.2:
            reasons.append(f"EMA21 {ema21_dist:.1f}% away")
        elif wick_pct < 0.4:
            reasons.append(f"EMA21 touch but wick {wick_pct:.0%} < 40%")
        if ema50_dist > 0.2:
            reasons.append(f"EMA50 {ema50_dist:.1f}% away")
        elif wick_pct < 0.4:
            reasons.append(f"EMA50 touch but wick {wick_pct:.0%} < 40%")
        logger.debug("SKIP PB LONG %s: %s", symbol, " | ".join(reasons))

    elif adx >= cfg.adx_min:
        # Log which trending precondition failed
        reasons = []
        if not long_bias_ok:
            reasons.append(f"bias={bias.value}")
        if not long_di_ok:
            reasons.append(f"+DI {plus_di:.0f} <= -DI {minus_di:.0f}")
        if not long_ema_ok:
            reasons.append(f"EMA9 < EMA21")
        if not long_candle_ok:
            reasons.append("bearish candle")
        if not long_rsi_ok:
            reasons.append(f"RSI {rsi:.0f} out 40-70")
        logger.debug("SKIP PB LONG %s: %s", symbol, " | ".join(reasons))

    # ── SHORT ──
    short_bias_ok = bias in (HTFBias.BEARISH, HTFBias.NEUTRAL)
    short_di_ok = minus_di > plus_di
    short_ema_ok = ema_f < ema_m
    short_candle_ok = close < opn
    short_rsi_ok = 30 < rsi < 60

    if short_bias_ok and short_di_ok and short_ema_ok and short_candle_ok and short_rsi_ok:
        # Wick ratio for short: upper wick / total range
        upper_wick = high - max(close, opn)
        wick_pct = upper_wick / candle_range

        ema21_dist = (ema_m - high) / ema_m * 100 if ema_m > 0 else 99
        ema50_dist = (ema_s - high) / ema_s * 100 if ema_s > 0 else 99

        # Layer 1: EMA21 touch + 40% wick rejection
        if high >= ema_m * 0.998 and close < ema_m and wick_pct >= 0.4:
            entry = ema_m
            sl = ema_m + 1.5 * atr
            tp = entry - 2.0 * (sl - entry)
            return Signal(
                type=SignalType.SHORT, source=SignalSource.MAIN,
                regime=Regime.TRENDING, entry_price=entry,
                sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
                reason=f"PB SHORT | EMA21 wick {wick_pct:.0%} ADX {adx:.0f}",
            )

        # Layer 2: EMA50 touch + 40% wick rejection
        if ema_s > 0 and high >= ema_s * 0.998 and close < ema_s and wick_pct >= 0.4:
            entry = ema_s
            sl = ema_s + 1.5 * atr
            tp = entry - 2.0 * (sl - entry)
            return Signal(
                type=SignalType.SHORT, source=SignalSource.MAIN,
                regime=Regime.TRENDING, entry_price=entry,
                sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
                reason=f"PB SHORT | EMA50 wick {wick_pct:.0%} ADX {adx:.0f}",
            )

        # Log why pullback didn't trigger
        reasons = []
        if ema21_dist > 0.2:
            reasons.append(f"EMA21 {ema21_dist:.1f}% away")
        elif wick_pct < 0.4:
            reasons.append(f"EMA21 touch but wick {wick_pct:.0%} < 40%")
        if ema50_dist > 0.2:
            reasons.append(f"EMA50 {ema50_dist:.1f}% away")
        elif wick_pct < 0.4:
            reasons.append(f"EMA50 touch but wick {wick_pct:.0%} < 40%")
        logger.debug("SKIP PB SHORT %s: %s", symbol, " | ".join(reasons))

    elif adx >= cfg.adx_min and not long_bias_ok:
        reasons = []
        if not short_bias_ok:
            reasons.append(f"bias={bias.value}")
        if not short_di_ok:
            reasons.append(f"-DI {minus_di:.0f} <= +DI {plus_di:.0f}")
        if not short_ema_ok:
            reasons.append(f"EMA9 > EMA21")
        if not short_candle_ok:
            reasons.append("bullish candle")
        if not short_rsi_ok:
            reasons.append(f"RSI {rsi:.0f} out 30-60")
        logger.debug("SKIP PB SHORT %s: %s", symbol, " | ".join(reasons))

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

def scan_main(df_15m: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias = HTFBias.NEUTRAL, symbol: str = "",
              extra_demand: list | None = None, extra_supply: list | None = None) -> Signal | None:
    """Scan 15m for ranging signals — BB mean reversion works in any regime."""
    regime = detect_regime(df_15m, cfg)
    logger.info(
        "Regime: %s | HTF: %s | ADX: %.1f",
        regime.value, bias.value, df_15m.iloc[-1].get("adx", 0),
    )

    signal = check_ranging_long(df_15m, cfg, bias, symbol, extra_demand=extra_demand)
    if signal:
        return signal
    return check_ranging_short(df_15m, cfg, bias, symbol, extra_supply=extra_supply)


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
