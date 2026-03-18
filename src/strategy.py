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
    confluence_score: int = 0  # S/D zone confluence (0-3 TFs overlapping)
    condition_pct: float = 1.0  # fraction of conditions met (0.75 or 1.0)


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


def get_rsi_thresholds(cfg: StrategyConfig, bias: HTFBias) -> tuple[float, float]:
    """RSI Hayden zones — adjust thresholds by H4 bias direction."""
    if bias == HTFBias.BULLISH:
        return cfg.rsi_bull_oversold, cfg.rsi_bull_overbought
    if bias == HTFBias.BEARISH:
        return cfg.rsi_bear_oversold, cfg.rsi_bear_overbought
    return cfg.rsi_oversold, cfg.rsi_overbought


def score_demand_confluence(
    price: float,
    zones_15m: list[tuple[float, float]],
    zones_h1: list[tuple[float, float]],
    zones_h4: list[tuple[float, float]],
    tolerance: float = 0.002,
) -> int:
    """Count how many timeframes have a demand zone near price (0-3)."""
    score = 0
    if in_demand_zone(price, zones_15m, tolerance):
        score += 1
    if in_demand_zone(price, zones_h1, tolerance):
        score += 1
    if in_demand_zone(price, zones_h4, tolerance):
        score += 1
    return score


def score_supply_confluence(
    price: float,
    zones_15m: list[tuple[float, float]],
    zones_h1: list[tuple[float, float]],
    zones_h4: list[tuple[float, float]],
    tolerance: float = 0.002,
) -> int:
    """Count how many timeframes have a supply zone near price (0-3)."""
    score = 0
    if in_supply_zone(price, zones_15m, tolerance):
        score += 1
    if in_supply_zone(price, zones_h1, tolerance):
        score += 1
    if in_supply_zone(price, zones_h4, tolerance):
        score += 1
    return score


# ── RANGING MODE (mean reversion, filtered by HTF) ──────────────────

def check_ranging_long(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias, symbol: str = "",
                       zones_15m: list | None = None, zones_h1: list | None = None, zones_h4: list | None = None) -> Signal | None:
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

    candle_range = high - low
    lower_wick = min(close, opn) - low
    wick_pct = lower_wick / candle_range if candle_range > 0 else 0

    # ── Mandatory conditions (all must pass) ──
    touch_lower = low <= bb_lower * (1 + cfg.bb_touch_pct / 100)
    close_inside = close > bb_lower
    # Anti-breakout: previous candle must also be above BB lower (not a fresh breakdown)
    prev_row = df.iloc[-3] if len(df) >= 4 else row
    prev_above_bb = prev_row["close"] > prev_row["bb_lower"]
    if not (touch_lower and close_inside and prev_above_bb):
        reasons = []
        if not touch_lower:
            dist = (low - bb_lower) / bb_lower * 100
            reasons.append(f"BB {dist:.1f}% away")
        if touch_lower and not close_inside:
            reasons.append("close below BB")
        if touch_lower and close_inside and not prev_above_bb:
            reasons.append("breakdown — prev candle below BB")
        logger.debug("SKIP LONG %s: %s", symbol, " | ".join(reasons))
        return None

    # Demand zone confluence — mandatory
    z_15m = zones_15m if zones_15m is not None else find_demand_zones(df)
    z_h1 = zones_h1 or []
    z_h4 = zones_h4 or []
    confluence = score_demand_confluence(low, z_15m, z_h1, z_h4)
    if confluence == 0:
        logger.debug("SKIP LONG %s: no demand zone near %.2f", symbol, low)
        return None

    # BB mid room — mandatory
    if bb_mid <= close:
        logger.debug("SKIP LONG %s: price above BB mid (no room for TP)", symbol)
        return None

    # ── Optional conditions (score 4/4 = full, 2+/4 = 75%) ──
    oversold_threshold, _ = get_rsi_thresholds(cfg, bias)
    opt_wick = wick_pct >= 0.15
    opt_bullish = close > opn
    opt_rsi = rsi <= oversold_threshold
    opt_vol = volume > vol_sma * cfg.volume_range_mult if vol_sma > 0 else True

    opt_count = sum([opt_wick, opt_bullish, opt_rsi, opt_vol])
    if opt_count < 2:
        reasons = []
        if not opt_wick:
            reasons.append(f"wick {wick_pct:.0%} < 15%")
        if not opt_bullish:
            reasons.append("bearish candle")
        if not opt_rsi:
            reasons.append(f"RSI {rsi:.0f} > {oversold_threshold}")
        if not opt_vol:
            ratio = volume / vol_sma if vol_sma > 0 else 0
            reasons.append(f"vol {ratio:.1f}x < {cfg.volume_range_mult}x")
        logger.debug("SKIP LONG %s: %s (%d/4 optional)", symbol, " | ".join(reasons), opt_count)
        return None

    condition_pct = 1.0 if opt_count == 4 else 0.75
    entry = close
    sl = entry - cfg.main_sl_ranging_atr_mult * atr
    tp1 = entry + (bb_mid - entry) * 0.50

    tag = "100%" if opt_count == 4 else f"75%({opt_count}/4)"
    return Signal(
        type=SignalType.LONG,
        source=SignalSource.MAIN,
        regime=Regime.RANGING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=None,
        atr=atr,
        reason=f"RANGE LONG | BB wick {wick_pct:.0%} + RSI {rsi:.0f} + demand x{confluence} [{tag}]",
        confluence_score=confluence,
        condition_pct=condition_pct,
    )


def check_ranging_short(df: pd.DataFrame, cfg: StrategyConfig, bias: HTFBias, symbol: str = "",
                        zones_15m: list | None = None, zones_h1: list | None = None, zones_h4: list | None = None) -> Signal | None:
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

    candle_range = high - low
    upper_wick = high - max(close, opn)
    wick_pct = upper_wick / candle_range if candle_range > 0 else 0

    # ── Mandatory conditions (all must pass) ──
    touch_upper = high >= bb_upper * (1 - cfg.bb_touch_pct / 100)
    close_inside = close < bb_upper
    # Anti-breakout: previous candle must also be below BB upper (not a fresh breakout)
    prev_row = df.iloc[-3] if len(df) >= 4 else row
    prev_below_bb = prev_row["close"] < prev_row["bb_upper"]
    if not (touch_upper and close_inside and prev_below_bb):
        reasons = []
        if not touch_upper:
            dist = (bb_upper - high) / bb_upper * 100
            reasons.append(f"BB {dist:.1f}% away")
        if touch_upper and not close_inside:
            reasons.append("close above BB")
        if touch_upper and close_inside and not prev_below_bb:
            reasons.append("breakout — prev candle above BB")
        logger.debug("SKIP SHORT %s: %s", symbol, " | ".join(reasons))
        return None

    # Supply zone confluence — mandatory
    z_15m = zones_15m if zones_15m is not None else find_supply_zones(df)
    z_h1 = zones_h1 or []
    z_h4 = zones_h4 or []
    confluence = score_supply_confluence(high, z_15m, z_h1, z_h4)
    if confluence == 0:
        logger.debug("SKIP SHORT %s: no supply zone near %.2f", symbol, high)
        return None

    # BB mid room — mandatory
    if bb_mid >= close:
        logger.debug("SKIP SHORT %s: price below BB mid (no room for TP)", symbol)
        return None

    # ── Optional conditions (score 4/4 = full, 2+/4 = 75%) ──
    _, overbought_threshold = get_rsi_thresholds(cfg, bias)
    opt_wick = wick_pct >= 0.15
    opt_bearish = close < opn
    opt_rsi = rsi >= overbought_threshold
    opt_vol = volume > vol_sma * cfg.volume_range_mult if vol_sma > 0 else True

    opt_count = sum([opt_wick, opt_bearish, opt_rsi, opt_vol])
    if opt_count < 2:
        reasons = []
        if not opt_wick:
            reasons.append(f"wick {wick_pct:.0%} < 15%")
        if not opt_bearish:
            reasons.append("bullish candle")
        if not opt_rsi:
            reasons.append(f"RSI {rsi:.0f} < {overbought_threshold}")
        if not opt_vol:
            ratio = volume / vol_sma if vol_sma > 0 else 0
            reasons.append(f"vol {ratio:.1f}x < {cfg.volume_range_mult}x")
        logger.debug("SKIP SHORT %s: %s (%d/4 optional)", symbol, " | ".join(reasons), opt_count)
        return None

    condition_pct = 1.0 if opt_count == 4 else 0.75
    entry = close
    sl = entry + cfg.main_sl_ranging_atr_mult * atr
    tp1 = entry - (entry - bb_mid) * 0.50

    tag = "100%" if opt_count == 4 else f"75%({opt_count}/4)"
    return Signal(
        type=SignalType.SHORT,
        source=SignalSource.MAIN,
        regime=Regime.RANGING,
        entry_price=entry,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=None,
        atr=atr,
        reason=f"RANGE SHORT | BB wick {wick_pct:.0%} + RSI {rsi:.0f} + supply x{confluence} [{tag}]",
        confluence_score=confluence,
        condition_pct=condition_pct,
    )


# ── TRENDING MODE (breakout on 1H, trailing SL only) ─────────────

def scan_trending_1h(df_1h: pd.DataFrame, cfg: TrendingConfig, bias: HTFBias) -> Signal | None:
    """Breakout momentum on 1H: ADX rising, volume surge, DI alignment, body strength."""
    if len(df_1h) < cfg.lookback + 5:
        return None

    row = df_1h.iloc[-2]  # Use closed candle
    prev = df_1h.iloc[-3]

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

    row = df.iloc[-2]  # Use closed candle — live candle wick/close unreliable

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
    # Mandatory: ADX, DI, EMA order already checked above
    long_di_ok = plus_di > minus_di
    long_ema_ok = ema_f > ema_m
    if not (long_di_ok and long_ema_ok):
        # Check short side instead
        pass
    else:
        lower_wick = min(close, opn) - low
        wick_pct = lower_wick / candle_range
        ema21_dist = (low - ema_m) / ema_m * 100 if ema_m > 0 else 99
        ema50_dist = (low - ema_s) / ema_s * 100 if ema_s > 0 else 99

        # Mandatory: EMA touch + close above EMA
        ema_level = None
        ema_label = ""
        if low <= ema_m * 1.002 and close > ema_m:
            ema_level, ema_label = ema_m, "EMA21"
        elif ema_s > 0 and low <= ema_s * 1.002 and close > ema_s:
            ema_level, ema_label = ema_s, "EMA50"

        if ema_level is not None:
            # Optional conditions (4): bias, wick, RSI, candle direction
            opt_bias = bias in (HTFBias.BULLISH, HTFBias.NEUTRAL)
            opt_wick = wick_pct >= 0.4
            opt_rsi = 40 < rsi < 70
            opt_candle = close > opn
            opt_count = sum([opt_bias, opt_wick, opt_rsi, opt_candle])

            if opt_count >= 2:
                condition_pct = 1.0 if opt_count == 4 else 0.75
                entry = ema_level
                sl = ema_level - 1.5 * atr
                tp = entry + 2.0 * (entry - sl)
                tag = "100%" if opt_count == 4 else f"75%({opt_count}/4)"
                return Signal(
                    type=SignalType.LONG, source=SignalSource.MAIN,
                    regime=Regime.TRENDING, entry_price=entry,
                    sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
                    reason=f"PB LONG | {ema_label} wick {wick_pct:.0%} ADX {adx:.0f} [{tag}]",
                    condition_pct=condition_pct,
                )
            else:
                reasons = []
                if not opt_bias:
                    reasons.append(f"bias={bias.value}")
                if not opt_wick:
                    reasons.append(f"wick {wick_pct:.0%} < 40%")
                if not opt_rsi:
                    reasons.append(f"RSI {rsi:.0f} out 40-70")
                if not opt_candle:
                    reasons.append("bearish candle")
                logger.debug("SKIP PB LONG %s: %s (%d/4 optional)", symbol, " | ".join(reasons), opt_count)
        else:
            reasons = []
            if ema21_dist > 0.2:
                reasons.append(f"EMA21 {ema21_dist:.1f}% away")
            elif wick_pct < 0.4:
                reasons.append(f"EMA21 touch but close below")
            if ema50_dist > 0.2:
                reasons.append(f"EMA50 {ema50_dist:.1f}% away")
            logger.debug("SKIP PB LONG %s: %s", symbol, " | ".join(reasons))

    # ── SHORT ──
    short_di_ok = minus_di > plus_di
    short_ema_ok = ema_f < ema_m
    if not (short_di_ok and short_ema_ok):
        pass
    else:
        upper_wick = high - max(close, opn)
        wick_pct = upper_wick / candle_range
        ema21_dist = (ema_m - high) / ema_m * 100 if ema_m > 0 else 99
        ema50_dist = (ema_s - high) / ema_s * 100 if ema_s > 0 else 99

        # Mandatory: EMA touch + close below EMA
        ema_level = None
        ema_label = ""
        if high >= ema_m * 0.998 and close < ema_m:
            ema_level, ema_label = ema_m, "EMA21"
        elif ema_s > 0 and high >= ema_s * 0.998 and close < ema_s:
            ema_level, ema_label = ema_s, "EMA50"

        if ema_level is not None:
            # Optional conditions (4): bias, wick, RSI, candle direction
            opt_bias = bias in (HTFBias.BEARISH, HTFBias.NEUTRAL)
            opt_wick = wick_pct >= 0.4
            opt_rsi = 30 < rsi < 60
            opt_candle = close < opn
            opt_count = sum([opt_bias, opt_wick, opt_rsi, opt_candle])

            if opt_count >= 2:
                condition_pct = 1.0 if opt_count == 4 else 0.75
                entry = ema_level
                sl = ema_level + 1.5 * atr
                tp = entry - 2.0 * (sl - entry)
                tag = "100%" if opt_count == 4 else f"75%({opt_count}/4)"
                return Signal(
                    type=SignalType.SHORT, source=SignalSource.MAIN,
                    regime=Regime.TRENDING, entry_price=entry,
                    sl_price=sl, tp1_price=tp, tp2_price=None, atr=atr,
                    reason=f"PB SHORT | {ema_label} wick {wick_pct:.0%} ADX {adx:.0f} [{tag}]",
                    condition_pct=condition_pct,
                )
            else:
                reasons = []
                if not opt_bias:
                    reasons.append(f"bias={bias.value}")
                if not opt_wick:
                    reasons.append(f"wick {wick_pct:.0%} < 40%")
                if not opt_rsi:
                    reasons.append(f"RSI {rsi:.0f} out 30-60")
                if not opt_candle:
                    reasons.append("bullish candle")
                logger.debug("SKIP PB SHORT %s: %s (%d/4 optional)", symbol, " | ".join(reasons), opt_count)
        else:
            reasons = []
            if ema21_dist > 0.2:
                reasons.append(f"EMA21 {ema21_dist:.1f}% away")
            if ema50_dist > 0.2:
                reasons.append(f"EMA50 {ema50_dist:.1f}% away")
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
              zones_h1_demand: list | None = None, zones_h1_supply: list | None = None,
              zones_h4_demand: list | None = None, zones_h4_supply: list | None = None) -> Signal | None:
    """Scan 15m for ranging signals — BB mean reversion works in any regime."""
    regime = detect_regime(df_15m, cfg)
    logger.info(
        "Regime: %s | HTF: %s | ADX: %.1f",
        regime.value, bias.value, df_15m.iloc[-1].get("adx", 0),
    )

    # 15m zones computed from current dataframe
    zones_15m_demand = find_demand_zones(df_15m)
    zones_15m_supply = find_supply_zones(df_15m)

    signal = check_ranging_long(df_15m, cfg, bias, symbol,
                                zones_15m=zones_15m_demand,
                                zones_h1=zones_h1_demand or [],
                                zones_h4=zones_h4_demand or [])
    if signal:
        return signal
    return check_ranging_short(df_15m, cfg, bias, symbol,
                               zones_15m=zones_15m_supply,
                               zones_h1=zones_h1_supply or [],
                               zones_h4=zones_h4_supply or [])


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
