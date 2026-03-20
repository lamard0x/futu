import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

from src.indicators import build_dataframe, add_ema, add_rsi, add_atr, add_volume_sma
from src.swing_config import SwingConfig

logger = logging.getLogger("futu.swing")


@dataclass
class SDZone:
    low: float
    high: float
    zone_type: str   # "demand" or "supply"
    fresh: bool = True
    strength: int = 1


@dataclass
class SwingSignal:
    symbol: str
    exchange: str
    side: str              # "LONG" or "SHORT"
    entry_low: float
    entry_high: float
    sl: float
    tp1: float
    tp2: float
    rr1: float
    rr2: float
    size_usd: float
    risk_usd: float
    # Mandatory conditions
    ema_ok: bool = False
    ema_detail: str = ""
    sd_zone_ok: bool = False
    sd_zone_detail: str = ""
    location_ok: bool = False
    location_detail: str = ""
    # Optional conditions (score 0-5)
    fib_ok: bool = False
    fib_detail: str = ""
    rsi_div_ok: bool = False
    rsi_div_detail: str = ""
    pin_bar_ok: bool = False
    pin_bar_detail: str = ""
    volume_ok: bool = False
    volume_detail: str = ""
    bos_ok: bool = False
    bos_detail: str = ""
    # Bonus
    nested_sd: bool = False
    bonus_notes: list = field(default_factory=list)

    @property
    def optional_score(self) -> int:
        return sum([
            self.fib_ok, self.rsi_div_ok, self.pin_bar_ok,
            self.volume_ok, self.bos_ok,
        ])

    @property
    def mandatory_pass(self) -> bool:
        return self.ema_ok and self.sd_zone_ok and self.location_ok


# ── Exchange helpers ──

async def create_public_exchange(name: str) -> ccxt.Exchange:
    exchange_classes = {
        "binance": ccxt.binance,
        "okx": ccxt.okx,
        "bybit": ccxt.bybit,
    }
    cls = exchange_classes.get(name)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {name}")
    ex = cls({"enableRateLimit": True})
    await ex.load_markets()
    return ex


async def get_top_symbols(exchange: ccxt.Exchange, limit: int = 20) -> list[str]:
    tickers = await exchange.fetch_tickers()
    spot_pairs = []
    for sym, t in tickers.items():
        if "/USDT" not in sym or ":" in sym:
            continue
        vol = float(t.get("quoteVolume") or 0)
        if vol <= 0:
            last = float(t.get("last") or 0)
            base_vol = float(t.get("baseVolume") or 0)
            vol = base_vol * last
        if vol > 0:
            spot_pairs.append((sym, vol))
    spot_pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in spot_pairs[:limit]]


async def fetch_multi_tf(
    exchange: ccxt.Exchange, symbol: str, delay: float = 0.3,
) -> dict[str, pd.DataFrame]:
    result = {}
    for tf, limit in [("1d", 200), ("4h", 200), ("1w", 100)]:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, tf, limit=limit)
            if len(ohlcv) >= 20:
                candles = [
                    {"timestamp": c[0], "open": c[1], "high": c[2],
                     "low": c[3], "close": c[4], "volume": c[5]}
                    for c in ohlcv
                ]
                result[tf] = build_dataframe(candles)
        except Exception as e:
            logger.debug("fetch %s %s %s: %s", symbol, tf, exchange.id, e)
        await asyncio.sleep(delay)
    return result


# ── Indicator helpers ──

def calc_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict:
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    diff = swing_high - swing_low
    if diff <= 0:
        return {}
    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_0236": swing_high - 0.236 * diff,
        "fib_0382": swing_high - 0.382 * diff,
        "fib_0500": swing_high - 0.500 * diff,
        "fib_0618": swing_high - 0.618 * diff,
        "fib_0660": swing_high - 0.660 * diff,
        "fib_0786": swing_high - 0.786 * diff,
    }


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df) < lookback + 5:
        return {"regular": False, "hidden": False, "type": None}

    closes = df["close"].values
    rsi_vals = df["rsi"].values
    recent = slice(-lookback, None)

    c = closes[recent]
    r = rsi_vals[recent]

    if len(c) < 10 or np.any(np.isnan(r)):
        return {"regular": False, "hidden": False, "type": None}

    mid = len(c) // 2
    # Find local lows in first and second half
    low1_idx = np.argmin(c[:mid])
    low2_idx = mid + np.argmin(c[mid:])
    # Find local highs
    high1_idx = np.argmax(c[:mid])
    high2_idx = mid + np.argmax(c[mid:])

    result = {"regular": False, "hidden": False, "type": None}

    # Regular bullish: price lower low, RSI higher low
    if c[low2_idx] < c[low1_idx] and r[low2_idx] > r[low1_idx]:
        result["regular"] = True
        result["type"] = "regular_bullish"

    # Regular bearish: price higher high, RSI lower high
    if c[high2_idx] > c[high1_idx] and r[high2_idx] < r[high1_idx]:
        result["regular"] = True
        result["type"] = "regular_bearish"

    # Hidden bullish: price higher low, RSI lower low
    if c[low2_idx] > c[low1_idx] and r[low2_idx] < r[low1_idx]:
        result["hidden"] = True
        result["type"] = "hidden_bullish"

    # Hidden bearish: price lower high, RSI higher high
    if c[high2_idx] < c[high1_idx] and r[high2_idx] > r[high1_idx]:
        result["hidden"] = True
        result["type"] = "hidden_bearish"

    return result


def detect_bos_or_weak_pullback(df_h4: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df_h4) < lookback:
        return {"bos": False, "weak_pullback": False, "type": None}

    recent = df_h4.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    closes = recent["close"].values

    result = {"bos": False, "weak_pullback": False, "type": None}

    # BOS bullish: latest close > previous swing high
    prev_highs = highs[:-3]
    if len(prev_highs) > 0:
        prev_swing_high = np.max(prev_highs)
        if closes[-1] > prev_swing_high:
            result["bos"] = True
            result["type"] = "bullish_bos"

    # BOS bearish: latest close < previous swing low
    prev_lows = lows[:-3]
    if len(prev_lows) > 0:
        prev_swing_low = np.min(prev_lows)
        if closes[-1] < prev_swing_low:
            result["bos"] = True
            result["type"] = "bearish_bos"

    # Weak pullback: small retracement (< 38.2% of last move)
    if not result["bos"] and len(closes) >= 10:
        last_move = abs(closes[-1] - closes[-10])
        max_retrace = 0
        for i in range(-5, 0):
            retrace = abs(closes[i] - closes[-1])
            max_retrace = max(max_retrace, retrace)
        if last_move > 0 and max_retrace / last_move < 0.382:
            result["weak_pullback"] = True
            if closes[-1] > closes[-10]:
                result["type"] = "bullish_weak_pb"
            else:
                result["type"] = "bearish_weak_pb"

    return result


def check_pin_bar(row: pd.Series) -> dict:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    body = abs(c - o)
    full_range = h - l
    if full_range <= 0:
        return {"is_pin": False, "type": None}

    body_ratio = body / full_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    result = {"is_pin": False, "type": None}

    # Bullish pin bar: long lower wick, small body at top
    if body_ratio < 0.35 and lower_wick > 2 * body and lower_wick > upper_wick * 2:
        result["is_pin"] = True
        result["type"] = "bullish_pin"

    # Bearish pin bar: long upper wick, small body at bottom
    if body_ratio < 0.35 and upper_wick > 2 * body and upper_wick > lower_wick * 2:
        result["is_pin"] = True
        result["type"] = "bearish_pin"

    # Bullish engulfing
    if len(row.index) > 0 and c > o and body_ratio > 0.6:
        result["is_pin"] = True
        result["type"] = "bullish_engulfing"

    return result


def check_location(price: float, wk_high: float, wk_low: float) -> dict:
    wk_range = wk_high - wk_low
    if wk_range <= 0:
        return {"ok": False, "pct": 50}
    pct = (price - wk_low) / wk_range * 100
    return {"ok": True, "pct": round(pct, 1)}


def find_fresh_sd_zones(df: pd.DataFrame, lookback: int = 100) -> list[SDZone]:
    zones = []
    recent = df.tail(lookback)
    if len(recent) < 20:
        return zones

    closes = recent["close"].values
    highs = recent["high"].values
    lows = recent["low"].values
    opens = recent["open"].values

    for i in range(2, len(recent) - 2):
        # Demand zone: strong move up from a base
        if (closes[i + 1] > highs[i] and
                closes[i + 1] > closes[i] * 1.01 and
                lows[i] < lows[i - 1]):
            zone_low = min(lows[i - 1], lows[i])
            zone_high = max(opens[i], closes[i])
            # Check freshness: price hasn't revisited
            fresh = True
            for j in range(i + 2, len(recent)):
                if lows[j] <= zone_high:
                    fresh = False
                    break
            zones.append(SDZone(
                low=zone_low, high=zone_high,
                zone_type="demand", fresh=fresh,
            ))

        # Supply zone: strong move down from a base
        if (closes[i + 1] < lows[i] and
                closes[i + 1] < closes[i] * 0.99 and
                highs[i] > highs[i - 1]):
            zone_low = min(opens[i], closes[i])
            zone_high = max(highs[i - 1], highs[i])
            fresh = True
            for j in range(i + 2, len(recent)):
                if highs[j] >= zone_low:
                    fresh = False
                    break
            zones.append(SDZone(
                low=zone_low, high=zone_high,
                zone_type="supply", fresh=fresh,
            ))

    return zones


def check_nested_sd(d1_zone: SDZone, wk_zones: list[SDZone]) -> bool:
    for wk in wk_zones:
        if wk.zone_type != d1_zone.zone_type:
            continue
        if d1_zone.low >= wk.low and d1_zone.high <= wk.high:
            return True
    return False


# ── Scoring ──

def score_swing_setup(
    side: str, price: float,
    df_daily: pd.DataFrame, df_h4: pd.DataFrame | None,
    df_weekly: pd.DataFrame | None, cfg: SwingConfig,
) -> SwingSignal | None:
    if len(df_daily) < 50:
        return None

    add_ema(df_daily, cfg.ema_fast)
    add_ema(df_daily, cfg.ema_slow)
    add_rsi(df_daily, cfg.rsi_period)
    add_atr(df_daily, 14)
    add_volume_sma(df_daily, cfg.volume_sma_period)

    last = df_daily.iloc[-1]
    prev = df_daily.iloc[-2]
    ema_fast_col = f"ema_{cfg.ema_fast}"
    ema_slow_col = f"ema_{cfg.ema_slow}"
    ema_fast_val = last.get(ema_fast_col, 0)
    ema_slow_val = last.get(ema_slow_col, 0)
    atr = last.get("atr", 0)
    if atr <= 0:
        return None

    signal = SwingSignal(
        symbol="", exchange="", side=side,
        entry_low=0, entry_high=0, sl=0,
        tp1=0, tp2=0, rr1=0, rr2=0,
        size_usd=0, risk_usd=0,
    )

    # ── Mandatory 1: EMA trend ──
    if side == "LONG" and ema_fast_val > ema_slow_val:
        signal.ema_ok = True
        signal.ema_detail = f"EMA{cfg.ema_fast} > EMA{cfg.ema_slow} (bullish trend)"
    elif side == "SHORT" and ema_fast_val < ema_slow_val:
        signal.ema_ok = True
        signal.ema_detail = f"EMA{cfg.ema_fast} < EMA{cfg.ema_slow} (bearish trend)"

    # ── Mandatory 2: S/D zone ──
    daily_zones = find_fresh_sd_zones(df_daily, cfg.sd_zone_lookback)
    target_type = "demand" if side == "LONG" else "supply"
    matching_zone = None
    for z in daily_zones:
        if z.zone_type == target_type and z.low <= price <= z.high:
            matching_zone = z
            signal.sd_zone_ok = True
            fresh_tag = "FRESH" if z.fresh else "TESTED"
            signal.sd_zone_detail = (
                f"Daily {target_type} zone {_price_fmt(z.low)}-{_price_fmt(z.high)} [{fresh_tag}]"
            )
            break

    # H4 zones for additional confluence
    h4_zones = []
    if df_h4 is not None and len(df_h4) >= 20:
        h4_zones = find_fresh_sd_zones(df_h4, cfg.sd_zone_lookback)
        for z in h4_zones:
            if z.zone_type == target_type and z.low <= price <= z.high:
                if not signal.sd_zone_ok:
                    matching_zone = z
                    signal.sd_zone_ok = True
                    fresh_tag = "FRESH" if z.fresh else "TESTED"
                    signal.sd_zone_detail = (
                        f"H4 {target_type} zone {_price_fmt(z.low)}-{_price_fmt(z.high)} [{fresh_tag}]"
                    )
                break

    # ── Mandatory 3: Location rule ──
    if df_weekly is not None and len(df_weekly) >= 10:
        wk_high = df_weekly["high"].max()
        wk_low = df_weekly["low"].min()
        loc = check_location(price, wk_high, wk_low)
        if side == "LONG" and loc["pct"] < 30:
            signal.location_ok = True
            signal.location_detail = f"Location: {loc['pct']}% (Weekly range) - OK"
        elif side == "SHORT" and loc["pct"] > 70:
            signal.location_ok = True
            signal.location_detail = f"Location: {loc['pct']}% (Weekly range) - OK"

    if not signal.mandatory_pass:
        return None

    # ── Optional 4: Fib golden pocket ──
    fibs = calc_fibonacci_levels(df_daily, cfg.fib_lookback)
    if fibs:
        fib_618 = fibs.get("fib_0618", 0)
        fib_660 = fibs.get("fib_0660", 0)
        if fib_618 and fib_660:
            if side == "LONG" and fib_660 <= price <= fib_618:
                signal.fib_ok = True
                signal.fib_detail = f"Fib 0.618 at {_price_fmt(fib_618)}"
            elif side == "SHORT" and fib_618 <= price <= fib_660:
                signal.fib_ok = True
                signal.fib_detail = f"Fib 0.618 at {_price_fmt(fib_618)}"

    # ── Optional 5: RSI divergence ──
    div = detect_rsi_divergence(df_daily, 20)
    if side == "LONG" and div["type"] in ("regular_bullish", "hidden_bullish"):
        signal.rsi_div_ok = True
        label = "Regular" if div["regular"] else "Hidden"
        signal.rsi_div_detail = f"{label} bullish RSI divergence"
    elif side == "SHORT" and div["type"] in ("regular_bearish", "hidden_bearish"):
        signal.rsi_div_ok = True
        label = "Regular" if div["regular"] else "Hidden"
        signal.rsi_div_detail = f"{label} bearish RSI divergence"

    # ── Optional 6: Pin bar / engulfing ──
    pin = check_pin_bar(last)
    if pin["is_pin"]:
        if side == "LONG" and pin["type"] in ("bullish_pin", "bullish_engulfing"):
            signal.pin_bar_ok = True
            signal.pin_bar_detail = f"{pin['type'].replace('_', ' ').title()} at zone"
        elif side == "SHORT" and pin["type"] in ("bearish_pin", "bearish_engulfing"):
            signal.pin_bar_ok = True
            signal.pin_bar_detail = f"{pin['type'].replace('_', ' ').title()} at zone"

    # ── Optional 7: Volume above SMA ──
    vol = last.get("volume", 0)
    vol_sma = last.get("volume_sma", 0)
    if vol_sma > 0 and vol > vol_sma:
        signal.volume_ok = True
        ratio = vol / vol_sma
        signal.volume_detail = f"Volume {ratio:.1f}x above avg"

    # ── Optional 8: BOS or weak pullback on H4 ──
    if df_h4 is not None and len(df_h4) >= 20:
        bos = detect_bos_or_weak_pullback(df_h4, cfg.bos_lookback)
        if bos["bos"] or bos["weak_pullback"]:
            bos_type = bos["type"] or ""
            if side == "LONG" and "bullish" in bos_type:
                signal.bos_ok = True
                label = "BOS" if bos["bos"] else "Weak pullback"
                signal.bos_detail = f"H4 {label}"
            elif side == "SHORT" and "bearish" in bos_type:
                signal.bos_ok = True
                label = "BOS" if bos["bos"] else "Weak pullback"
                signal.bos_detail = f"H4 {label}"

    if signal.optional_score < cfg.min_optional_score:
        return None

    # ── Bonus: Nested S/D (Level on Level) ──
    if matching_zone and df_weekly is not None and len(df_weekly) >= 20:
        wk_zones = find_fresh_sd_zones(df_weekly, 50)
        if check_nested_sd(matching_zone, wk_zones):
            signal.nested_sd = True
            signal.bonus_notes.append("LOL (D1 zone inside Weekly zone)")

    # ── Entry / SL / TP calc ──
    if matching_zone:
        signal.entry_low = matching_zone.low
        signal.entry_high = matching_zone.high
    else:
        signal.entry_low = price - atr * 0.2
        signal.entry_high = price + atr * 0.2

    entry_mid = (signal.entry_low + signal.entry_high) / 2

    if side == "LONG":
        signal.sl = signal.entry_low - atr * 1.5
        sl_dist = entry_mid - signal.sl
        signal.tp1 = entry_mid + sl_dist * 2.0
        signal.tp2 = entry_mid + sl_dist * 3.5
    else:
        signal.sl = signal.entry_high + atr * 1.5
        sl_dist = signal.sl - entry_mid
        signal.tp1 = entry_mid - sl_dist * 2.0
        signal.tp2 = entry_mid - sl_dist * 3.5

    if sl_dist > 0:
        signal.rr1 = abs(signal.tp1 - entry_mid) / sl_dist
        signal.rr2 = abs(signal.tp2 - entry_mid) / sl_dist

    if signal.rr1 < cfg.min_rr:
        return None

    return signal


def calc_swing_size(
    capital: float, risk_pct: float,
    entry: float, sl: float, max_positions: int,
) -> tuple[float, float]:
    risk_usd = capital * risk_pct
    sl_distance_pct = abs(entry - sl) / entry if entry > 0 else 1
    if sl_distance_pct <= 0:
        return 0.0, risk_usd
    size = risk_usd / sl_distance_pct
    max_per_position = capital / max_positions
    size = min(size, max_per_position)
    return round(size, 2), round(risk_usd, 2)


def deduplicate(signals: list[SwingSignal]) -> list[SwingSignal]:
    best = {}
    for s in signals:
        coin = s.symbol.split("/")[0]
        existing = best.get(coin)
        if existing is None or s.optional_score > existing.optional_score:
            best[coin] = s
        elif s.optional_score == existing.optional_score and s.rr1 > existing.rr1:
            best[coin] = s
    result = list(best.values())
    result.sort(key=lambda x: (x.optional_score, x.rr1), reverse=True)
    return result


def _price_fmt(price: float) -> str:
    """Dynamic precision: $0.001234 → 4 decimals, $1.23 → 4, $123.4 → 2, $1234 → 1."""
    if price <= 0:
        return "$0"
    if price < 0.01:
        return f"${price:.6f}"
    if price < 1:
        return f"${price:.4f}"
    if price < 100:
        return f"${price:.4f}"
    if price < 10000:
        return f"${price:.2f}"
    return f"${price:.1f}"


def format_swing_telegram(signal: SwingSignal) -> str:
    coin = signal.symbol.split("/")[0]
    pf = _price_fmt
    side_emoji = "\U0001f7e2" if signal.side == "LONG" else "\U0001f534"

    size_line = (
        f"\U0001f4b0 Size: ${signal.size_usd:.0f} spot"
        f" (risk ${signal.risk_usd:.0f} = {signal.risk_usd / signal.size_usd * 100:.0f}%)"
        if signal.size_usd > 0 else f"\U0001f4b0 Size: $0 (risk ${signal.risk_usd:.0f})"
    )

    lines = [
        f"{side_emoji} <b>SWING {signal.side} {coin}/USDT</b>  ({signal.exchange.upper()})",
        "",
        f"\U0001f4cd Entry: {pf(signal.entry_low)} \u2013 {pf(signal.entry_high)}",
        f"\U0001f6d1 SL: {pf(signal.sl)}",
        f"\U0001f3af TP1: {pf(signal.tp1)}  (1:{signal.rr1:.1f})",
        f"\U0001f3af TP2: {pf(signal.tp2)}  (1:{signal.rr2:.1f})",
        size_line,
        "",
        f"<b>\u2705 MANDATORY  3/3</b>",
        f"  \u2022 {signal.ema_detail}",
        f"  \u2022 {signal.sd_zone_detail}",
        f"  \u2022 {signal.location_detail}",
        "",
        f"<b>\u2b50 OPTIONAL  {signal.optional_score}/5</b>",
    ]

    optional_checks = [
        (signal.fib_ok, signal.fib_detail, "Fib golden pocket"),
        (signal.rsi_div_ok, signal.rsi_div_detail, "RSI divergence"),
        (signal.pin_bar_ok, signal.pin_bar_detail, "Pin bar / engulfing"),
        (signal.volume_ok, signal.volume_detail, "Volume above avg"),
        (signal.bos_ok, signal.bos_detail, "H4 BOS"),
    ]
    for ok, detail, fallback in optional_checks:
        mark = "\u2705" if ok else "\u274c"
        text = detail if ok else f"No {fallback}"
        lines.append(f"  {mark} {text}")

    if signal.bonus_notes:
        lines.append("")
        lines.append(f"\U0001f48e {', '.join(signal.bonus_notes)}")

    return "\n".join(lines)


# ── Main scan orchestrator ──

async def run_swing_scan(cfg: SwingConfig) -> list[SwingSignal]:
    all_signals = []

    for ex_name in cfg.exchanges:
        exchange = None
        try:
            exchange = await create_public_exchange(ex_name)
            symbols = await get_top_symbols(exchange, cfg.top_symbols)
            logger.info("Swing scan %s: %d symbols", ex_name, len(symbols))

            for symbol in symbols:
                try:
                    data = await fetch_multi_tf(exchange, symbol, cfg.rate_limit_delay)
                    df_daily = data.get("1d")
                    df_h4 = data.get("4h")
                    df_weekly = data.get("1w")

                    if df_daily is None or len(df_daily) < 50:
                        continue

                    price = df_daily.iloc[-1]["close"]

                    for side in ("LONG", "SHORT"):
                        sig = score_swing_setup(
                            side, price, df_daily.copy(), df_h4, df_weekly, cfg,
                        )
                        if sig is None:
                            continue

                        sig.symbol = symbol
                        sig.exchange = ex_name
                        entry_mid = (sig.entry_low + sig.entry_high) / 2
                        sig.size_usd, sig.risk_usd = calc_swing_size(
                            cfg.capital, cfg.risk_pct,
                            entry_mid, sig.sl, cfg.max_positions,
                        )
                        all_signals.append(sig)
                        logger.info(
                            "Swing signal: %s %s %s score=%d R:R=%.1f",
                            side, symbol, ex_name, sig.optional_score, sig.rr1,
                        )
                except Exception as e:
                    logger.debug("Swing scan %s %s error: %s", ex_name, symbol, e)
        except Exception as e:
            logger.warning("Swing exchange %s error: %s", ex_name, e)
        finally:
            if exchange:
                try:
                    await exchange.close()
                except Exception:
                    pass

    deduped = deduplicate(all_signals)
    logger.info("Swing scan complete: %d raw → %d deduped signals",
                len(all_signals), len(deduped))
    return deduped
