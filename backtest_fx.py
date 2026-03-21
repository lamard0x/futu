"""
FUTU FX Backtest — Session-based strategy, news filter, money management
Exness MT5 data, spread-based fees
Usage: python backtest_fx.py [days] [timeframe]
  e.g. python backtest_fx.py 60 15m
"""
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from src.indicators import compute_all
from src.config import IndicatorConfig, StrategyConfig, RiskConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backtest_fx")

# ═══ Config ═══
IND_CFG = IndicatorConfig()
STRAT_CFG = StrategyConfig()
RISK_CFG = RiskConfig()
BALANCE_START = 200.0
LEVERAGE = 200
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# ═══ Session Config ═══
# Asian: 00:00-07:00 UTC — BB ranging only (mean reversion)
# London: 07:00-16:00 UTC — Trending pullback/breakout only
# NY: 12:00-21:00 UTC — Trending (continues London, or reversal)
# Off hours: 21:00-00:00 UTC — No trading

SESSION_ASIAN = "asian"
SESSION_LONDON = "london"
SESSION_NY = "ny"
SESSION_OFF = "off"

ASIAN_SYMBOLS = ["EURUSDm", "GBPUSDm", "AUDUSDm", "USDJPYm"]
TRENDING_SYMBOLS = ["GBPUSDm", "GBPJPYm", "XAUUSDm"]
ALL_SYMBOLS = list(set(ASIAN_SYMBOLS + TRENDING_SYMBOLS))

# Spread cost in pips (typical Exness micro)
SPREAD_PIPS = {
    "EURUSDm": 1.2, "GBPUSDm": 1.5, "USDJPYm": 1.3,
    "AUDUSDm": 1.4, "GBPJPYm": 2.5, "XAUUSDm": 25.0,
}
PIP_SIZE = {
    "EURUSDm": 0.0001, "GBPUSDm": 0.0001, "USDJPYm": 0.01,
    "AUDUSDm": 0.0001, "GBPJPYm": 0.01, "XAUUSDm": 0.1,
}
PIP_VALUE = {
    "EURUSDm": 10.0, "GBPUSDm": 10.0, "USDJPYm": 6.5,
    "AUDUSDm": 10.0, "GBPJPYm": 6.5, "XAUUSDm": 10.0,
}

# Money management
RISK_RANGING = 0.02   # 2% per ranging trade
RISK_TRENDING = 0.02  # 2% per trending trade
MAX_DAILY_LOSS = 1.0  # disabled — let it run
MAX_CONCURRENT = 999  # unlimited — 1 per symbol, daily loss cap is the safety net

# TP/SL params
RANGING_SL_ATR = 0.5
RANGING_MIN_RR = 1.3
TRENDING_SL_ATR = 1.5
TRENDING_MIN_RR = 1.5
TRAILING_BE_PCT = 0.50   # move SL to breakeven at 50% of TP
PARTIAL_CLOSE_PCT = 0.50  # close 50% at TP1

# Trending detection
T_ADX_MIN = 30
T_LOOKBACK = 20
T_BODY_PCT = 0.5
T_VOL_MULT = 1.2

# ═══ News Calendar ═══
CALENDAR_PATH = Path(__file__).parent / "data" / "fx_calendar_2026.json"

# Hardcoded recurring high-impact events (fallback)
RECURRING_EVENTS = [
    # NFP — first Friday of month, 13:30 UTC
    {"recurring": "nfp", "currencies": ["USD"], "impact": "high",
     "event_name": "Non-Farm Payrolls"},
    # FOMC — 8 times/year, ~18:00 UTC (Wed)
    {"recurring": "fomc", "currencies": ["USD"], "impact": "high",
     "event_name": "FOMC Statement"},
    # ECB Rate Decision — ~12:45 UTC
    {"recurring": "ecb", "currencies": ["EUR"], "impact": "high",
     "event_name": "ECB Interest Rate Decision"},
    # BOE Rate Decision
    {"recurring": "boe", "currencies": ["GBP"], "impact": "high",
     "event_name": "BOE Interest Rate Decision"},
    # RBA Rate Decision
    {"recurring": "rba", "currencies": ["AUD"], "impact": "high",
     "event_name": "RBA Interest Rate Decision"},
    # BOJ Rate Decision
    {"recurring": "boj", "currencies": ["JPY"], "impact": "high",
     "event_name": "BOJ Interest Rate Decision"},
    # CPI events (monthly, approximate)
    {"recurring": "cpi_us", "currencies": ["USD"], "impact": "high",
     "event_name": "CPI m/m"},
    {"recurring": "cpi_uk", "currencies": ["GBP"], "impact": "high",
     "event_name": "CPI y/y"},
]

# Currency mapping for symbol → affected currencies
SYMBOL_CURRENCIES = {
    "EURUSDm": ["EUR", "USD"],
    "GBPUSDm": ["GBP", "USD"],
    "USDJPYm": ["USD", "JPY"],
    "AUDUSDm": ["AUD", "USD"],
    "GBPJPYm": ["GBP", "JPY"],
    "XAUUSDm": ["XAU", "USD"],
}


@dataclass
class NewsEvent:
    dt: datetime
    currency: str
    impact: str  # "high" or "medium"
    event_name: str


@dataclass
class Trade:
    symbol: str
    side: str
    regime: str
    session: str
    entry_price: float
    sl_price: float
    tp_price: float
    lots: float
    lots_remaining: float
    notional: float
    entry_time: str
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    pnl_gross: float = 0.0
    spread_cost: float = 0.0
    pnl_net: float = 0.0
    candles_held: int = 0
    partial_closed: bool = False
    partial_pnl: float = 0.0


@dataclass
class BacktestState:
    balance: float = BALANCE_START
    trades: list = field(default_factory=list)
    open_positions: list = field(default_factory=list)
    daily_loss: float = 0.0
    daily_date: str = ""
    wins: int = 0
    losses: int = 0
    max_drawdown: float = 0.0
    peak_balance: float = BALANCE_START
    news_reaction_trades: int = 0
    session_stats: dict = field(default_factory=lambda: {
        SESSION_ASIAN: {"trades": 0, "wins": 0, "pnl": 0.0},
        SESSION_LONDON: {"trades": 0, "wins": 0, "pnl": 0.0},
        SESSION_NY: {"trades": 0, "wins": 0, "pnl": 0.0},
        "news": {"trades": 0, "wins": 0, "pnl": 0.0},
    })


# ═══ MT5 Data ═══

TF_MAP = {
    "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def fetch_mt5(symbol: str, tf_str: str, since: datetime) -> list[dict]:
    tf = TF_MAP.get(tf_str)
    if tf is None:
        return []
    rates = mt5.copy_rates_range(symbol, tf, since, datetime.now(timezone.utc))
    if rates is None or len(rates) == 0:
        return []
    return [
        {"timestamp": int(r[0]) * 1000, "open": r[1], "high": r[2],
         "low": r[3], "close": r[4], "volume": float(r[5])}
        for r in rates
    ]


# ═══ Session Detection ═══

def get_session(dt_utc: datetime) -> str:
    """Determine trading session from UTC datetime."""
    if dt_utc.weekday() >= 5:
        return SESSION_OFF
    hour = dt_utc.hour
    if 0 <= hour < 9:
        return SESSION_ASIAN  # Asian + early London (07-09) for ranging
    if 9 <= hour < 12:
        return SESSION_LONDON
    if 12 <= hour < 16:
        # London/NY overlap — treat as London for trending
        return SESSION_LONDON
    if 16 <= hour < 21:
        return SESSION_NY
    return SESSION_OFF


def is_weekend(ts_sec: int) -> bool:
    dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    return dt.weekday() >= 5


def bar_to_utc(bar_time, row) -> datetime:
    """Convert bar index to UTC datetime."""
    ts_sec = int(row.get("timestamp", 0)) // 1000
    if ts_sec > 0:
        return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    if hasattr(bar_time, 'year'):
        if bar_time.tzinfo is None:
            return bar_time.replace(tzinfo=timezone.utc)
        return bar_time
    return datetime.now(timezone.utc)


# ═══ News Calendar — Comprehensive 2026 Calendar ═══

def _dt(y: int, m: int, d: int, h: int, mi: int = 0) -> str:
    """Helper: create ISO datetime string with UTC timezone."""
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc).isoformat()


def generate_2026_calendar() -> list[dict]:
    """
    Comprehensive 2026 economic calendar with exact dates.
    Sources: central bank published schedules, BLS release calendar,
    ONS/ABS/Statistics Bureau schedules.
    Returns list of {datetime, currency, impact, event_name}.
    """
    events = []
    Y = 2026

    # ──── NFP: First Friday of each month, 13:30 UTC ────
    nfp_dates = [
        (1, 9), (2, 6), (3, 6), (4, 3), (5, 1), (6, 5),
        (7, 3), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4),
    ]
    for m, d in nfp_dates:
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "high",
            "event_name": "Non-Farm Payrolls",
        })
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "high",
            "event_name": "Unemployment Rate",
        })
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "medium",
            "event_name": "Average Hourly Earnings m/m",
        })

    # ──── US CPI: ~13th of month, 13:30 UTC (Tue/Wed) ────
    cpi_us_dates = [
        (1, 14), (2, 11), (3, 11), (4, 14), (5, 12), (6, 10),
        (7, 14), (8, 12), (9, 15), (10, 13), (11, 12), (12, 10),
    ]
    for m, d in cpi_us_dates:
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "high",
            "event_name": "CPI m/m",
        })
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "high",
            "event_name": "Core CPI m/m",
        })

    # ──── FOMC: 8 meetings/year, statement 18:00 UTC (Wed) ────
    fomc_dates = [
        (1, 28), (3, 18), (5, 6), (6, 17),
        (7, 29), (9, 16), (11, 4), (12, 16),
    ]
    for m, d in fomc_dates:
        events.append({
            "datetime": _dt(Y, m, d, 18, 0),
            "currency": "USD", "impact": "high",
            "event_name": "FOMC Statement",
        })
        events.append({
            "datetime": _dt(Y, m, d, 18, 0),
            "currency": "USD", "impact": "high",
            "event_name": "Federal Funds Rate",
        })
        events.append({
            "datetime": _dt(Y, m, d, 18, 30),
            "currency": "USD", "impact": "high",
            "event_name": "FOMC Press Conference",
        })

    # FOMC Minutes: ~3 weeks after meeting, 18:00 UTC (Wed)
    fomc_minutes_dates = [
        (1, 7), (2, 18), (4, 8), (5, 27),
        (7, 8), (8, 19), (10, 7), (11, 25),
    ]
    for m, d in fomc_minutes_dates:
        events.append({
            "datetime": _dt(Y, m, d, 18, 0),
            "currency": "USD", "impact": "medium",
            "event_name": "FOMC Meeting Minutes",
        })

    # ──── ECB Rate Decision: 8 meetings, 12:15 UTC (Thu) ────
    ecb_dates = [
        (1, 22), (3, 5), (4, 16), (6, 4),
        (7, 16), (9, 10), (10, 22), (12, 10),
    ]
    for m, d in ecb_dates:
        events.append({
            "datetime": _dt(Y, m, d, 12, 15),
            "currency": "EUR", "impact": "high",
            "event_name": "ECB Interest Rate Decision",
        })
        events.append({
            "datetime": _dt(Y, m, d, 12, 45),
            "currency": "EUR", "impact": "high",
            "event_name": "ECB Press Conference",
        })

    # ──── BOE Rate Decision: 8 meetings, 12:00 UTC (Thu) ────
    boe_dates = [
        (2, 5), (3, 19), (5, 7), (6, 18),
        (8, 6), (9, 17), (11, 5), (12, 17),
    ]
    for m, d in boe_dates:
        events.append({
            "datetime": _dt(Y, m, d, 12, 0),
            "currency": "GBP", "impact": "high",
            "event_name": "BOE Interest Rate Decision",
        })
        events.append({
            "datetime": _dt(Y, m, d, 12, 0),
            "currency": "GBP", "impact": "high",
            "event_name": "BOE Monetary Policy Summary",
        })

    # ──── RBA Rate Decision: 8 meetings, 03:30 UTC (Tue) ────
    rba_dates = [
        (2, 17), (3, 31), (5, 19), (7, 7),
        (8, 4), (9, 1), (11, 3), (12, 1),
    ]
    for m, d in rba_dates:
        events.append({
            "datetime": _dt(Y, m, d, 3, 30),
            "currency": "AUD", "impact": "high",
            "event_name": "RBA Interest Rate Decision",
        })
        events.append({
            "datetime": _dt(Y, m, d, 4, 30),
            "currency": "AUD", "impact": "high",
            "event_name": "RBA Rate Statement",
        })

    # ──── BOJ Rate Decision: 8 meetings, ~03:00 UTC ────
    boj_dates = [
        (1, 23), (3, 13), (4, 30), (6, 18),
        (7, 30), (9, 17), (10, 29), (12, 18),
    ]
    for m, d in boj_dates:
        events.append({
            "datetime": _dt(Y, m, d, 3, 0),
            "currency": "JPY", "impact": "high",
            "event_name": "BOJ Interest Rate Decision",
        })
        events.append({
            "datetime": _dt(Y, m, d, 6, 30),
            "currency": "JPY", "impact": "high",
            "event_name": "BOJ Press Conference",
        })

    # ──── US GDP (Advance, Prelim, Final): quarterly, 13:30 UTC ────
    gdp_dates = [
        (1, 29, "GDP q/q (Advance)"),
        (2, 26, "GDP q/q (Prelim)"),
        (3, 26, "GDP q/q (Final)"),
        (4, 29, "GDP q/q (Advance)"),
        (5, 28, "GDP q/q (Prelim)"),
        (6, 25, "GDP q/q (Final)"),
        (7, 30, "GDP q/q (Advance)"),
        (8, 27, "GDP q/q (Prelim)"),
        (9, 24, "GDP q/q (Final)"),
        (10, 29, "GDP q/q (Advance)"),
        (11, 25, "GDP q/q (Prelim)"),
        (12, 22, "GDP q/q (Final)"),
    ]
    for m, d, name in gdp_dates:
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "medium",
            "event_name": name,
        })

    # ──── US Retail Sales: monthly ~15th, 13:30 UTC ────
    retail_dates = [
        (1, 16), (2, 13), (3, 17), (4, 15), (5, 15), (6, 16),
        (7, 16), (8, 14), (9, 16), (10, 16), (11, 17), (12, 15),
    ]
    for m, d in retail_dates:
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "medium",
            "event_name": "Retail Sales m/m",
        })
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "medium",
            "event_name": "Core Retail Sales m/m",
        })

    # ──── US PPI: monthly, 13:30 UTC ────
    ppi_dates = [
        (1, 15), (2, 12), (3, 12), (4, 9), (5, 13), (6, 11),
        (7, 15), (8, 13), (9, 16), (10, 14), (11, 13), (12, 11),
    ]
    for m, d in ppi_dates:
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "medium",
            "event_name": "PPI m/m",
        })

    # ──── US ISM Manufacturing PMI: 1st business day, 15:00 UTC ────
    ism_mfg_dates = [
        (1, 5), (2, 2), (3, 2), (4, 1), (5, 1), (6, 1),
        (7, 1), (8, 3), (9, 1), (10, 1), (11, 2), (12, 1),
    ]
    for m, d in ism_mfg_dates:
        events.append({
            "datetime": _dt(Y, m, d, 15, 0),
            "currency": "USD", "impact": "medium",
            "event_name": "ISM Manufacturing PMI",
        })

    # ──── US ISM Services PMI: 3rd business day, 15:00 UTC ────
    ism_svc_dates = [
        (1, 7), (2, 4), (3, 4), (4, 3), (5, 5), (6, 3),
        (7, 7), (8, 5), (9, 3), (10, 5), (11, 4), (12, 3),
    ]
    for m, d in ism_svc_dates:
        events.append({
            "datetime": _dt(Y, m, d, 15, 0),
            "currency": "USD", "impact": "medium",
            "event_name": "ISM Services PMI",
        })

    # ──── UK CPI: monthly ~15th-17th, 07:00 UTC (Wed) ────
    uk_cpi_dates = [
        (1, 15), (2, 18), (3, 18), (4, 15), (5, 20), (6, 17),
        (7, 15), (8, 19), (9, 16), (10, 21), (11, 18), (12, 16),
    ]
    for m, d in uk_cpi_dates:
        events.append({
            "datetime": _dt(Y, m, d, 7, 0),
            "currency": "GBP", "impact": "high",
            "event_name": "CPI y/y",
        })

    # ──── UK GDP: monthly, 07:00 UTC ────
    uk_gdp_dates = [
        (1, 16), (2, 13), (3, 13), (4, 10), (5, 14), (6, 12),
        (7, 10), (8, 14), (9, 11), (10, 9), (11, 13), (12, 11),
    ]
    for m, d in uk_gdp_dates:
        events.append({
            "datetime": _dt(Y, m, d, 7, 0),
            "currency": "GBP", "impact": "medium",
            "event_name": "GDP m/m",
        })

    # ──── UK Employment / Claimant Count: monthly, 07:00 UTC (Tue) ────
    uk_emp_dates = [
        (1, 21), (2, 17), (3, 17), (4, 14), (5, 13), (6, 17),
        (7, 15), (8, 12), (9, 15), (10, 13), (11, 10), (12, 15),
    ]
    for m, d in uk_emp_dates:
        events.append({
            "datetime": _dt(Y, m, d, 7, 0),
            "currency": "GBP", "impact": "medium",
            "event_name": "Claimant Count Change",
        })

    # ──── AU Employment Change: monthly ~3rd Thu, 00:30 UTC ────
    au_emp_dates = [
        (1, 22), (2, 19), (3, 19), (4, 16), (5, 14), (6, 18),
        (7, 16), (8, 13), (9, 17), (10, 15), (11, 12), (12, 17),
    ]
    for m, d in au_emp_dates:
        events.append({
            "datetime": _dt(Y, m, d, 0, 30),
            "currency": "AUD", "impact": "high",
            "event_name": "Employment Change",
        })
        events.append({
            "datetime": _dt(Y, m, d, 0, 30),
            "currency": "AUD", "impact": "medium",
            "event_name": "Unemployment Rate",
        })

    # ──── AU CPI: quarterly, 00:30 UTC ────
    au_cpi_dates = [
        (1, 28), (4, 29), (7, 29), (10, 28),
    ]
    for m, d in au_cpi_dates:
        events.append({
            "datetime": _dt(Y, m, d, 0, 30),
            "currency": "AUD", "impact": "high",
            "event_name": "CPI q/q",
        })

    # ──── JP CPI (National): monthly ~3rd Fri, 23:30 UTC (prev day) ────
    jp_cpi_dates = [
        (1, 23), (2, 20), (3, 20), (4, 17), (5, 22), (6, 19),
        (7, 17), (8, 21), (9, 18), (10, 23), (11, 20), (12, 18),
    ]
    for m, d in jp_cpi_dates:
        events.append({
            "datetime": _dt(Y, m, d, 23, 30),
            "currency": "JPY", "impact": "medium",
            "event_name": "National Core CPI y/y",
        })

    # ──── EUR CPI (Flash): monthly ~end of month, 10:00 UTC ────
    eur_cpi_dates = [
        (1, 7), (2, 27), (3, 31), (4, 30), (6, 1), (6, 30),
        (7, 31), (9, 1), (9, 30), (10, 30), (12, 1), (12, 17),
    ]
    for m, d in eur_cpi_dates:
        events.append({
            "datetime": _dt(Y, m, d, 10, 0),
            "currency": "EUR", "impact": "high",
            "event_name": "CPI Flash Estimate y/y",
        })

    # ──── EUR GDP (Flash): quarterly, 10:00 UTC ────
    eur_gdp_dates = [
        (1, 30), (4, 30), (7, 31), (10, 30),
    ]
    for m, d in eur_gdp_dates:
        events.append({
            "datetime": _dt(Y, m, d, 10, 0),
            "currency": "EUR", "impact": "medium",
            "event_name": "GDP Flash Estimate q/q",
        })

    # ──── EUR/DE PMI (Flash): monthly ~3rd week, 08:30-09:00 UTC ────
    eur_pmi_dates = [
        (1, 23), (2, 20), (3, 23), (4, 23), (5, 21), (6, 22),
        (7, 23), (8, 21), (9, 22), (10, 22), (11, 20), (12, 16),
    ]
    for m, d in eur_pmi_dates:
        events.append({
            "datetime": _dt(Y, m, d, 8, 30),
            "currency": "EUR", "impact": "medium",
            "event_name": "Manufacturing PMI (Flash)",
        })
        events.append({
            "datetime": _dt(Y, m, d, 9, 0),
            "currency": "EUR", "impact": "medium",
            "event_name": "Services PMI (Flash)",
        })

    # ──── US PCE Price Index (Fed preferred inflation), 13:30 UTC ────
    pce_dates = [
        (1, 30), (2, 27), (3, 27), (4, 30), (5, 29), (6, 26),
        (7, 31), (8, 28), (9, 25), (10, 30), (11, 25), (12, 23),
    ]
    for m, d in pce_dates:
        events.append({
            "datetime": _dt(Y, m, d, 13, 30),
            "currency": "USD", "impact": "high",
            "event_name": "Core PCE Price Index m/m",
        })

    # Sort by datetime
    events.sort(key=lambda e: e["datetime"])
    return events


def fetch_calendar(start_date: datetime, end_date: datetime) -> list[NewsEvent]:
    """
    Load comprehensive 2026 calendar (hardcoded with exact dates).
    Uses JSON cache if available, otherwise generates and saves.
    """
    cached = _load_cached_calendar(start_date, end_date)
    if cached:
        log.info("Loaded %d calendar events from cache", len(cached))
        return cached

    all_events = generate_2026_calendar()
    _save_calendar_cache(all_events)

    parsed = _parse_events(all_events)
    filtered = [
        e for e in parsed
        if start_date - timedelta(days=1) <= e.dt <= end_date + timedelta(days=1)
    ]
    log.info(
        "Generated 2026 calendar: %d total events, %d in backtest period",
        len(parsed), len(filtered),
    )
    return filtered


def _load_cached_calendar(
    start_date: datetime, end_date: datetime
) -> list[NewsEvent]:
    """Load cached calendar if it covers the backtest period."""
    if not CALENDAR_PATH.exists():
        return []
    try:
        data = json.loads(CALENDAR_PATH.read_text())
        events = _parse_events(data)
        if not events:
            return []
        earliest = min(e.dt for e in events)
        latest = max(e.dt for e in events)
        if earliest <= start_date and latest >= end_date - timedelta(days=7):
            return [
                e for e in events
                if start_date - timedelta(days=1)
                <= e.dt <= end_date + timedelta(days=1)
            ]
        return []
    except Exception:
        return []


def _save_calendar_cache(events: list[dict]):
    """Save calendar events to JSON cache."""
    CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        CALENDAR_PATH.write_text(json.dumps(events, indent=2))
        log.info("Saved %d events to %s", len(events), CALENDAR_PATH)
    except Exception:
        pass


def _parse_events(raw: list[dict]) -> list[NewsEvent]:
    """Parse raw event dicts into NewsEvent objects."""
    events = []
    for e in raw:
        try:
            dt = datetime.fromisoformat(e["datetime"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            events.append(NewsEvent(
                dt=dt,
                currency=e["currency"],
                impact=e["impact"],
                event_name=e["event_name"],
            ))
        except Exception:
            continue
    return events


def is_news_blackout(
    dt_utc: datetime, symbol: str, news_events: list[NewsEvent]
) -> bool:
    """Check if current time is within news blackout window for symbol."""
    currencies = SYMBOL_CURRENCIES.get(symbol, [])
    for event in news_events:
        if event.currency not in currencies:
            continue
        if event.impact == "high":
            before = timedelta(minutes=30)
            after = timedelta(minutes=30)
        elif event.impact == "medium":
            before = timedelta(minutes=15)
            after = timedelta(minutes=15)
        else:
            continue
        if (event.dt - before) <= dt_utc <= (event.dt + after):
            return True
    return False


def get_news_reaction_window(
    dt_utc: datetime, symbol: str, news_events: list[NewsEvent],
) -> NewsEvent | None:
    """Check if we're in the 30-90 min AFTER a high-impact news event.
    This is the reaction window where we look for trend continuation."""
    currencies = SYMBOL_CURRENCIES.get(symbol, [])
    for event in news_events:
        if event.impact != "high":
            continue
        if event.currency not in currencies:
            continue
        mins_after = (dt_utc - event.dt).total_seconds() / 60
        if 30 <= mins_after <= 90:
            return event
    return None


def scan_news_reaction(df, i, symbol, atr):
    """After high-impact news settles (30 min), check if price moved > 1 ATR.
    If so, enter in the direction of the move (momentum continuation)."""
    if i < 6 or atr <= 0:
        return None

    row = df.iloc[i]
    close = row["close"]
    # Compare current price vs price 6 candles ago (~30 min on 5m)
    pre_news_close = df.iloc[i - 6]["close"]
    move = close - pre_news_close
    move_atr = abs(move) / atr

    if move_atr < 1.0:
        return None  # Not enough momentum

    pip_size = PIP_SIZE.get(symbol, 0.0001)

    if move > 0:
        # Bullish reaction — enter long
        sl = close - TRENDING_SL_ATR * atr
        risk = abs(close - sl)
        tp = close + 2.0 * risk
        return {"side": "long", "entry": close, "sl": sl, "tp": tp,
                "reason": f"NEWS LONG | move {move_atr:.1f}x ATR"}
    else:
        # Bearish reaction — enter short
        sl = close + TRENDING_SL_ATR * atr
        risk = abs(close - sl)
        tp = close - 2.0 * risk
        return {"side": "short", "entry": close, "sl": sl, "tp": tp,
                "reason": f"NEWS SHORT | move {move_atr:.1f}x ATR"}


# ═══ Strategy (self-contained) ═══

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


def get_rsi_thresholds(cfg, bias):
    """RSI Hayden zones — bias-adjusted like crypto."""
    if bias == "bullish":
        return cfg.rsi_bull_oversold, cfg.rsi_bull_overbought
    if bias == "bearish":
        return cfg.rsi_bear_oversold, cfg.rsi_bear_overbought
    return cfg.rsi_oversold, cfg.rsi_overbought


def scan_ranging(row, prev_row, bias, cfg):
    """BB mean reversion — matching crypto filters:
    Mandatory: BB touch, close inside 25%, anti-breakout, BB mid room
    Optional (3/4 with-trend, 4/4 counter-trend): wick, candle dir, RSI, volume
    """
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

    # Prev candle values for anti-breakout
    p_close = prev_row["close"]
    p_bbl = prev_row.get("bb_lower") or 0
    p_bbu = prev_row.get("bb_upper") or 0

    oversold, overbought = get_rsi_thresholds(cfg, bias)

    # ── LONG ──
    touch_lower = low <= bbl * (1 + cfg.bb_touch_pct / 100)
    close_inside_long = close > bbl
    prev_above_bb = p_close > p_bbl if p_bbl > 0 else True
    mid_room_long = bbm > close

    if touch_lower and close_inside_long and prev_above_bb and mid_room_long:
        lower_wick = min(close, opn) - low
        wick_pct = lower_wick / candle_range

        opt_wick = wick_pct >= 0.12
        opt_bullish = close > opn
        opt_rsi = rsi <= oversold
        opt_vol = vol > vsma * cfg.volume_range_mult if vsma > 0 else True
        opt_count = sum([opt_wick, opt_bullish, opt_rsi, opt_vol])
        min_opt = 3  # No counter-trend penalty — session filter is enough

        if opt_count >= min_opt:
            sl = close - RANGING_SL_ATR * atr
            tp = close + (bbm - close) * 0.50
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= RANGING_MIN_RR:
                return {"side": "long", "entry": close, "sl": sl, "tp": tp}

    # ── SHORT ──
    touch_upper = high >= bbu * (1 - cfg.bb_touch_pct / 100)
    close_inside_short = close < bbu
    prev_below_bb = p_close < p_bbu if p_bbu > 0 else True
    mid_room_short = bbm < close

    if touch_upper and close_inside_short and prev_below_bb and mid_room_short:
        upper_wick = high - max(close, opn)
        wick_pct = upper_wick / candle_range

        opt_wick = wick_pct >= 0.12
        opt_bearish = close < opn
        opt_rsi = rsi >= overbought
        opt_vol = vol > vsma * cfg.volume_range_mult if vsma > 0 else True
        opt_count = sum([opt_wick, opt_bearish, opt_rsi, opt_vol])
        min_opt = 3  # No counter-trend penalty — session filter is enough

        if opt_count >= min_opt:
            sl = close + RANGING_SL_ATR * atr
            tp = close - (close - bbm) * 0.50
            risk = abs(close - sl)
            reward = abs(tp - close)
            if risk > 0 and reward / risk >= RANGING_MIN_RR:
                return {"side": "short", "entry": close, "sl": sl, "tp": tp}

    return None


def scan_trending(row, prev_row, df_slice, bias):
    """Trending breakout/pullback — used in London/NY sessions."""
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

    # TP = 2x risk, SL = 1.5 ATR from EMA
    sl_dist = TRENDING_SL_ATR * atr

    if (plus_di > minus_di and close > recent_high and close > opn
            and ema_f > ema_m and 50 < rsi < 80
            and bias in ("bullish", "neutral")):
        sl = ema_f - sl_dist if ema_f > 0 else close - sl_dist
        sl = min(sl, close - 0.5 * atr)  # ensure SL below entry
        risk = abs(close - sl)
        tp = close + 2.0 * risk
        if risk > 0 and (tp - close) / risk >= TRENDING_MIN_RR:
            return {"side": "long", "entry": close, "sl": sl, "tp": tp}

    if (minus_di > plus_di and close < recent_low and close < opn
            and ema_f < ema_m and 20 < rsi < 50
            and bias in ("bearish", "neutral")):
        sl = ema_f + sl_dist if ema_f > 0 else close + sl_dist
        sl = max(sl, close + 0.5 * atr)
        risk = abs(close - sl)
        tp = close - 2.0 * risk
        if risk > 0 and (close - tp) / risk >= TRENDING_MIN_RR:
            return {"side": "short", "entry": close, "sl": sl, "tp": tp}

    return None


def check_exit(trade, row, candles_held):
    """Check SL/TP exit conditions."""
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

    # Time exit for trending (48 candles max)
    if trade.regime == "trending" and candles_held >= 48:
        return row["close"], "TIME"

    return None


def check_partial_close(trade, row) -> float:
    """
    Check if price reached 50% of TP distance for partial close.
    Returns partial exit price or 0.
    """
    if trade.partial_closed:
        return 0.0

    tp_dist = abs(trade.tp_price - trade.entry_price)
    half_tp = tp_dist * TRAILING_BE_PCT

    if trade.side == "long":
        partial_level = trade.entry_price + half_tp
        if row["high"] >= partial_level:
            return partial_level
    else:
        partial_level = trade.entry_price - half_tp
        if row["low"] <= partial_level:
            return partial_level

    return 0.0


def update_trailing_sl(trade, row):
    """
    Trailing SL logic:
    - Before partial: use chandelier exit
    - After partial: chandelier continues on remaining
    - Move to breakeven at 50% of TP distance
    """
    # Breakeven logic — move SL to entry once price reaches 50% of TP
    tp_dist = abs(trade.tp_price - trade.entry_price)
    half_tp = trade.entry_price + tp_dist * TRAILING_BE_PCT if trade.side == "long" \
        else trade.entry_price - tp_dist * TRAILING_BE_PCT

    if trade.side == "long":
        if row["high"] >= half_tp and trade.sl_price < trade.entry_price:
            trade.sl_price = trade.entry_price
    else:
        if row["low"] <= half_tp and trade.sl_price > trade.entry_price:
            trade.sl_price = trade.entry_price

    # Chandelier trailing
    chand_long = row.get("chandelier_long")
    chand_short = row.get("chandelier_short")
    if trade.side == "long" and chand_long:
        if chand_long > trade.entry_price and chand_long > trade.sl_price:
            trade.sl_price = chand_long
    elif trade.side == "short" and chand_short:
        if chand_short < trade.entry_price and chand_short < trade.sl_price:
            trade.sl_price = chand_short


# ═══ FX Position Sizing ═══

def calc_lot_size(entry, sl, risk_pct, symbol):
    """Risk-based lot calculation. Fixed sizing on BALANCE_START."""
    pip_size = PIP_SIZE.get(symbol, 0.0001)
    pip_val = PIP_VALUE.get(symbol, 10.0)
    sl_pips = abs(entry - sl) / pip_size
    if sl_pips <= 0:
        return 0.0, 0.0
    risk_amount = BALANCE_START * risk_pct
    lots = risk_amount / (sl_pips * pip_val)
    max_lots = (BALANCE_START * LEVERAGE) / 100_000
    lots = min(lots, max_lots)
    lots = round(lots, 2)
    if lots < 0.01:
        return 0.0, 0.0
    notional = lots * 100_000
    return lots, notional


# ═══ Main Backtest ═══

def run_backtest():
    log.info("=" * 60)
    log.info("FUTU FX Backtest — %d Days — Session-Based Strategy", DAYS)
    log.info("Balance: $%.0f | Leverage: 1:%d | Spread-based fees",
             BALANCE_START, LEVERAGE)
    log.info("Asian ranging: %s", ", ".join(ASIAN_SYMBOLS))
    log.info("London/NY trending: %s", ", ".join(TRENDING_SYMBOLS))
    log.info("Risk: Ranging %.0f%% | Trending %.0f%% | Max daily %.0f%%",
             RISK_RANGING * 100, RISK_TRENDING * 100, MAX_DAILY_LOSS * 100)
    log.info("=" * 60)

    if not mt5.initialize():
        log.error("MT5 init failed: %s", mt5.last_error())
        return

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=DAYS + 10)
    backtest_start = now - timedelta(days=DAYS)

    # Fetch news calendar
    log.info("Fetching news calendar...")
    news_events = fetch_calendar(backtest_start, now)
    log.info("News events loaded: %d (high: %d, medium: %d)",
             len(news_events),
             sum(1 for e in news_events if e.impact == "high"),
             sum(1 for e in news_events if e.impact == "medium"))
    news_blocked_count = 0

    state = BacktestState()

    for symbol in ALL_SYMBOLS:
        short = symbol.replace("m", "")
        log.info("\n--- %s ---", short)

        is_ranging_sym = symbol in ASIAN_SYMBOLS
        is_trending_sym = symbol in TRENDING_SYMBOLS

        # Fetch data
        candles_5m = fetch_mt5(symbol, "5m", since)
        candles_4h = fetch_mt5(symbol, "4h", since)
        candles_1h = fetch_mt5(symbol, "1h", since) if is_trending_sym else []

        if len(candles_5m) < 200 or len(candles_4h) < 20:
            log.info("SKIP %s — not enough data (5m=%d, 4h=%d)",
                     short, len(candles_5m), len(candles_4h))
            continue

        df_5m = compute_all(candles_5m, IND_CFG)
        df_4h = compute_all(candles_4h, IND_CFG)
        df_1h = None
        if is_trending_sym and len(candles_1h) >= 50:
            df_1h = compute_all(candles_1h, IND_CFG)

        # HTF bias lookup
        htf_biases = {}
        for i in range(IND_CFG.ema_slow + 5, len(df_4h)):
            chunk = df_4h.iloc[:i + 1]
            bias = detect_htf_bias(chunk)
            htf_biases[df_4h.index[i]] = bias

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
            nonlocal news_blocked_count
            t.exit_price = exit_price
            t.exit_time = str(bar_time)
            t.exit_reason = reason
            pip_size = PIP_SIZE.get(symbol, 0.0001)
            pip_val = PIP_VALUE.get(symbol, 10.0)
            if t.side == "long":
                pips = (t.exit_price - t.entry_price) / pip_size
            else:
                pips = (t.entry_price - t.exit_price) / pip_size
            # PnL on remaining lots
            t.pnl_gross = pips * pip_val * t.lots_remaining + t.partial_pnl
            spread = SPREAD_PIPS.get(symbol, 1.5)
            t.spread_cost = spread * pip_val * t.lots  # spread on original lots
            t.pnl_net = t.pnl_gross - t.spread_cost
            state.balance += t.pnl_net
            session = t.session
            if t.pnl_net >= 0:
                state.wins += 1
                state.session_stats[session]["wins"] += 1
            else:
                state.losses += 1
                state.daily_loss += abs(t.pnl_net)
            state.session_stats[session]["trades"] += 1
            state.session_stats[session]["pnl"] += t.pnl_net
            state.trades.append(t)
            if state.balance > state.peak_balance:
                state.peak_balance = state.balance
            dd = (state.peak_balance - state.balance) / state.peak_balance
            if dd > state.max_drawdown:
                state.max_drawdown = dd

        def do_partial_close(t, partial_price, row):
            """Close 50% at partial TP, move SL to breakeven."""
            pip_size = PIP_SIZE.get(symbol, 0.0001)
            pip_val = PIP_VALUE.get(symbol, 10.0)
            close_lots = t.lots * PARTIAL_CLOSE_PCT
            if t.side == "long":
                pips = (partial_price - t.entry_price) / pip_size
            else:
                pips = (t.entry_price - partial_price) / pip_size
            t.partial_pnl = pips * pip_val * close_lots
            t.lots_remaining = t.lots - close_lots
            t.partial_closed = True
            # Move SL to breakeven
            t.sl_price = t.entry_price

        def count_open_positions():
            return len(state.open_positions)

        def has_open_for_symbol(sym):
            return any(t.symbol == sym for t in state.open_positions)

        # ═══ Ranging: 5m direct entry (Asian session only) ═══
        if is_ranging_sym:
            for i in range(IND_CFG.ema_slow + 5, len(df_5m)):
                bar_time = df_5m.index[i]
                row = df_5m.iloc[i]
                dt_utc = bar_to_utc(bar_time, row)

                if dt_utc.weekday() >= 5:
                    continue
                if dt_utc < backtest_start.replace(tzinfo=timezone.utc):
                    continue

                session = get_session(dt_utc)

                # Always process exits regardless of session
                closed = []
                for t in state.open_positions:
                    if t.symbol != symbol or t.regime not in ("ranging", "news"):
                        continue
                    t.candles_held += 1
                    update_trailing_sl(t, row)
                    partial_price = check_partial_close(t, row)
                    if partial_price > 0:
                        do_partial_close(t, partial_price, row)
                    result = check_exit(t, row, t.candles_held)
                    if result:
                        close_trade(t, result[0], result[1], bar_time)
                        closed.append(t)
                for t in closed:
                    state.open_positions.remove(t)

                # Only enter during Asian session
                if session != SESSION_ASIAN:
                    continue

                day_str = str(bar_time)[:10]
                if state.daily_date != day_str:
                    state.daily_loss = 0.0
                    state.daily_date = day_str

                if state.daily_loss >= BALANCE_START * MAX_DAILY_LOSS:
                    continue
                if has_open_for_symbol(symbol):
                    continue

                # News blackout
                if is_news_blackout(dt_utc, symbol, news_events):
                    news_blocked_count += 1
                    continue

                # Direct 5m scan — no 15m flag needed
                if i < 1:
                    continue
                prev_row = df_5m.iloc[i - 1]
                bias = get_bias(bar_time)
                rsig = scan_ranging(row, prev_row, bias, STRAT_CFG)
                if rsig is None:
                    continue

                entry = rsig["entry"]
                sl = rsig["sl"]
                tp = rsig["tp"]
                lots, notional = calc_lot_size(entry, sl, RISK_RANGING, symbol)
                if lots > 0:
                    trade = Trade(
                        symbol=symbol, side=rsig["side"],
                        regime="ranging", session=SESSION_ASIAN,
                        entry_price=entry, sl_price=sl, tp_price=tp,
                        lots=lots, lots_remaining=lots,
                        notional=notional, entry_time=str(bar_time),
                    )
                    state.open_positions.append(trade)
                    sym_ranging += 1

        # ═══ News Reaction on 5m (all symbols, after high-impact news) ═══
        sym_news = 0
        for i in range(IND_CFG.ema_slow + 5, len(df_5m)):
            bar_time = df_5m.index[i]
            row = df_5m.iloc[i]
            dt_utc = bar_to_utc(bar_time, row)

            if dt_utc.weekday() >= 5:
                continue
            if dt_utc < backtest_start.replace(tzinfo=timezone.utc):
                continue

            # Only check during news reaction window
            news_event = get_news_reaction_window(dt_utc, symbol, news_events)
            if news_event is None:
                continue

            day_str = str(bar_time)[:10]
            if state.daily_date != day_str:
                state.daily_loss = 0.0
                state.daily_date = day_str

            if state.daily_loss >= BALANCE_START * MAX_DAILY_LOSS:
                continue
            if count_open_positions() >= MAX_CONCURRENT:
                continue
            if has_open_for_symbol(symbol):
                continue

            atr = row.get("atr") or 0
            nsig = scan_news_reaction(df_5m, i, symbol, atr)
            if nsig is None:
                continue

            lots, notional = calc_lot_size(
                nsig["entry"], nsig["sl"], RISK_TRENDING, symbol)
            if lots > 0:
                trade = Trade(
                    symbol=symbol, side=nsig["side"],
                    regime="news", session="news",
                    entry_price=nsig["entry"], sl_price=nsig["sl"],
                    tp_price=nsig["tp"], lots=lots,
                    lots_remaining=lots, notional=notional,
                    entry_time=str(bar_time),
                )
                state.open_positions.append(trade)
                sym_news += 1

        # ═══ Trending on 1H (London/NY sessions only) ═══
        if df_1h is not None and is_trending_sym:
            for i in range(IND_CFG.ema_slow + 5, len(df_1h)):
                bar_time = df_1h.index[i]
                row = df_1h.iloc[i]
                prev_row = df_1h.iloc[i - 1]
                dt_utc = bar_to_utc(bar_time, row)

                if dt_utc.weekday() >= 5:
                    continue
                if dt_utc < backtest_start.replace(tzinfo=timezone.utc):
                    continue

                session = get_session(dt_utc)

                day_str = str(bar_time)[:10]
                if state.daily_date != day_str:
                    state.daily_loss = 0.0
                    state.daily_date = day_str

                # Process exits (always, regardless of session)
                closed = []
                for t in state.open_positions:
                    if t.symbol != symbol or t.regime != "trending":
                        continue
                    t.candles_held += 1
                    update_trailing_sl(t, row)
                    partial_price = check_partial_close(t, row)
                    if partial_price > 0:
                        do_partial_close(t, partial_price, row)
                    result = check_exit(t, row, t.candles_held)
                    if result:
                        close_trade(t, result[0], result[1], bar_time)
                        closed.append(t)
                for t in closed:
                    state.open_positions.remove(t)

                # Only enter in London/NY
                if session not in (SESSION_LONDON, SESSION_NY):
                    continue

                if state.daily_loss >= BALANCE_START * MAX_DAILY_LOSS:
                    continue
                if count_open_positions() >= MAX_CONCURRENT:
                    continue
                if has_open_for_symbol(symbol):
                    continue

                # News blackout
                if is_news_blackout(dt_utc, symbol, news_events):
                    news_blocked_count += 1
                    continue

                bias = get_bias(bar_time)
                df_slice = df_1h.iloc[max(0, i - T_LOOKBACK - 5):i]
                tsig = scan_trending(row, prev_row, df_slice, bias)
                if tsig:
                    entry_session = session
                    lots, notional = calc_lot_size(
                        tsig["entry"], tsig["sl"], RISK_TRENDING, symbol,
                    )
                    if lots > 0:
                        trade = Trade(
                            symbol=symbol, side=tsig["side"],
                            regime="trending", session=entry_session,
                            entry_price=tsig["entry"], sl_price=tsig["sl"],
                            tp_price=tsig["tp"], lots=lots,
                            lots_remaining=lots, notional=notional,
                            entry_time=str(bar_time),
                        )
                        state.open_positions.append(trade)
                        sym_trending += 1

        log.info("%s: %d ranging + %d trending + %d news", short, sym_ranging, sym_trending, sym_news)

    mt5.shutdown()
    log.info("\nNews-blocked entries: %d", news_blocked_count)
    print_results(state)


def print_results(state):
    trades = state.trades
    total = len(trades)

    log.info("\n" + "=" * 60)
    log.info("FX BACKTEST RESULTS — %d DAYS (Session-Based)", DAYS)
    log.info("=" * 60)

    if total == 0:
        log.info("No trades executed.")
        return

    total_pnl_net = sum(t.pnl_net for t in trades)
    total_pnl_gross = sum(t.pnl_gross for t in trades)
    total_spread = sum(t.spread_cost for t in trades)
    win_rate = state.wins / total * 100

    winners = [t for t in trades if t.pnl_net >= 0]
    losers = [t for t in trades if t.pnl_net < 0]
    avg_win = sum(t.pnl_net for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t.pnl_net for t in losers) / len(losers) if losers else 0
    avg_rr = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    profit_factor = (
        sum(t.pnl_net for t in winners) / abs(sum(t.pnl_net for t in losers))
    ) if losers else float('inf')
    avg_hold = sum(t.candles_held for t in trades) / total

    ranging_trades = [t for t in trades if t.regime == "ranging"]
    trending_trades = [t for t in trades if t.regime == "trending"]
    news_trades = [t for t in trades if t.regime == "news"]

    log.info("")
    log.info("OVERVIEW")
    log.info("  Starting Balance:  $%.2f", BALANCE_START)
    log.info("  Final Balance:     $%.2f", state.balance)
    log.info("  Net PnL:           $%+.2f (%.1f%%)",
             total_pnl_net, total_pnl_net / BALANCE_START * 100)
    log.info("  Gross PnL:         $%.2f", total_pnl_gross)
    log.info("  Spread Cost:       -$%.2f", total_spread)
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
    log.info("  Best Trade:        $%.2f",
             max((t.pnl_net for t in trades), default=0))
    log.info("  Worst Trade:       $%.2f",
             min((t.pnl_net for t in trades), default=0))
    log.info("  Avg Hold:          %.1f candles", avg_hold)

    log.info("")
    log.info("BY REGIME")
    if ranging_trades:
        rw = sum(1 for t in ranging_trades if t.pnl_net >= 0)
        rp = sum(t.pnl_net for t in ranging_trades)
        log.info("  Ranging:   %d trades | %d wins (%.0f%%) | PnL $%+.2f",
                 len(ranging_trades), rw,
                 rw / len(ranging_trades) * 100, rp)
    if trending_trades:
        tw = sum(1 for t in trending_trades if t.pnl_net >= 0)
        tp_pnl = sum(t.pnl_net for t in trending_trades)
        log.info("  Trending:  %d trades | %d wins (%.0f%%) | PnL $%+.2f",
                 len(trending_trades), tw,
                 tw / len(trending_trades) * 100, tp_pnl)
    if news_trades:
        nw = sum(1 for t in news_trades if t.pnl_net >= 0)
        np_pnl = sum(t.pnl_net for t in news_trades)
        log.info("  News:      %d trades | %d wins (%.0f%%) | PnL $%+.2f",
                 len(news_trades), nw,
                 nw / len(news_trades) * 100, np_pnl)

    # Session breakdown
    log.info("")
    log.info("BY SESSION")
    for sess_name, stats in state.session_stats.items():
        if stats["trades"] > 0:
            wr = stats["wins"] / stats["trades"] * 100
            log.info("  %-8s %3d trades | %d wins (%.0f%%) | PnL $%+.2f",
                     sess_name.capitalize(), stats["trades"],
                     stats["wins"], wr, stats["pnl"])
        else:
            log.info("  %-8s   0 trades", sess_name.capitalize())

    # By symbol
    sym_pnl = {}
    for t in trades:
        s = t.symbol.replace("m", "")
        sym_pnl[s] = sym_pnl.get(s, 0) + t.pnl_net
    log.info("")
    log.info("BY SYMBOL")
    for s, pnl in sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True):
        cnt = sum(1 for t in trades if t.symbol.replace("m", "") == s)
        log.info("  %-8s %3d trades | PnL $%+.2f", s, cnt, pnl)

    # Daily PnL
    log.info("")
    log.info("DAILY PNL")
    daily = {}
    for t in trades:
        day = t.exit_time[:10]
        daily[day] = daily.get(day, 0) + t.pnl_net
    for day in sorted(daily.keys()):
        n = min(int(abs(daily[day]) / 0.5), 30)
        bar = ("+" if daily[day] >= 0 else "-") * n
        log.info("  %s  $%+7.2f  %s", day, daily[day], bar[:30])

    # Partial close stats
    partial_trades = [t for t in trades if t.partial_closed]
    if partial_trades:
        log.info("")
        log.info("PARTIAL CLOSES")
        log.info("  Trades with partial: %d / %d",
                 len(partial_trades), total)
        partial_pnl = sum(t.partial_pnl for t in partial_trades)
        log.info("  Partial PnL locked:  $%+.2f", partial_pnl)

    # Top/worst trades
    log.info("")
    log.info("TOP 5 TRADES")
    for t in sorted(trades, key=lambda x: x.pnl_net, reverse=True)[:5]:
        log.info("  %s %-5s %-8s [%s] entry=%.5f exit=%.5f pnl=$%+.2f (%s)",
                 t.symbol.replace("m", ""), t.side.upper(), t.regime,
                 t.session, t.entry_price, t.exit_price,
                 t.pnl_net, t.exit_reason)

    log.info("")
    log.info("WORST 5 TRADES")
    for t in sorted(trades, key=lambda x: x.pnl_net)[:5]:
        log.info("  %s %-5s %-8s [%s] entry=%.5f exit=%.5f pnl=$%+.2f (%s)",
                 t.symbol.replace("m", ""), t.side.upper(), t.regime,
                 t.session, t.entry_price, t.exit_price,
                 t.pnl_net, t.exit_reason)


if __name__ == "__main__":
    run_backtest()
