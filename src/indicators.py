import pandas as pd
import numpy as np
from src.config import IndicatorConfig


def build_dataframe(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> str:
    name = f"ema_{period}"
    df[name] = df[col].ewm(span=period, adjust=False).mean()
    return name


def add_rsi(df: pd.DataFrame, period: int = 10) -> str:
    name = "rsi"
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[name] = 100 - (100 / (1 + rs))
    return name


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> tuple[str, str, str]:
    mid = f"bb_mid"
    upper = f"bb_upper"
    lower = f"bb_lower"
    df[mid] = df["close"].rolling(period).mean()
    rolling_std = df["close"].rolling(period).std()
    df[upper] = df[mid] + std * rolling_std
    df[lower] = df[mid] - std * rolling_std
    df["bb_width"] = (df[upper] - df[lower]) / df[mid]
    df["bb_width_sma"] = df["bb_width"].rolling(period).mean()
    return mid, upper, lower


def add_adx(df: pd.DataFrame, period: int = 14) -> str:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_smooth)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, min_periods=period).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return "adx"


def add_atr(df: pd.DataFrame, period: int = 14) -> str:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return "atr"


def add_vwap(df: pd.DataFrame) -> str:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return "vwap"


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> str:
    df["volume_sma"] = df["volume"].rolling(period).mean()
    return "volume_sma"


def add_chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> tuple[str, str]:
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period).mean()

    df["chandelier_long"] = highest - mult * atr_val
    df["chandelier_short"] = lowest + mult * atr_val
    return "chandelier_long", "chandelier_short"


def calc_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Calculate Fibonacci retracement levels from swing high/low."""
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


def compute_all(candles: list[dict], cfg: IndicatorConfig) -> pd.DataFrame:
    df = build_dataframe(candles)

    add_ema(df, cfg.ema_fast)
    add_ema(df, cfg.ema_mid)
    add_ema(df, cfg.ema_slow)
    add_rsi(df, cfg.rsi_period)
    add_bollinger_bands(df, cfg.bb_period, cfg.bb_std)
    add_adx(df, cfg.adx_period)
    add_atr(df, cfg.atr_period)
    add_vwap(df)
    add_volume_sma(df, cfg.volume_sma_period)
    add_chandelier_exit(df, cfg.chandelier_period, cfg.chandelier_mult)

    return df
