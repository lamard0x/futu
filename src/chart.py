import io
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.indicators import compute_all
from src.config import IndicatorConfig

logger = logging.getLogger("futu.chart")

COLORS = {
    "bg": "#1a1a2e",
    "grid": "#2a2a3e",
    "text": "#e0e0e0",
    "bull": "#00d474",
    "bear": "#ff3b69",
    "ema9": "#ffeb3b",
    "ema21": "#ff9800",
    "ema50": "#2196f3",
    "bb_upper": "#9c27b0",
    "bb_lower": "#9c27b0",
    "bb_fill": "#9c27b055",
    "volume": "#3a3a5e",
    "vol_bull": "#00d47444",
    "vol_bear": "#ff3b6944",
    "entry_long": "#00ff88",
    "entry_short": "#ff4466",
    "sl": "#ff0000",
    "tp": "#00ff00",
    "chandelier": "#ff6600",
}


def generate_chart(
    candles: list[dict],
    indicator_cfg: IndicatorConfig,
    symbol: str = "BTC/USDT",
    entries: list[dict] | None = None,
    regime: str = "",
    bias: str = "",
) -> bytes:
    df = compute_all(candles, indicator_cfg)
    df = df.tail(80)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [5, 1.2, 1]},
        facecolor=COLORS["bg"],
    )
    fig.subplots_adjust(hspace=0.08)

    ax_price, ax_vol, ax_rsi = axes
    for ax in axes:
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.grid(True, color=COLORS["grid"], alpha=0.3, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])

    x = range(len(df))
    dates = df.index

    # ── Candlesticks ──
    for i, (idx, row) in enumerate(df.iterrows()):
        color = COLORS["bull"] if row["close"] >= row["open"] else COLORS["bear"]
        ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        ax_price.bar(i, body_height, bottom=body_bottom, width=0.6,
                     color=color, edgecolor=color, linewidth=0.5)

    # ── EMAs ──
    ema_fast_col = f"ema_{indicator_cfg.ema_fast}"
    ema_mid_col = f"ema_{indicator_cfg.ema_mid}"
    ema_slow_col = f"ema_{indicator_cfg.ema_slow}"
    if ema_fast_col in df.columns:
        ax_price.plot(x, df[ema_fast_col], color=COLORS["ema9"], linewidth=1, label=f"EMA {indicator_cfg.ema_fast}", alpha=0.9)
    if ema_mid_col in df.columns:
        ax_price.plot(x, df[ema_mid_col], color=COLORS["ema21"], linewidth=1, label=f"EMA {indicator_cfg.ema_mid}", alpha=0.9)
    if ema_slow_col in df.columns:
        ax_price.plot(x, df[ema_slow_col], color=COLORS["ema50"], linewidth=1.2, label=f"EMA {indicator_cfg.ema_slow}", alpha=0.9)

    # ── Bollinger Bands ──
    if "bb_upper" in df.columns:
        ax_price.plot(x, df["bb_upper"], color=COLORS["bb_upper"], linewidth=0.8, linestyle="--", alpha=0.7)
        ax_price.plot(x, df["bb_lower"], color=COLORS["bb_lower"], linewidth=0.8, linestyle="--", alpha=0.7)
        ax_price.fill_between(x, df["bb_upper"], df["bb_lower"], color=COLORS["bb_fill"], alpha=0.15)

    # ── Chandelier Exit ──
    if "chandelier_long" in df.columns:
        ax_price.plot(x, df["chandelier_long"], color=COLORS["chandelier"], linewidth=0.8, linestyle=":", alpha=0.7, label="Chandelier")

    # ── Entry/Exit markers ──
    if entries:
        for entry in entries:
            ts = entry.get("timestamp")
            idx = len(df) - 1  # default to last candle
            if ts is not None:
                mask = df.index == ts
                if mask.any():
                    idx = mask.values.nonzero()[0][0]

            if entry.get("type") == "entry":
                color = COLORS["entry_long"] if entry.get("side") == "buy" else COLORS["entry_short"]
                marker = "^" if entry.get("side") == "buy" else "v"
                ax_price.scatter(idx, entry["price"], color=color, marker=marker, s=120, zorder=5, edgecolors="white", linewidth=0.8)
                ax_price.annotate(
                    f"{'LONG' if entry.get('side')=='buy' else 'SHORT'}\n${entry['price']:,.0f}",
                    (idx, entry["price"]), fontsize=7, color=color,
                    ha="center", va="bottom" if entry.get("side") == "buy" else "top",
                    fontweight="bold",
                )

            if entry.get("sl"):
                ax_price.axhline(y=entry["sl"], color=COLORS["sl"], linewidth=0.8, linestyle="--", alpha=0.6)
                ax_price.annotate(f"SL ${entry['sl']:,.0f}", (len(df)-1, entry["sl"]),
                                  fontsize=7, color=COLORS["sl"], ha="right", va="bottom")

            if entry.get("tp"):
                ax_price.axhline(y=entry["tp"], color=COLORS["tp"], linewidth=0.8, linestyle="--", alpha=0.6)
                ax_price.annotate(f"TP ${entry['tp']:,.0f}", (len(df)-1, entry["tp"]),
                                  fontsize=7, color=COLORS["tp"], ha="right", va="bottom")

    # ── Volume ──
    for i, (idx, row) in enumerate(df.iterrows()):
        color = COLORS["vol_bull"] if row["close"] >= row["open"] else COLORS["vol_bear"]
        ax_vol.bar(i, row["volume"], width=0.6, color=color)
    if "volume_sma" in df.columns:
        ax_vol.plot(x, df["volume_sma"], color=COLORS["ema21"], linewidth=0.8, alpha=0.7)
    ax_vol.set_ylabel("Vol", color=COLORS["text"], fontsize=8)

    # ── RSI ──
    if "rsi" in df.columns:
        ax_rsi.plot(x, df["rsi"], color=COLORS["ema9"], linewidth=1)
        ax_rsi.axhline(y=70, color=COLORS["bear"], linewidth=0.5, linestyle="--", alpha=0.5)
        ax_rsi.axhline(y=30, color=COLORS["bull"], linewidth=0.5, linestyle="--", alpha=0.5)
        ax_rsi.axhline(y=50, color=COLORS["text"], linewidth=0.3, linestyle=":", alpha=0.3)
        ax_rsi.fill_between(x, 30, df["rsi"].clip(upper=30), color=COLORS["bull"], alpha=0.15)
        ax_rsi.fill_between(x, 70, df["rsi"].clip(lower=70), color=COLORS["bear"], alpha=0.15)
        ax_rsi.set_ylim(10, 90)
        ax_rsi.set_ylabel("RSI", color=COLORS["text"], fontsize=8)

    # ── X axis labels ──
    tick_positions = list(range(0, len(df), max(1, len(df) // 8)))
    tick_labels = [dates[i].strftime("%m/%d %H:%M") if hasattr(dates[i], "strftime")
                   else str(dates[i])[-14:-3] for i in tick_positions]
    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([] if ax != ax_rsi else tick_labels, fontsize=7, color=COLORS["text"], rotation=30)

    # ── Title ──
    last = df.iloc[-1]
    price_str = f"${last['close']:,.1f}"
    regime_str = f" | {regime}" if regime else ""
    bias_str = f" | HTF: {bias}" if bias else ""
    adx_str = f" | ADX: {last.get('adx', 0):.0f}" if "adx" in df.columns else ""
    rsi_str = f" | RSI: {last.get('rsi', 0):.0f}" if "rsi" in df.columns else ""

    ax_price.set_title(
        f"{symbol} 15m — {price_str}{regime_str}{bias_str}{adx_str}{rsi_str}",
        color=COLORS["text"], fontsize=11, fontweight="bold", pad=10,
    )

    ax_price.legend(loc="upper left", fontsize=7, facecolor=COLORS["bg"],
                    edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
