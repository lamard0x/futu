from dataclasses import dataclass, field


@dataclass
class SwingConfig:
    enabled: bool = True
    capital: float = 500.0
    risk_pct: float = 0.02          # 2% = $10 per trade
    max_positions: int = 3           # $500/3 ~ $166 each
    max_drawdown_pct: float = 0.15   # 15% = $75 stop scanning
    min_rr: float = 2.0              # swing needs R:R >= 1:2
    min_optional_score: int = 3      # 3/5 optional conditions
    hold_days_min: int = 2
    hold_days_max: int = 14
    scan_hours_utc: list = field(default_factory=lambda: [0, 4, 8, 12, 16, 20])  # every 4h
    scan_minute_utc: int = 5
    top_symbols: int = 100           # top 100 by volume per exchange
    rate_limit_delay: float = 0.3    # seconds between API calls
    exchanges: list = field(default_factory=lambda: [
        "binance", "okx", "bybit",
    ])
    # Indicator params for swing
    ema_fast: int = 21
    ema_slow: int = 50
    rsi_period: int = 14             # standard RSI for swing
    volume_sma_period: int = 20
    fib_lookback: int = 50           # bars to find swing high/low
    sd_zone_lookback: int = 100      # bars for S/D zone detection
    bos_lookback: int = 20           # H4 bars for BOS detection
