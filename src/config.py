import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OVERRIDE_PATH = Path(__file__).parent.parent / "data" / "config_overrides.json"


@dataclass
class ExchangeConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: str = "true"
    symbol: str = "BTC/USDT:USDT"
    leverage: int = 10
    margin_mode: str = "cross"


@dataclass
class IndicatorConfig:
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    rsi_period: int = 10
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    atr_period: int = 14
    volume_sma_period: int = 20
    chandelier_period: int = 22
    chandelier_mult: float = 3.0


@dataclass
class StrategyConfig:
    # Regime detection
    adx_trending: float = 25.0
    adx_ranging: float = 25.0

    # Trending mode entry (now filtered by H4 bias)
    rsi_trend_bull: float = 50.0
    volume_trend_mult: float = 1.2

    # Ranging mode entry — relaxed (HTF bias does the filtering now)
    rsi_oversold: float = 40.0
    rsi_overbought: float = 60.0
    bb_touch_pct: float = 0.5
    volume_range_mult: float = 0.8

    # Candle strength filter
    min_body_ratio: float = 0.3

    # Alert mode (1-min scan)
    volume_alert_mult: float = 2.5
    bb_alert_std: float = 2.5
    alert_target_pct: float = 0.25
    alert_sl_atr_mult: float = 1.0

    # Main mode (15-min) — TP > SL for positive expectancy
    main_tp1_atr_mult: float = 1.5
    main_tp2_atr_mult: float = 2.5
    main_sl_trending_atr_mult: float = 1.5
    main_sl_ranging_atr_mult: float = 1.0
    partial_close_pct: float = 0.5

    # Ranging time exit
    ranging_max_candles: int = 15


@dataclass
class RiskConfig:
    account_balance: float = 300.0
    risk_per_trade_main: float = 0.02
    risk_per_trade_scalp: float = 0.01
    max_daily_loss_pct: float = 0.06
    max_positions: int = 3
    max_symbols: int = 10
    cooldown_candles: int = 2
    min_rr_trending: float = 1.3
    min_rr_ranging: float = 1.2


@dataclass
class FundingConfig:
    enabled: bool = False
    min_rate: float = 0.0001
    max_position_pct: float = 0.05
    check_interval: int = 3600
    symbols_to_scan: int = 20


@dataclass
class WebhookConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    secret: str = ""


@dataclass
class TimeframeConfig:
    htf: str = "4h"
    main_tf: str = "15m"
    alert_tf: str = "1m"
    candle_limit: int = 200


@dataclass
class Config:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    timeframe: TimeframeConfig = field(default_factory=TimeframeConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)

    def __post_init__(self):
        self._load_env()
        self._load_overrides()

    def _load_env(self):
        self.exchange.api_key = os.getenv("BYBIT_API_KEY", "")
        self.exchange.api_secret = os.getenv("BYBIT_API_SECRET", "")
        self.exchange.testnet = os.getenv("BYBIT_TESTNET", "true").lower()
        self.exchange.symbol = os.getenv("TRADING_SYMBOL", "BTC/USDT:USDT")
        self.exchange.leverage = int(os.getenv("TRADING_LEVERAGE", "10"))
        self.risk.account_balance = float(os.getenv("ACCOUNT_BALANCE", "300"))
        self.webhook.secret = os.getenv("WEBHOOK_SECRET", "")
        self.funding.enabled = os.getenv("FUNDING_ENABLED", "false").lower() == "true"
        self.webhook.enabled = os.getenv("WEBHOOK_ENABLED", "false").lower() == "true"

    def _load_overrides(self):
        if not OVERRIDE_PATH.exists():
            return
        try:
            overrides = json.loads(OVERRIDE_PATH.read_text())
            self._apply_overrides(overrides)
        except (json.JSONDecodeError, KeyError):
            pass

    def _apply_overrides(self, overrides: dict):
        for section_name, values in overrides.items():
            section = getattr(self, section_name, None)
            if section is None:
                continue
            for key, val in values.items():
                if hasattr(section, key):
                    setattr(section, key, type(getattr(section, key))(val))

    def save_overrides(self, overrides: dict):
        OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if OVERRIDE_PATH.exists():
            try:
                existing = json.loads(OVERRIDE_PATH.read_text())
            except json.JSONDecodeError:
                pass
        for section, values in overrides.items():
            if section not in existing:
                existing[section] = {}
            existing[section].update(values)
        OVERRIDE_PATH.write_text(json.dumps(existing, indent=2))
        self._apply_overrides(overrides)

    def to_dict(self) -> dict:
        return asdict(self)
