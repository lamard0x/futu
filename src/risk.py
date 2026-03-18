import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import RiskConfig
from src.strategy import Signal, SignalSource, Regime

logger = logging.getLogger("futu.risk")


@dataclass
class DailyStats:
    date: str = ""
    total_loss: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0


@dataclass
class RiskManager:
    config: RiskConfig
    daily: DailyStats = field(default_factory=DailyStats)
    cooldown_remaining: int = 0
    has_position: bool = False

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _reset_if_new_day(self):
        today = self._today()
        if self.daily.date != today:
            if self.daily.date:
                logger.info(
                    "Day reset | prev: %s trades, PnL: $%.2f",
                    self.daily.trade_count, self.daily.pnl,
                )
            self.daily = DailyStats(date=today)

    def can_trade(self) -> tuple[bool, str]:
        self._reset_if_new_day()

        max_loss = self.config.account_balance * self.config.max_daily_loss_pct
        if self.daily.total_loss >= max_loss:
            return False, f"daily loss cap hit: ${self.daily.total_loss:.2f} >= ${max_loss:.2f}"

        return True, "ok"

    def can_trade_new(self) -> tuple[bool, str]:
        self._reset_if_new_day()

        max_loss = self.config.account_balance * self.config.max_daily_loss_pct
        if self.daily.total_loss >= max_loss:
            return False, f"daily loss cap hit"

        return True, "ok"

    def calc_position_size(self, signal: Signal, leverage: int = 10) -> float:
        risk_pct = (
            self.config.risk_per_trade_main
            if signal.source == SignalSource.MAIN
            else self.config.risk_per_trade_scalp
        )
        risk_amount = self.config.account_balance * risk_pct
        sl_distance_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price
        if sl_distance_pct == 0:
            return 0.0
        position_value = risk_amount / sl_distance_pct

        # Cap notional: expect max ~6 concurrent positions
        max_notional = (self.config.account_balance * leverage) / 6
        if position_value > max_notional:
            logger.info(
                "Size capped: $%.0f -> $%.0f (margin limit)",
                position_value, max_notional,
            )
            position_value = max_notional

        amount_in_coin = position_value / signal.entry_price
        return round(amount_in_coin, 6)

    def check_rr(self, signal: Signal) -> tuple[bool, float]:
        risk = abs(signal.entry_price - signal.sl_price)
        reward = abs(signal.tp1_price - signal.entry_price)
        if risk == 0:
            return False, 0.0
        rr = reward / risk
        if signal.regime == Regime.TRENDING:
            min_rr = self.config.min_rr_trending
        elif signal.regime == Regime.RANGING and signal.confluence_score >= 2:
            min_rr = self.config.min_rr_ranging_confluence
        else:
            min_rr = self.config.min_rr_ranging
        return rr >= min_rr, rr

    def on_trade_opened(self):
        self._reset_if_new_day()
        self.has_position = True
        self.daily.trade_count += 1

    def on_trade_closed(self, pnl: float):
        self.has_position = False
        self.daily.pnl += pnl
        if pnl < 0:
            self.daily.total_loss += abs(pnl)
            self.daily.losses += 1
            self.cooldown_remaining = self.config.cooldown_candles
            logger.info("Loss: $%.2f | cooldown %d candles", pnl, self.cooldown_remaining)
        else:
            self.daily.wins += 1
            logger.info("Win: $%.2f", pnl)
        self.config.account_balance += pnl
        logger.info(
            "Balance: $%.2f | Day W/L: %d/%d | PnL: $%.2f",
            self.config.account_balance,
            self.daily.wins, self.daily.losses, self.daily.pnl,
        )

    def tick_cooldown(self):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
