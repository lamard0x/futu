import asyncio
import logging
from datetime import datetime, timezone

from src.config import Config
from src.exchange import BybitExchange
from src.risk import RiskManager
from src import telegram

logger = logging.getLogger("futu.funding")

FUNDING_HOURS = [0, 8, 16]  # Bybit funding settlement hours (UTC)


class FundingArbitrage:
    def __init__(self, config: Config, exchange: BybitExchange, risk: RiskManager):
        self.config = config
        self.exchange = exchange
        self.risk = risk
        self.active_positions: dict[str, dict] = {}
        self.running = False

    async def run(self):
        if not self.config.funding.enabled:
            logger.info("Funding arbitrage disabled")
            return

        self.running = True
        logger.info(
            "Funding arbitrage started | min_rate: %.4f%% | max_pct: %.0f%%",
            self.config.funding.min_rate * 100,
            self.config.funding.max_position_pct * 100,
        )

        while self.running:
            try:
                await self._check_and_close()
                await self._scan_opportunities()
            except Exception as e:
                logger.error("Funding error: %s", e, exc_info=True)
            await asyncio.sleep(self.config.funding.check_interval)

    async def stop(self):
        self.running = False

    async def _scan_opportunities(self):
        now = datetime.now(timezone.utc)
        next_funding = self._next_funding_time(now)
        minutes_until = (next_funding - now).total_seconds() / 60

        if minutes_until > 60:
            return

        if len(self.active_positions) >= 2:
            return

        can_trade, _ = self.risk.can_trade_new()
        if not can_trade:
            return

        rates = await self.exchange.fetch_all_funding_rates(
            self.config.funding.symbols_to_scan
        )

        for r in rates:
            sym = r["symbol"]
            rate = r["rate"]

            if abs(rate) < self.config.funding.min_rate:
                continue

            if sym in self.active_positions:
                continue

            await self._open_funding_position(sym, rate)
            break

    async def _open_funding_position(self, symbol: str, rate: float):
        side = "sell" if rate > 0 else "buy"
        balance = self.config.risk.account_balance
        amount_usd = balance * self.config.funding.max_position_pct

        orig = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            ticker = await self.exchange.get_ticker()
            price = ticker["last"]
            if price <= 0:
                return

            amount = round(amount_usd / price, 6)
            if amount <= 0:
                return

            sl_pct = 0.005
            sl_price = price * (1 - sl_pct) if side == "buy" else price * (1 + sl_pct)

            await self.exchange.setup_symbol(symbol)
            order = await self.exchange.place_market_order(
                side=side, amount=amount, sl_price=round(sl_price, 2),
            )

            if order.status in ("open", "closed", "filled", "new", "New"):
                self.active_positions[symbol] = {
                    "side": side,
                    "amount": amount,
                    "entry_price": price,
                    "rate": rate,
                    "opened_at": datetime.now(timezone.utc),
                }
                logger.info(
                    "Funding position: %s %s %.6f @ %.2f | rate: %.4f%%",
                    side, symbol.split("/")[0], amount, price, rate * 100,
                )
                await telegram.send_message(
                    f"💰 <b>FUNDING TRADE</b>\n"
                    f"🪙 {symbol.split('/')[0]} | {side.upper()}\n"
                    f"📊 Rate: <code>{rate*100:+.4f}%</code>\n"
                    f"📦 Size: <code>{amount:.6f}</code>\n"
                    f"💵 Expected: <code>${amount_usd * abs(rate):.2f}</code>"
                )
        except Exception as e:
            logger.warning("Funding open error %s: %s", symbol, e)
        finally:
            self.exchange.config.symbol = orig

    async def _check_and_close(self):
        now = datetime.now(timezone.utc)
        to_close = []

        for sym, pos in self.active_positions.items():
            age_minutes = (now - pos["opened_at"]).total_seconds() / 60
            if age_minutes > 30:
                to_close.append(sym)

        for sym in to_close:
            await self._close_funding_position(sym)

    async def _close_funding_position(self, symbol: str):
        pos = self.active_positions.get(symbol)
        if pos is None:
            return

        orig = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            close_side = "buy" if pos["side"] == "sell" else "sell"
            await self.exchange.place_market_order(
                side=close_side, amount=pos["amount"],
            )
            real_pnl = await self.exchange.get_closed_pnl(symbol)
            self.risk.on_trade_closed(real_pnl)
            del self.active_positions[symbol]

            logger.info("Funding closed %s | PnL: $%.2f", symbol.split("/")[0], real_pnl)
            await telegram.notify_close(
                symbol, real_pnl, "Funding collected",
                self.config.risk.account_balance,
            )
        except Exception as e:
            logger.warning("Funding close error %s: %s", symbol, e)
        finally:
            self.exchange.config.symbol = orig

    def _next_funding_time(self, now: datetime) -> datetime:
        for h in FUNDING_HOURS:
            candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if candidate > now:
                return candidate
        return now.replace(hour=0, minute=0, second=0, microsecond=0).replace(
            day=now.day + 1
        )
