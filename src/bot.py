import asyncio
import logging
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.config import Config
from src.exchange import Exchange
from src.indicators import compute_all
from src.strategy import (
    scan_main, scan_alert, detect_htf_bias,
    Signal, SignalType, SignalSource, Regime, HTFBias,
)
from src.risk import RiskManager
from src import telegram
from src.telegram import CommandListener
from src.funding import FundingArbitrage
from src.webhook import WebhookServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/futu.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("futu.bot")


@dataclass
class SymbolState:
    symbol: str
    bias: HTFBias = HTFBias.NEUTRAL
    has_position: bool = False
    position_candle_count: int = 0
    cooldown: int = 0


class FutuBot:
    def __init__(self):
        self.config = Config()
        self.exchange = Exchange(self.config.exchange)
        self.risk = RiskManager(config=self.config.risk)
        self.symbols: list[str] = []
        self.states: dict[str, SymbolState] = {}
        self.last_main_scan: datetime | None = None
        self.last_htf_scan: datetime | None = None
        self.last_symbol_refresh: datetime | None = None
        self.running: bool = False

    async def start(self):
        logger.info("=" * 50)
        logger.info("FUTU Scalper v1 — Multi-Symbol")
        logger.info("Leverage: %dx | Top: %d coins by volume",
                     self.config.exchange.leverage, self.config.risk.max_symbols)
        logger.info("Balance: $%.2f | Risk: %.0f%%/%.0f%%",
                     self.config.risk.account_balance,
                     self.config.risk.risk_per_trade_main * 100,
                     self.config.risk.risk_per_trade_scalp * 100)
        logger.info("=" * 50)

        await self.exchange.connect()
        await self._refresh_symbols()
        await telegram.notify_startup(self.symbols, self.config.risk.account_balance)
        self.cmd_listener = CommandListener(self)
        await self.cmd_listener.start()
        self.funding = FundingArbitrage(self.config, self.exchange, self.risk)
        asyncio.create_task(self.funding.run())
        self.webhook = WebhookServer(self.config, self)
        await self.webhook.start()
        self.running = True

        try:
            await self._run_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error("Fatal error: %s", e, exc_info=True)
            await telegram.notify_error(str(e))
        finally:
            await self.cmd_listener.stop()
            await self.funding.stop()
            await self.webhook.stop()
            await self.exchange.disconnect()

    async def _refresh_symbols(self):
        self.symbols = await self.exchange.get_top_volume_symbols(
            self.config.risk.max_symbols
        )
        for sym in self.symbols:
            if sym not in self.states:
                self.states[sym] = SymbolState(symbol=sym)
                await self.exchange.setup_symbol(sym)
        # Remove symbols no longer in top
        for sym in list(self.states.keys()):
            if sym not in self.symbols and not self.states[sym].has_position:
                del self.states[sym]
        self.last_symbol_refresh = datetime.now(timezone.utc)
        logger.info("Active symbols: %s", [s.split("/")[0] for s in self.symbols])

    async def _run_loop(self):
        while True:
            if self.running:
                try:
                    await self._tick()
                except Exception as e:
                    logger.error("Tick error: %s", e, exc_info=True)
            await asyncio.sleep(60)

    async def _tick(self):
        now = datetime.now(timezone.utc)

        # Refresh top volume symbols every 4 hours
        if self._should_refresh_symbols(now):
            await self._refresh_symbols()

        # Update H4 bias every 4 hours
        if self._should_htf_scan(now):
            await self._update_all_htf_bias()
            self.last_htf_scan = now

        # Monitor existing positions
        await self._monitor_all_positions()

        # Daily summary at day reset
        if self.risk.daily.date and self.risk.daily.date != self.risk._today() and self.risk.daily.trade_count > 0:
            await telegram.notify_daily_summary(
                self.risk.daily.wins, self.risk.daily.losses,
                self.risk.daily.pnl, self.config.risk.account_balance,
                self.risk.daily.trade_count,
            )

        # Check daily loss cap
        can_trade, reason = self.risk.can_trade()
        if not can_trade and "daily loss" in reason:
            logger.warning("Daily loss cap hit, stopping all trading")
            return

        # Main scan every 15 minutes
        if self._should_main_scan(now):
            await self._scan_all_symbols()
            self.last_main_scan = now

    def _should_refresh_symbols(self, now: datetime) -> bool:
        if self.last_symbol_refresh is None:
            return True
        return (now - self.last_symbol_refresh).total_seconds() >= 3600  # 1h

    def _should_htf_scan(self, now: datetime) -> bool:
        if self.last_htf_scan is None:
            return True
        return (now - self.last_htf_scan).total_seconds() >= 14400  # 4h

    def _should_main_scan(self, now: datetime) -> bool:
        if self.last_main_scan is None:
            return True
        elapsed = (now - self.last_main_scan).total_seconds()
        return elapsed >= self._tf_to_seconds(self.config.timeframe.main_tf)

    def _tf_to_seconds(self, tf: str) -> int:
        unit = tf[-1]
        val = int(tf[:-1])
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        return val * multipliers.get(unit, 60)

    async def _update_all_htf_bias(self):
        biases = {}
        for sym in self.symbols:
            try:
                candles = await self.exchange.fetch_candles(
                    self.config.timeframe.htf, 200, symbol=sym,
                )
                if len(candles) >= 50:
                    df = compute_all(candles, self.config.indicators)
                    self.states[sym].bias = detect_htf_bias(df)
                    biases[sym] = self.states[sym].bias.value
                    logger.info("H4 bias %s: %s", sym.split("/")[0], self.states[sym].bias.value)
            except Exception as e:
                logger.warning("HTF scan error %s: %s", sym, e)
            await asyncio.sleep(0.2)
        if biases:
            await telegram.notify_bias_update(biases)

    async def _scan_all_symbols(self):
        open_positions = sum(1 for s in self.states.values() if s.has_position)
        signals_found = 0

        for sym in self.symbols:
            state = self.states.get(sym)
            if state is None:
                continue

            if state.has_position:
                continue

            if state.cooldown > 0:
                state.cooldown -= 1
                continue

            if open_positions >= self.config.risk.max_positions:
                break

            can_trade, reason = self.risk.can_trade_new()
            if not can_trade:
                break

            try:
                signal = await self._scan_symbol(sym, state.bias)
                if signal and signal.type != SignalType.NONE:
                    signals_found += 1
                    await self._execute_signal(signal, sym)
                    open_positions += 1
            except Exception as e:
                logger.warning("Scan error %s: %s", sym, e)

            await asyncio.sleep(0.2)

        await telegram.notify_heartbeat(
            len(self.symbols), signals_found,
            open_positions, self.config.risk.account_balance,
        )

    async def _scan_symbol(self, symbol: str, bias: HTFBias) -> Signal | None:
        candles = await self.exchange.fetch_candles(
            self.config.timeframe.main_tf,
            self.config.timeframe.candle_limit,
            symbol=symbol,
        )
        if len(candles) < 50:
            return None

        df = compute_all(candles, self.config.indicators)
        signal = scan_main(df, self.config.strategy, bias)

        if signal:
            rr_ok, rr = self.risk.check_rr(signal)
            if not rr_ok:
                return None
            signal.reason = f"[{symbol.split('/')[0]}] {signal.reason}"
            logger.info("SIGNAL: %s | R:R %.2f", signal.reason, rr)

        return signal

    async def _execute_signal(self, signal: Signal, symbol: str):
        amount = self.risk.calc_position_size(signal)
        if amount <= 0:
            logger.warning("Position size too small for %s", symbol)
            return

        side = "buy" if signal.type == SignalType.LONG else "sell"
        tp_target = signal.tp1_price
        if signal.tp2_price and signal.regime == Regime.TRENDING:
            tp_target = signal.tp2_price

        rr = abs(tp_target - signal.entry_price) / abs(signal.entry_price - signal.sl_price) if signal.sl_price != signal.entry_price else 0

        logger.info(
            "EXEC %s: %s %.6f @ %.2f | SL: %.2f | TP: %.2f",
            symbol.split("/")[0], side, amount, signal.entry_price,
            signal.sl_price, tp_target,
        )

        # Temporarily set symbol for order
        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            order = await self.exchange.place_market_order(
                side=side, amount=amount,
                tp_price=tp_target, sl_price=signal.sl_price,
            )
            if order.status in ("open", "closed", "filled", "new", "New"):
                self.states[symbol].has_position = True
                self.states[symbol].position_candle_count = 0
                self.risk.on_trade_opened()
                logger.info("Trade opened: %s %s", symbol.split("/")[0], order.order_id)
                await telegram.notify_signal(
                    symbol, side, signal.entry_price, signal.sl_price,
                    tp_target, amount, rr, signal.regime.value,
                )
        finally:
            self.exchange.config.symbol = orig_symbol

    async def _monitor_all_positions(self):
        for sym, state in self.states.items():
            if not state.has_position:
                continue
            try:
                await self._monitor_position(sym, state)
            except Exception as e:
                logger.warning("Monitor error %s: %s", sym, e)

    async def _monitor_position(self, symbol: str, state: SymbolState):
        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            position = await self.exchange.get_position()
            if position is None:
                real_pnl = await self.exchange.get_closed_pnl(symbol)
                state.has_position = False
                self.risk.on_trade_closed(real_pnl)
                logger.info("%s position closed (TP/SL hit) PnL: $%.2f", symbol.split("/")[0], real_pnl)
                await telegram.notify_close(
                    symbol, real_pnl, "TP/SL hit", self.config.risk.account_balance,
                )
                return

            state.position_candle_count += 1

            # Time exit for ranging
            if state.position_candle_count >= self.config.strategy.ranging_max_candles:
                logger.info("%s time exit after %d candles", symbol.split("/")[0], state.position_candle_count)
                await self.exchange.close_position()
                pnl = position["unrealized_pnl"]
                state.has_position = False
                self.risk.on_trade_closed(pnl)
                if pnl < 0:
                    state.cooldown = self.config.risk.cooldown_candles
                await telegram.notify_close(
                    symbol, pnl, f"Time exit ({state.position_candle_count} candles)",
                    self.config.risk.account_balance,
                )
                return

            # Trailing SL with activation price
            # Only activate trailing after position is profitable (>0.3%)
            entry = position["entry_price"]
            notional = entry * position["size"] if position["size"] > 0 else 1
            pnl_pct = position["unrealized_pnl"] / notional

            if pnl_pct > 0.003:
                candles = await self.exchange.fetch_candles(self.config.timeframe.main_tf, 30, symbol=symbol)
                df = compute_all(candles, self.config.indicators)
                row = df.iloc[-1]
                new_sl = None

                if position["side"] == "long":
                    chandelier = row["chandelier_long"]
                    # Only ratchet up, never down
                    if chandelier > entry:
                        new_sl = chandelier
                elif position["side"] == "short":
                    chandelier = row["chandelier_short"]
                    if chandelier < entry:
                        new_sl = chandelier

                if new_sl is not None:
                    logger.info(
                        "%s trailing SL activated: %.2f (pnl: %.2f%%)",
                        symbol.split("/")[0], new_sl, pnl_pct * 100,
                    )
                    await self.exchange.update_tp_sl(sl_price=new_sl)
        finally:
            self.exchange.config.symbol = orig_symbol


async def main():
    bot = FutuBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
