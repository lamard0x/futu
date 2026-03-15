import asyncio
import logging
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.config import Config
from src.exchange import Exchange
from src.indicators import compute_all
from src.strategy import (
    scan_main, scan_alert, scan_trending_1h, detect_htf_bias,
    Signal, SignalType, SignalSource, Regime, HTFBias,
)
from src.risk import RiskManager
from src import telegram
from src.telegram import CommandListener
from src.funding import FundingArbitrage
from src.webhook import WebhookServer
from src.chart import generate_chart

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
    has_position: bool = False          # ranging position
    has_trending_position: bool = False  # trending position (independent)
    position_candle_count: int = 0
    trending_candle_count: int = 0
    cooldown: int = 0


class FutuBot:
    def __init__(self):
        self.config = Config()
        self.exchange = Exchange(self.config.exchange)
        self.risk = RiskManager(config=self.config.risk)
        self.symbols: list[str] = []
        self.states: dict[str, SymbolState] = {}
        self.last_main_scan: datetime | None = None
        self.last_trending_scan: datetime | None = None
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
        await self._sync_open_positions()
        await telegram.notify_startup(self.symbols, self.config.risk.account_balance)
        self.cmd_listener = CommandListener(self)
        await self.cmd_listener.start()
        self.funding = FundingArbitrage(self.config, self.exchange, self.risk)
        asyncio.create_task(self.funding.run())
        self.webhook = WebhookServer(self.config, self)
        await self.webhook.start()
        self.running = True

        logger.info("Entering main loop...")
        try:
            await self._run_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error("Fatal error: %s", e, exc_info=True)
            import traceback
            traceback.print_exc()
            await telegram.notify_error(str(e))
        finally:
            await self.cmd_listener.stop()
            await self.funding.stop()
            await self.webhook.stop()
            await self.exchange.disconnect()

    async def _sync_open_positions(self):
        """Sync open positions from exchange on startup — prevents orphaned trades."""
        try:
            positions = await self.exchange.exchange.fetch_positions()
            synced = 0
            for p in positions:
                size = float(p.get("contracts", 0))
                if size <= 0:
                    continue
                sym = p["symbol"]
                if sym not in self.states:
                    self.states[sym] = SymbolState(symbol=sym)
                self.states[sym].has_position = True
                synced += 1
                logger.info("Synced position: %s %s %.4f @ %.2f",
                            sym.split("/")[0], p.get("side", "?"),
                            size, float(p.get("entryPrice") or 0))
            if synced:
                logger.info("Synced %d open positions from exchange", synced)
            else:
                logger.info("No open positions on exchange")
            # Sync balance from exchange
            await self._sync_balance()
        except Exception as e:
            logger.warning("Position sync error: %s", e)

    async def _sync_balance(self):
        """Sync real balance for display, but position sizing uses fixed $300."""
        try:
            balance = await self.exchange.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            total = float(usdt.get("total") or 0)
            if total > 0:
                logger.info("Exchange balance: $%.2f | Trading with: $%.2f",
                            total, self.config.risk.account_balance)
        except Exception as e:
            logger.warning("Balance sync error: %s", e)

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
        logger.info("Main loop started")
        while True:
            if self.running:
                try:
                    await self._tick()
                except Exception as e:
                    logger.error("Tick error: %s", e, exc_info=True)
            logger.info("Sleeping 30s until next tick...")
            await asyncio.sleep(30)

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

        # Ranging scan every 15 minutes (15m TF)
        if self._should_main_scan(now):
            await self._scan_all_symbols()
            self.last_main_scan = now
            logger.info("Ranging scan complete, next in %s", self.config.timeframe.main_tf)

        # Trending scan every 1 hour (1H TF)
        if self.config.trending.enabled and self._should_trending_scan(now):
            try:
                await self._scan_trending_symbols()
            except Exception as e:
                logger.error("Trending scan error: %s", e, exc_info=True)
            self.last_trending_scan = now

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

    def _should_trending_scan(self, now: datetime) -> bool:
        if self.last_trending_scan is None:
            return True
        elapsed = (now - self.last_trending_scan).total_seconds()
        return elapsed >= self._tf_to_seconds(self.config.timeframe.trending_tf)

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
        """Scan all symbols for ranging signals on 15m. No position cap — 1 per symbol."""
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

            can_trade, reason = self.risk.can_trade_new()
            if not can_trade:
                break

            try:
                signal = await asyncio.wait_for(
                    self._scan_symbol(sym, state.bias), timeout=30,
                )
                if signal and signal.type != SignalType.NONE:
                    signals_found += 1
                    await self._execute_signal(signal, sym)
                    open_positions += 1
            except asyncio.TimeoutError:
                logger.warning("Scan timeout %s", sym.split("/")[0])
            except Exception as e:
                logger.warning("Scan error %s: %s", sym, e)

            await asyncio.sleep(0.2)

        try:
            await asyncio.wait_for(
                telegram.notify_heartbeat(
                    len(self.symbols), signals_found,
                    open_positions, self.config.risk.account_balance,
                ), timeout=15,
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Heartbeat notify failed: %s", e)

    async def _scan_trending_symbols(self):
        """Scan trending symbols for breakout signals on 1H. Independent from ranging."""
        trending_syms = self.config.trending.symbols
        signals_found = 0

        for sym in trending_syms:
            state = self.states.get(sym)
            if state is None:
                continue

            if state.has_trending_position:
                continue

            can_trade, reason = self.risk.can_trade()
            if not can_trade and "daily loss" in reason:
                break

            try:
                candles = await self.exchange.fetch_candles(
                    self.config.timeframe.trending_tf,
                    self.config.timeframe.candle_limit,
                    symbol=sym,
                )
                if len(candles) < 50:
                    continue

                df = compute_all(candles, self.config.indicators)
                signal = scan_trending_1h(df, self.config.trending, state.bias)

                if signal and signal.type != SignalType.NONE:
                    rr_ok, rr = self.risk.check_rr(signal)
                    if not rr_ok:
                        continue
                    signal.reason = f"[{sym.split('/')[0]}] {signal.reason}"
                    logger.info("TREND SIGNAL: %s | R:R %.2f", signal.reason, rr)
                    signals_found += 1
                    await self._execute_trending(signal, sym)
            except Exception as e:
                logger.warning("Trending scan error %s: %s", sym, e)

            await asyncio.sleep(0.2)

        if signals_found > 0:
            logger.info("Trending scan: %d signals found", signals_found)

    async def _execute_trending(self, signal: Signal, symbol: str):
        """Execute trending trade — no fixed TP, only trailing SL."""
        amount = self.risk.calc_position_size(signal, leverage=self.config.exchange.leverage)
        if amount <= 0:
            logger.warning("Trending position size too small for %s", symbol)
            return

        side = "buy" if signal.type == SignalType.LONG else "sell"

        logger.info(
            "TRENDING EXEC %s: %s %.6f @ %.2f | SL: %.2f",
            symbol.split("/")[0], side, amount, signal.entry_price, signal.sl_price,
        )

        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            order = await self.exchange.place_market_order(
                side=side, amount=amount,
                tp_price=None, sl_price=signal.sl_price,
            )
            if order.status not in ("canceled", "cancelled", "rejected", "expired"):
                self.states[symbol].has_trending_position = True
                self.states[symbol].trending_candle_count = 0
                self.risk.on_trade_opened()
                logger.info("Trending trade opened: %s %s", symbol.split("/")[0], order.order_id)
                rr = abs(signal.tp1_price - signal.entry_price) / abs(signal.entry_price - signal.sl_price) if signal.sl_price != signal.entry_price else 0
                await telegram.notify_signal(
                    symbol, side, signal.entry_price, signal.sl_price,
                    signal.tp1_price, amount, rr, "trending", None,
                )
        finally:
            self.exchange.config.symbol = orig_symbol

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
        amount = self.risk.calc_position_size(signal, leverage=self.config.exchange.leverage)
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
            logger.info("Order result: %s status=%s", symbol.split("/")[0], order.status)
            if order.status not in ("canceled", "cancelled", "rejected", "expired"):
                self.states[symbol].has_position = True
                self.states[symbol].position_candle_count = 0
                self.risk.on_trade_opened()
                logger.info("Trade opened: %s %s", symbol.split("/")[0], order.order_id)
                try:
                    candles = await self.exchange.fetch_candles(
                        self.config.timeframe.main_tf, 100, symbol=symbol,
                    )
                    chart_bytes = generate_chart(
                        candles, self.config.indicators,
                        symbol=symbol.split("/")[0] + "/USDT",
                        entries=[{
                            "timestamp": None, "type": "entry",
                            "side": side, "price": signal.entry_price,
                            "sl": signal.sl_price, "tp": tp_target,
                        }],
                        regime=signal.regime.value,
                        bias=self.states.get(symbol, SymbolState(symbol=symbol)).bias.value,
                    )
                except Exception as e:
                    logger.warning("Chart generation failed: %s", e)
                    chart_bytes = None
                await telegram.notify_signal(
                    symbol, side, signal.entry_price, signal.sl_price,
                    tp_target, amount, rr, signal.regime.value, chart_bytes,
                )
        finally:
            self.exchange.config.symbol = orig_symbol

    async def _monitor_all_positions(self):
        for sym, state in self.states.items():
            if state.has_position:
                try:
                    await self._monitor_position(sym, state, regime="ranging")
                except Exception as e:
                    logger.warning("Monitor error %s ranging: %s", sym, e)
            if state.has_trending_position:
                try:
                    await self._monitor_position(sym, state, regime="trending")
                except Exception as e:
                    logger.warning("Monitor error %s trending: %s", sym, e)

    async def _monitor_position(self, symbol: str, state: SymbolState, regime: str = "ranging"):
        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            position = await self.exchange.get_position()
            if position is None:
                real_pnl = await self.exchange.get_closed_pnl(symbol)
                if regime == "trending":
                    state.has_trending_position = False
                else:
                    state.has_position = False
                self.risk.on_trade_closed(real_pnl)
                await self._sync_balance()
                logger.info("%s %s closed (TP/SL hit) PnL: $%.2f",
                            symbol.split("/")[0], regime, real_pnl)
                await telegram.notify_close(
                    symbol, real_pnl, "TP/SL hit", self.config.risk.account_balance,
                )
                return

            if regime == "trending":
                state.trending_candle_count += 1
                max_candles = self.config.trending.max_hold_bars
                candle_count = state.trending_candle_count
            else:
                state.position_candle_count += 1
                max_candles = self.config.strategy.ranging_max_candles
                candle_count = state.position_candle_count

            # Time exit
            if candle_count >= max_candles:
                logger.info("%s %s time exit after %d candles",
                            symbol.split("/")[0], regime, candle_count)
                await self.exchange.close_position()
                pnl = position["unrealized_pnl"]
                if regime == "trending":
                    state.has_trending_position = False
                else:
                    state.has_position = False
                self.risk.on_trade_closed(pnl)
                await self._sync_balance()
                if pnl < 0:
                    state.cooldown = self.config.risk.cooldown_candles
                await telegram.notify_close(
                    symbol, pnl, f"{regime} time exit ({candle_count} bars)",
                    self.config.risk.account_balance,
                )
                return

            # Trailing SL (chandelier) — activate after >0.3% profit
            entry = position["entry_price"]
            notional = entry * position["size"] if position["size"] > 0 else 1
            pnl_pct = position["unrealized_pnl"] / notional

            if pnl_pct > 0.003:
                tf = self.config.timeframe.trending_tf if regime == "trending" else self.config.timeframe.main_tf
                candles = await self.exchange.fetch_candles(tf, 30, symbol=symbol)
                df = compute_all(candles, self.config.indicators)
                row = df.iloc[-1]
                new_sl = None

                if position["side"] == "long":
                    chandelier = row["chandelier_long"]
                    if chandelier > entry:
                        new_sl = chandelier
                elif position["side"] == "short":
                    chandelier = row["chandelier_short"]
                    if chandelier < entry:
                        new_sl = chandelier

                if new_sl is not None:
                    logger.info(
                        "%s %s trailing SL: %.2f (pnl: %.2f%%)",
                        symbol.split("/")[0], regime, new_sl, pnl_pct * 100,
                    )
                    await self.exchange.update_tp_sl(sl_price=new_sl)
        finally:
            self.exchange.config.symbol = orig_symbol


async def main():
    bot = FutuBot()
    await bot.start()


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error("Bot crashed, restarting in 10s: %s", e)
            import time
            time.sleep(10)
