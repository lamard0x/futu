import asyncio
import logging
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field

from src.config import Config
from src.exchange import Exchange
from src.indicators import compute_all
from src.strategy import (
    scan_main, scan_alert, scan_trending_1h, scan_trending_pullback,
    confirm_on_5m, detect_htf_bias, find_demand_zones, find_supply_zones,
    Signal, SignalType, SignalSource, Regime, HTFBias,
)
from src.risk import RiskManager
from src import telegram
from src.telegram import CommandListener
from src.funding import FundingArbitrage
from src.webhook import WebhookServer
from src.chart import generate_chart
from src.swing_scanner import run_swing_scan, format_swing_telegram

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

# Strategy reject reasons → separate debug log file
_strategy_logger = logging.getLogger("futu.strategy")
_reject_handler = logging.FileHandler("logs/rejects.log", encoding="utf-8")
_reject_handler.setLevel(logging.DEBUG)
_reject_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
_strategy_logger.addHandler(_reject_handler)
_strategy_logger.setLevel(logging.DEBUG)


@dataclass
class SymbolState:
    symbol: str
    bias_h1: HTFBias = HTFBias.NEUTRAL   # H1 bias for ranging
    bias_h4: HTFBias = HTFBias.NEUTRAL   # H4 bias for trending
    has_position: bool = False          # ranging position
    has_trending_position: bool = False  # trending position (independent)
    position_candle_count: int = 0
    trending_candle_count: int = 0
    cooldown: int = 0
    pending_signal: Signal | None = None  # 15m signal waiting for 5m confirm
    pending_ticks: int = 0                # how many ticks waiting for confirm
    partial_closed: bool = False          # TP1 hit, 50% closed, trailing rest
    limit_order_id: str | None = None     # pending limit order
    limit_order_ticks: int = 0            # ticks since limit placed
    limit_order_signal: Signal | None = None  # signal that triggered limit
    tp_price: float = 0.0                 # saved TP for trailing re-set
    sl_price: float = 0.0                 # saved SL for client-side monitor


class FutuBot:
    def __init__(self):
        self.config = Config()
        self.exchange = Exchange(self.config.exchange)
        self.risk = RiskManager(config=self.config.risk)
        self.symbols: list[str] = []
        self.states: dict[str, SymbolState] = {}
        self.last_main_scan: datetime | None = None
        self.last_5m_scan: datetime | None = None
        self.last_trending_scan: datetime | None = None
        self.last_trending_fast_scan: datetime | None = None
        self.last_htf_scan: datetime | None = None
        self.last_symbol_refresh: datetime | None = None
        self.last_swing_scan: datetime | None = None
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
        # Setup ranging symbols (fixed list)
        for sym in self.config.risk.ranging_symbols:
            if sym not in self.states:
                self.states[sym] = SymbolState(symbol=sym)
                await self.exchange.setup_symbol(sym)
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
            # Cancel ALL pending limit orders left from previous session
            try:
                open_orders = await self.exchange.exchange.fetch_open_orders()
                for o in open_orders:
                    try:
                        await self.exchange.exchange.cancel_order(o["id"], o["symbol"])
                        logger.info("Startup: cancelled stale order %s %s", o["symbol"].split("/")[0], o["id"])
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Startup cancel orders error: %s", e)

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
                self.states[sym].limit_order_id = None
                # Sync TP and SL from existing algo orders
                try:
                    inst_id = sym.replace("/", "-").replace(":USDT", "-SWAP")
                    r = await self.exchange.exchange.private_get_trade_orders_algo_pending({
                        "instId": inst_id, "ordType": "conditional",
                    })
                    for o in r.get("data", []):
                        tp_px = o.get("tpTriggerPx")
                        sl_px = o.get("slTriggerPx")
                        if tp_px:
                            self.states[sym].tp_price = float(tp_px)
                            logger.info("Synced TP for %s: %s", sym.split("/")[0], tp_px)
                        if sl_px:
                            self.states[sym].sl_price = float(sl_px)
                            logger.info("Synced SL for %s: %s", sym.split("/")[0], sl_px)
                except Exception:
                    pass
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
        """Sync real balance from OKX — update config for sizing + display."""
        try:
            balance = await self.exchange.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            total = float(usdt.get("total") or 0)
            if total > 0:
                self.config.risk.account_balance = total
                self.risk.config.account_balance = total
                logger.info("Balance synced: $%.2f", total)
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
        # Remove symbols no longer in top (but keep ranging + positions)
        ranging_set = set(self.config.risk.ranging_symbols)
        for sym in list(self.states.keys()):
            if (sym not in self.symbols
                    and sym not in ranging_set
                    and not self.states[sym].has_position
                    and not self.states[sym].has_trending_position):
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

        # HTF bias now updates with each ranging scan (not standalone)

        # Monitor pending limit orders
        await self._monitor_limit_orders()

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

        # 15m scan: detect ranging setups
        if self._should_main_scan(now):
            await self._scan_all_symbols(self.config.timeframe.main_tf)
            self.last_main_scan = now

        # 5m confirm removed — signals execute directly on 15m

        # Trending scan every 1 hour (1H TF) — breakout + pullback
        if self.config.trending.enabled and self._should_trending_scan(now):
            try:
                await self._scan_trending_symbols()
            except Exception as e:
                logger.error("Trending scan error: %s", e, exc_info=True)
            self.last_trending_scan = now

        # 30m pullback scan — more frequent trend entries
        if self.config.trending.enabled and self._should_trending_fast_scan(now):
            try:
                await self._scan_trending_pullback()
            except Exception as e:
                logger.error("Trending pullback scan error: %s", e, exc_info=True)
            self.last_trending_fast_scan = now

        # Daily swing scan
        if self.config.swing.enabled and self._should_swing_scan(now):
            try:
                await self._run_swing_scan()
            except Exception as e:
                logger.error("Swing scan error: %s", e, exc_info=True)
            self.last_swing_scan = now

    def _should_refresh_symbols(self, now: datetime) -> bool:
        if self.last_symbol_refresh is None:
            return True
        return (now - self.last_symbol_refresh).total_seconds() >= 3600  # 1h

    def _should_htf_scan(self, now: datetime) -> bool:
        if self.last_htf_scan is None:
            return True
        return (now - self.last_htf_scan).total_seconds() >= 3600  # 1h

    def _should_main_scan(self, now: datetime) -> bool:
        if self.last_main_scan is None:
            return True
        elapsed = (now - self.last_main_scan).total_seconds()
        return elapsed >= self._tf_to_seconds(self.config.timeframe.main_tf)

    def _should_5m_confirm(self, now: datetime) -> bool:
        # Only run if there are pending signals
        has_pending = any(s.pending_signal for s in self.states.values())
        if not has_pending:
            return False
        if self.last_5m_scan is None:
            return True
        return (now - self.last_5m_scan).total_seconds() >= 300  # 5min

    def _should_trending_scan(self, now: datetime) -> bool:
        if self.last_trending_scan is None:
            return True
        elapsed = (now - self.last_trending_scan).total_seconds()
        return elapsed >= self._tf_to_seconds(self.config.timeframe.trending_tf)

    def _should_swing_scan(self, now: datetime) -> bool:
        cfg = self.config.swing
        if now.hour not in cfg.scan_hours_utc or now.minute < cfg.scan_minute_utc:
            return False
        if self.last_swing_scan is None:
            return True
        return (now - self.last_swing_scan).total_seconds() >= 18000  # 5h gap

    def _should_trending_fast_scan(self, now: datetime) -> bool:
        if self.last_trending_fast_scan is None:
            return True
        elapsed = (now - self.last_trending_fast_scan).total_seconds()
        return elapsed >= self._tf_to_seconds(self.config.timeframe.trending_tf_fast)

    def _tf_to_seconds(self, tf: str) -> int:
        unit = tf[-1]
        val = int(tf[:-1])
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        return val * multipliers.get(unit, 60)

    async def _update_all_htf_bias(self):
        biases = {}
        all_syms = set(self.symbols) | set(self.config.risk.ranging_symbols)
        for sym in all_syms:
            try:
                # H1 bias for ranging
                candles_h1 = await self.exchange.fetch_candles(
                    self.config.timeframe.htf, 200, symbol=sym,
                )
                if len(candles_h1) >= 50:
                    df_h1 = compute_all(candles_h1, self.config.indicators)
                    self.states[sym].bias_h1 = detect_htf_bias(df_h1)
                    logger.info("H1 bias %s: %s", sym.split("/")[0], self.states[sym].bias_h1.value)
                # H4 bias for trending
                candles_h4 = await self.exchange.fetch_candles(
                    self.config.timeframe.htf_trending, 200, symbol=sym,
                )
                if len(candles_h4) >= 50:
                    df_h4 = compute_all(candles_h4, self.config.indicators)
                    self.states[sym].bias_h4 = detect_htf_bias(df_h4)
                    logger.info("H4 bias %s: %s", sym.split("/")[0], self.states[sym].bias_h4.value)
                biases[sym] = f"H1:{self.states[sym].bias_h1.value}/H4:{self.states[sym].bias_h4.value}"
            except Exception as e:
                logger.warning("HTF scan error %s: %s", sym, e)
            await asyncio.sleep(0.2)
        if biases:
            await telegram.notify_bias_update(biases)

    async def _scan_all_symbols(self, tf: str = None):
        """Multi-TF ranging scan on fixed large-cap list."""
        scan_tf = tf or self.config.timeframe.main_tf

        # Update HTF bias every ranging scan — not just every 1h
        if scan_tf == self.config.timeframe.main_tf:
            await self._update_all_htf_bias()

        open_positions = sum(1 for s in self.states.values() if s.has_position)
        signals_found = 0

        ranging_syms = self.config.risk.ranging_symbols
        for sym in ranging_syms:
            state = self.states.get(sym)
            if state is None:
                continue
            # Skip if already has ANY position on this symbol
            if state.has_position or state.has_trending_position:
                continue
            # Skip if in cooldown (e.g. after order failure)
            if state.cooldown > 0:
                state.cooldown -= 1
                continue

            can_trade, reason = self.risk.can_trade_new()
            if not can_trade:
                break

            try:
                # 15m scan: detect setup → execute directly (no 5m confirm)
                if scan_tf == self.config.timeframe.main_tf:
                    signal = await asyncio.wait_for(
                        self._scan_symbol(sym, state.bias_h1, state.bias_h4), timeout=30)
                    if signal and signal.type != SignalType.NONE:
                        logger.info("15m SIGNAL %s %s — executing directly",
                                    sym.split("/")[0], signal.type.value)
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
        """Scan top volume symbols for breakout signals on 1H."""
        # Use auto top volume list (refreshed every hour)
        trending_syms = self.symbols  # from _refresh_symbols (top 20 volume)
        signals_found = 0

        for sym in trending_syms:
            state = self.states.get(sym)
            if state is None:
                self.states[sym] = SymbolState(symbol=sym)
                await self.exchange.setup_symbol(sym)
                state = self.states[sym]

            # Skip if already has ANY position on this symbol
            if state.has_trending_position or state.has_position:
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

                # Pullback only — no breakout (breakout = chasing price)
                pb_signal = scan_trending_pullback(df, self.config.trending, state.bias_h4, symbol=sym.split("/")[0])
                if pb_signal and pb_signal.type != SignalType.NONE:
                    rr_ok, rr = self.risk.check_rr(pb_signal)
                    if not rr_ok:
                        logger.info("PB R:R REJECT %s: %.2f < min", sym.split("/")[0], rr)
                        continue
                    pb_signal.reason = f"[{sym.split('/')[0]}] {pb_signal.reason}"
                    logger.info("PULLBACK SIGNAL: %s | R:R %.2f", pb_signal.reason, rr)
                    signals_found += 1
                    await self._execute_trending(pb_signal, sym)
            except Exception as e:
                logger.warning("Trending scan error %s: %s", sym, e)

            await asyncio.sleep(0.2)

        if signals_found > 0:
            logger.info("Trending scan: %d signals found", signals_found)

    async def _scan_trending_pullback(self):
        """Scan top volume symbols for pullback signals on 30m."""
        trending_syms = self.symbols
        signals_found = 0

        for sym in trending_syms:
            state = self.states.get(sym)
            if state is None:
                self.states[sym] = SymbolState(symbol=sym)
                await self.exchange.setup_symbol(sym)
                state = self.states[sym]

            if state.has_trending_position or state.has_position:
                continue

            can_trade, reason = self.risk.can_trade()
            if not can_trade and "daily loss" in reason:
                break

            try:
                candles = await self.exchange.fetch_candles(
                    self.config.timeframe.trending_tf_fast,
                    self.config.timeframe.candle_limit,
                    symbol=sym,
                )
                if len(candles) < 30:
                    continue

                df = compute_all(candles, self.config.indicators)
                signal = scan_trending_pullback(df, self.config.trending, state.bias_h4, symbol=sym.split("/")[0])

                if signal and signal.type != SignalType.NONE:
                    rr_ok, rr = self.risk.check_rr(signal)
                    if not rr_ok:
                        logger.info("PB30 R:R REJECT %s: %.2f < min", sym.split("/")[0], rr)
                        continue
                    signal.reason = f"[{sym.split('/')[0]}] {signal.reason}"
                    logger.info("PB30 SIGNAL: %s | R:R %.2f", signal.reason, rr)
                    signals_found += 1
                    await self._execute_trending(signal, sym)
            except Exception as e:
                logger.warning("Pullback 30m scan error %s: %s", sym, e)

            await asyncio.sleep(0.2)

        if signals_found > 0:
            logger.info("Pullback 30m scan: %d signals found", signals_found)

    async def _execute_trending(self, signal: Signal, symbol: str):
        """Execute trending trade — limit order at breakout level (retest entry). Half size."""
        amount = self.risk.calc_position_size(signal, leverage=self.config.exchange.leverage)
        amount = amount * 0.5  # trending = 50% size of ranging
        if amount <= 0:
            logger.warning("Trending position size too small for %s", symbol)
            return

        side = "buy" if signal.type == SignalType.LONG else "sell"
        tp_target = signal.tp1_price
        limit_price = signal.entry_price  # breakout level — wait for retest

        logger.info(
            "TRENDING LIMIT %s: %s %.6f @ %g | SL: %g | TP: %g",
            symbol.split("/")[0], side, amount, limit_price,
            signal.sl_price, tp_target,
        )

        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            # Cancel stale algo orders before new entry
            await self.exchange._cancel_algo_orders()

            order = await self.exchange.place_limit_order(
                side=side, amount=amount, price=limit_price,
            )
            if order.status in ("closed", "filled"):
                self.states[symbol].has_trending_position = True
                self.states[symbol].trending_candle_count = 0
                self.states[symbol].partial_closed = False
                self.risk.on_trade_opened()
                await asyncio.sleep(0.5)
                await self.exchange.update_tp_sl(tp_price=tp_target, sl_price=signal.sl_price)
                logger.info("Trending filled immediately: %s", symbol.split("/")[0])
            elif order.status not in ("canceled", "cancelled", "rejected", "expired"):
                # Pending limit — track it
                state = self.states[symbol]
                state.limit_order_id = order.order_id
                state.limit_order_ticks = 0
                state.limit_order_signal = signal
                state.has_trending_position = True
                logger.info("Trending limit pending: %s — retest entry", symbol.split("/")[0])
        finally:
            self.exchange.config.symbol = orig_symbol

    async def _run_swing_scan(self):
        logger.info("Starting daily swing scan...")
        try:
            signals = await run_swing_scan(self.config.swing)
            if not signals:
                await telegram.send_message("📊 <b>Swing Scan</b>\nNo signals today.")
                logger.info("Swing scan: no signals")
                return
            for sig in signals:
                msg = format_swing_telegram(sig)
                await telegram.send_message(msg)
                await asyncio.sleep(1)
            await telegram.send_message(
                f"📊 <b>Swing Scan Complete</b>\n{len(signals)} signal(s) found."
            )
            logger.info("Swing scan: %d signals sent", len(signals))
        except Exception as e:
            logger.error("Swing scan failed: %s", e, exc_info=True)
            await telegram.send_message(f"⚠️ Swing scan error: {e}")

    async def run_swing_scan_manual(self):
        await self._run_swing_scan()

    async def _scan_symbol(self, symbol: str, bias: HTFBias, bias_h4: HTFBias = HTFBias.NEUTRAL) -> Signal | None:
        candles = await self.exchange.fetch_candles(
            self.config.timeframe.main_tf,
            self.config.timeframe.candle_limit,
            symbol=symbol,
        )
        if len(candles) < 50:
            return None

        # Fetch H1/H4 for demand/supply zone confluence scoring
        zones_h1_demand, zones_h1_supply = [], []
        zones_h4_demand, zones_h4_supply = [], []
        for htf in ("1h", "4h"):
            try:
                htf_candles = await self.exchange.fetch_candles(htf, 100, symbol=symbol)
                if len(htf_candles) >= 20:
                    htf_df = compute_all(htf_candles, self.config.indicators)
                    if htf == "1h":
                        zones_h1_demand = find_demand_zones(htf_df)
                        zones_h1_supply = find_supply_zones(htf_df)
                    else:
                        zones_h4_demand = find_demand_zones(htf_df, strength=2)
                        zones_h4_supply = find_supply_zones(htf_df, strength=2)
            except Exception:
                pass

        df = compute_all(candles, self.config.indicators)
        signal = scan_main(df, self.config.strategy, bias, symbol=symbol.split("/")[0],
                          zones_h1_demand=zones_h1_demand, zones_h1_supply=zones_h1_supply,
                          zones_h4_demand=zones_h4_demand, zones_h4_supply=zones_h4_supply)

        # If no ranging signal, try trending pullback on same 15m data
        if signal is None:
            pb_signal = scan_trending_pullback(df, self.config.trending, bias_h4, symbol=symbol.split("/")[0])
            if pb_signal and pb_signal.type != SignalType.NONE:
                signal = pb_signal

        if signal:
            rr_ok, rr = self.risk.check_rr(signal)
            if not rr_ok:
                logger.info("R:R REJECT %s: %.2f < min", symbol.split("/")[0], rr)
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

        # Ranging: limit order at BB band for better entry
        # Trending: limit order at breakout level (retest entry)
        limit_price = signal.entry_price  # BB lower/upper for ranging, breakout level for trending

        logger.info(
            "LIMIT %s: %s %.6f @ %g | SL: %g | TP: %g",
            symbol.split("/")[0], side, amount, limit_price,
            signal.sl_price, tp_target,
        )

        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            # Cancel stale algo orders before new entry
            await self.exchange._cancel_algo_orders()

            try:
                order = await self.exchange.place_limit_order(
                    side=side, amount=amount, price=limit_price,
                )
            except Exception as e:
                logger.warning("Limit order FAILED %s: %s", symbol.split("/")[0], e)
                self.states[symbol].cooldown = 6  # block re-signal for ~3 min
                return
            logger.info("Limit order: %s status=%s id=%s", symbol.split("/")[0], order.status, order.order_id)
            if order.status in ("canceled", "cancelled", "rejected", "expired"):
                logger.warning("Limit order rejected: %s", symbol.split("/")[0])
            elif order.status in ("closed", "filled"):
                # Filled immediately
                self.states[symbol].has_position = True
                self.states[symbol].position_candle_count = 0
                self.states[symbol].partial_closed = False
                self.states[symbol].tp_price = tp_target
                self.states[symbol].sl_price = signal.sl_price
                self.risk.on_trade_opened()
                # Set TP/SL
                await asyncio.sleep(0.5)
                await self.exchange.update_tp_sl(tp_price=tp_target, sl_price=signal.sl_price)
                logger.info("Trade opened (filled): %s %s", symbol.split("/")[0], order.order_id)
            else:
                # Pending — track limit order for monitoring
                state = self.states[symbol]
                state.limit_order_id = order.order_id
                state.limit_order_ticks = 0
                state.limit_order_signal = signal
                state.has_position = True  # block new signals for this symbol
                logger.info("Limit order pending: %s %s — will cancel after 15 min", symbol.split("/")[0], order.order_id)
        finally:
            self.exchange.config.symbol = orig_symbol

    async def _monitor_limit_orders(self):
        """Check pending limit orders — fill → set TP/SL, or cancel after 6 ticks."""
        for sym, state in list(self.states.items()):
            if not state.limit_order_id:
                continue
            state.limit_order_ticks += 1

            orig_symbol = self.exchange.config.symbol
            self.exchange.config.symbol = sym
            try:
                # Check if filled
                position = await self.exchange.get_position()
                if position is not None:
                    # Filled — set TP/SL
                    signal = state.limit_order_signal
                    tp_sl_ok = False
                    if signal:
                        tp = signal.tp1_price
                        if signal.tp2_price and signal.regime == Regime.TRENDING:
                            tp = signal.tp2_price
                        state.tp_price = tp
                        state.sl_price = signal.sl_price
                        try:
                            await self.exchange.update_tp_sl(tp_price=tp, sl_price=signal.sl_price)
                            tp_sl_ok = True
                        except Exception as e:
                            err_str = str(e)
                            if "51280" in err_str:
                                # SL rejected — price already past SL level
                                # TP was set successfully (set first), check if price past SL
                                mark = float(position.get("markPrice") or 0)
                                sl_breached = False
                                if position["side"] == "long" and mark > 0 and mark <= signal.sl_price:
                                    sl_breached = True
                                elif position["side"] == "short" and mark > 0 and mark >= signal.sl_price:
                                    sl_breached = True
                                if sl_breached:
                                    logger.error("SL past price %s: mark=%s SL=%s — closing", sym.split("/")[0], mark, signal.sl_price)
                                    close_side = "sell" if position["side"] == "long" else "buy"
                                    try:
                                        await self.exchange.place_market_order(close_side, abs(position["size"]), reduce_only=True)
                                    except Exception as close_err:
                                        logger.error("Emergency close FAILED %s: %s", sym.split("/")[0], close_err)
                                    state.has_position = False
                                    state.has_trending_position = False
                                else:
                                    # Mark NOT past SL yet — keep position, client-side SL will monitor
                                    logger.warning("SL reject %s but mark=%s OK — using client SL at %s", sym.split("/")[0], mark, signal.sl_price)
                                    tp_sl_ok = True
                            else:
                                logger.error("TP/SL FAILED for %s: %s — closing position", sym.split("/")[0], e)
                                close_side = "sell" if position["side"] == "long" else "buy"
                                try:
                                    await self.exchange.place_market_order(close_side, abs(position["size"]), reduce_only=True)
                                    logger.info("Emergency close %s — no TP/SL", sym.split("/")[0])
                                except Exception as close_err:
                                    logger.error("Emergency close FAILED %s: %s", sym.split("/")[0], close_err)
                                state.has_position = False
                                state.has_trending_position = False
                    else:
                        tp_sl_ok = True
                    state.limit_order_id = None
                    state.limit_order_signal = None
                    state.limit_order_ticks = 0
                    state.position_candle_count = 0
                    state.partial_closed = False
                    if tp_sl_ok:
                        self.risk.on_trade_opened()
                        logger.info("Limit filled: %s — TP/SL set", sym.split("/")[0])
                    if signal and tp_sl_ok:
                        side = "buy" if signal.type == SignalType.LONG else "sell"
                        tp = signal.tp1_price
                        rr = abs(tp - signal.entry_price) / abs(signal.entry_price - signal.sl_price) if signal.sl_price != signal.entry_price else 0
                        chart_bytes = None
                        try:
                            candles = await self.exchange.fetch_candles(
                                self.config.timeframe.main_tf, 100, symbol=sym,
                            )
                            chart_bytes = generate_chart(
                                candles, self.config.indicators,
                                symbol=sym.split("/")[0] + "/USDT",
                                entries=[{
                                    "timestamp": None, "type": "entry",
                                    "side": side, "price": signal.entry_price,
                                    "sl": signal.sl_price, "tp": tp,
                                }],
                                regime=signal.regime.value,
                                bias=state.bias_h1.value,
                            )
                        except Exception:
                            pass
                        await telegram.notify_signal(
                            sym, side, signal.entry_price, signal.sl_price,
                            tp, position["size"], rr, signal.regime.value, chart_bytes,
                        )
                elif state.limit_order_ticks >= 30:
                    # 30 ticks (15 min) — cancel
                    try:
                        await self.exchange.exchange.cancel_order(
                            state.limit_order_id, sym)
                    except Exception:
                        pass
                    logger.info("Limit cancelled: %s — not filled after 15 min",
                                sym.split("/")[0])
                    state.limit_order_id = None
                    state.limit_order_signal = None
                    state.limit_order_ticks = 0
                    state.has_position = False
                    state.has_trending_position = False
            except Exception as e:
                logger.warning("Monitor limit error %s: %s", sym, e)
            finally:
                self.exchange.config.symbol = orig_symbol

    async def _monitor_all_positions(self):
        for sym, state in list(self.states.items()):
            # Skip if pending limit order (handled by _monitor_limit_orders)
            if state.limit_order_id:
                continue
            if state.has_position or state.has_trending_position:
                try:
                    await self._monitor_position(sym, state)
                except Exception as e:
                    logger.warning("Monitor error %s: %s", sym, e)

    async def _monitor_position(self, symbol: str, state: SymbolState):
        """Monitor a single symbol's position. One exchange position per symbol."""
        regime = "trending" if state.has_trending_position else "ranging"
        orig_symbol = self.exchange.config.symbol
        self.exchange.config.symbol = symbol
        try:
            position = await self.exchange.get_position()
            if position is None:
                # Cancel remaining algo orders to prevent ghost positions
                await self.exchange._cancel_algo_orders()
                real_pnl = await self.exchange.get_closed_pnl(symbol)
                # Clear both flags — only one exchange position per symbol
                state.has_position = False
                state.has_trending_position = False
                state.partial_closed = False
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
            if max_candles > 0 and candle_count >= max_candles:
                logger.info("%s %s time exit after %d candles",
                            symbol.split("/")[0], regime, candle_count)
                await self.exchange.close_position()
                pnl = position["unrealized_pnl"]
                state.has_position = False
                state.has_trending_position = False
                state.partial_closed = False
                self.risk.on_trade_closed(pnl)
                await self._sync_balance()
                if pnl < 0:
                    state.cooldown = self.config.risk.cooldown_candles
                await telegram.notify_close(
                    symbol, pnl, f"{regime} time exit ({candle_count} bars)",
                    self.config.risk.account_balance,
                )
                return

            # Move SL to breakeven when price reaches 50% of TP distance
            entry = position["entry_price"]
            tp = state.tp_price
            mark = float(position.get("markPrice") or 0)
            if mark <= 0:
                mark = entry  # fallback — trailing won't trigger but client SL still works

            if tp > 0 and not state.partial_closed:
                if position["side"] == "long":
                    tp_dist = tp - entry
                    mid_point = entry + tp_dist * 0.5
                    if tp_dist > 0 and mark >= mid_point and mark > entry:
                        logger.info(
                            "%s %s SL → breakeven (mark %g >= 50%% target %g)",
                            symbol.split("/")[0], regime, mark, mid_point,
                        )
                        await self.exchange.update_tp_sl(sl_price=entry)
                        state.sl_price = entry
                        state.partial_closed = True
                elif position["side"] == "short":
                    tp_dist = entry - tp
                    mid_point = entry - tp_dist * 0.5
                    if tp_dist > 0 and mark <= mid_point and mark < entry:
                        logger.info(
                            "%s %s SL → breakeven (mark %g <= 50%% target %g)",
                            symbol.split("/")[0], regime, mark, mid_point,
                        )
                        await self.exchange.update_tp_sl(sl_price=entry)
                        state.sl_price = entry
                        state.partial_closed = True

            # Client-side SL monitor — backup if OKX algo order fails
            if state.sl_price > 0:
                sl_breached = False
                if position["side"] == "long" and mark <= state.sl_price:
                    sl_breached = True
                elif position["side"] == "short" and mark >= state.sl_price:
                    sl_breached = True
                if sl_breached:
                    logger.error(
                        "CLIENT SL BREACH %s: mark %g vs SL %g — emergency close",
                        symbol.split("/")[0], mark, state.sl_price,
                    )
                    try:
                        await self.exchange.close_position()
                        pnl = position["unrealized_pnl"]
                        state.has_position = False
                        state.has_trending_position = False
                        state.partial_closed = False
                        self.risk.on_trade_closed(pnl)
                        await self._sync_balance()
                        await telegram.notify_close(
                            symbol, pnl, "CLIENT SL (algo failed)",
                            self.config.risk.account_balance,
                        )
                    except Exception as close_err:
                        logger.error("CLIENT SL close FAILED %s: %s", symbol.split("/")[0], close_err)
                    return
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
