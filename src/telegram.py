import os
import logging
import asyncio
import aiohttp
from datetime import datetime, timezone

logger = logging.getLogger("futu.telegram")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

COMMANDS = [
    {"command": "status", "description": "Bot status + balance + positions"},
    {"command": "on", "description": "Enable trading"},
    {"command": "off", "description": "Pause trading"},
    {"command": "positions", "description": "Show open positions"},
    {"command": "stats", "description": "Today's trading stats"},
    {"command": "bias", "description": "Current H4 bias all coins"},
    {"command": "chart", "description": "Chart with indicators (e.g. /chart BTC)"},
    {"command": "config", "description": "Show current config"},
    {"command": "swing", "description": "Run swing scanner now"},
    {"command": "help", "description": "List all commands"},
]


# ── Core send ──

async def send_message(text: str, parse_mode: str = "HTML"):
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured (token=%s, chat=%s)", bool(BOT_TOKEN), bool(CHAT_ID))
        return
    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{API_URL}/sendMessage",
                json={
                    "chat_id": CHAT_ID,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            )
            data = await resp.json()
            if not data.get("ok"):
                logger.warning("Telegram API error: %s", data.get("description", "unknown"))
    except Exception as e:
        logger.warning("Telegram send error: %s", e)


async def send_photo(photo_bytes: bytes, caption: str = ""):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        data = aiohttp.FormData()
        data.add_field("chat_id", CHAT_ID)
        data.add_field("photo", photo_bytes, filename="chart.png", content_type="image/png")
        if caption:
            data.add_field("caption", caption)
            data.add_field("parse_mode", "HTML")
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{API_URL}/sendPhoto",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30),
            )
    except Exception as e:
        logger.warning("Telegram send photo error: %s", e)


async def register_commands():
    if not BOT_TOKEN:
        return
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{API_URL}/setMyCommands",
                json={"commands": COMMANDS},
                timeout=aiohttp.ClientTimeout(total=10),
            )
        logger.info("Telegram commands registered")
    except Exception as e:
        logger.warning("Register commands error: %s", e)


# ── Command listener ──

class CommandListener:
    def __init__(self, bot):
        self.bot = bot
        self.last_update_id = 0
        self.running = False

    async def start(self):
        await register_commands()
        self.running = True
        asyncio.create_task(self._poll_loop())
        logger.info("Telegram command listener started")

    async def stop(self):
        self.running = False

    async def _poll_loop(self):
        logger.info("Poll loop task started")
        while self.running:
            try:
                await self._poll()
            except asyncio.CancelledError:
                logger.info("Poll loop cancelled")
                return
            except Exception as e:
                logger.warning("Telegram poll error: %s", e)
            await asyncio.sleep(2)
        logger.info("Poll loop ended (running=False)")

    async def _poll(self):
        if not BOT_TOKEN:
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{API_URL}/getUpdates",
                    params={"offset": self.last_update_id + 1, "timeout": 5},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
        except Exception:
            return

        for update in data.get("result", []):
            self.last_update_id = update["update_id"]
            msg = update.get("message", {})
            text = msg.get("text", "")
            chat_id = str(msg.get("chat", {}).get("id", ""))

            if chat_id != CHAT_ID:
                continue

            if text.startswith("/"):
                parts = text.split()
                cmd = parts[0].replace("/", "").replace("@", " ").split()[0]
                args = parts[1:] if len(parts) > 1 else []
                logger.info("Command received: /%s", cmd)
                try:
                    await self._handle_command(cmd, args)
                except Exception as e:
                    logger.warning("Command /%s error: %s", cmd, e)
                    await send_message(f"Error: {e}")

    async def _handle_command(self, cmd: str, args: list[str] = None):
        if cmd == "chart":
            await self._cmd_chart(args or [])
            return
        handlers = {
            "start": self._cmd_help,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "on": self._cmd_on,
            "off": self._cmd_off,
            "positions": self._cmd_positions,
            "stats": self._cmd_stats,
            "bias": self._cmd_bias,
            "config": self._cmd_config,
            "swing": self._cmd_swing,
        }
        handler = handlers.get(cmd)
        if handler:
            await handler()

    async def _cmd_help(self):
        lines = ["📋 <b>FUTU Bot Commands</b>\n"]
        for c in COMMANDS:
            lines.append(f"/{c['command']} — {c['description']}")
        await send_message("\n".join(lines))

    async def _cmd_status(self):
        bot = self.bot
        open_pos = sum(1 for s in bot.states.values()
                       if s.has_position or s.has_trending_position)
        status = "🟢 TRADING" if bot.running else "🔴 PAUSED"
        text = (
            f"<b>{status}</b>\n"
            f"💰 Balance: <code>${bot.config.risk.account_balance:.2f}</code>\n"
            f"📊 Symbols: {len(bot.symbols)}\n"
            f"📈 Open positions: {open_pos}/{bot.config.risk.max_positions}\n"
            f"📅 Today trades: {bot.risk.daily.trade_count}\n"
            f"💵 Today PnL: <code>${bot.risk.daily.pnl:+.2f}</code>"
        )
        await send_message(text)

    async def _cmd_on(self):
        self.bot.running = True
        await send_message("🟢 <b>Trading ENABLED</b>")

    async def _cmd_off(self):
        self.bot.running = False
        await send_message("🔴 <b>Trading PAUSED</b>\nPositions still monitored.")

    async def _cmd_positions(self):
        open_syms = [s for s, st in self.bot.states.items()
                     if st.has_position or st.has_trending_position]
        if not open_syms:
            await send_message("📭 No open positions")
            return

        lines = ["📈 <b>Open Positions</b>\n"]
        for sym in open_syms:
            state = self.bot.states[sym]
            orig = self.bot.exchange.config.symbol
            self.bot.exchange.config.symbol = sym
            try:
                pos = await self.bot.exchange.get_position()
                if pos:
                    emoji = "🟢" if pos["side"] == "long" else "🔴"
                    lines.append(
                        f"{emoji} <b>{sym.split('/')[0]}</b> {pos['side'].upper()}\n"
                        f"   Entry: <code>{pos['entry_price']:.2f}</code>\n"
                        f"   Size: <code>{pos['size']}</code>\n"
                        f"   uPnL: <code>${pos['unrealized_pnl']:+.2f}</code>\n"
                        f"   Candles: {state.position_candle_count}"
                    )
            except Exception as e:
                lines.append(f"⚠️ {sym.split('/')[0]}: error")
            finally:
                self.bot.exchange.config.symbol = orig
        await send_message("\n".join(lines))

    async def _cmd_stats(self):
        d = self.bot.risk.daily
        wr = (d.wins / d.trade_count * 100) if d.trade_count > 0 else 0
        text = (
            "📊 <b>Today's Stats</b>\n"
            f"📈 Trades: {d.trade_count} | W/L: {d.wins}/{d.losses}\n"
            f"🎯 Win Rate: <code>{wr:.1f}%</code>\n"
            f"💵 PnL: <code>${d.pnl:+.2f}</code>\n"
            f"📉 Total Loss: <code>${d.total_loss:.2f}</code>\n"
            f"💰 Balance: <code>${self.bot.config.risk.account_balance:.2f}</code>\n"
            f"❄️ Cooldown: {self.bot.risk.cooldown_remaining} candles"
        )
        await send_message(text)

    async def _cmd_bias(self):
        lines = ["🔄 <b>Bias (H1/H4)</b>\n"]
        all_syms = set(self.bot.symbols) | set(self.bot.config.risk.ranging_symbols)
        for sym in sorted(all_syms):
            state = self.bot.states.get(sym)
            if state:
                h1 = state.bias_h1.value
                h4 = state.bias_h4.value
                e1 = {"bullish": "🟢", "bearish": "🔴"}.get(h1, "⚪")
                e4 = {"bullish": "🟢", "bearish": "🔴"}.get(h4, "⚪")
                lines.append(f"  {sym.split('/')[0]}: {e1}{h1} / {e4}{h4}")
        await send_message("\n".join(lines))

    async def _cmd_config(self):
        cfg = self.bot.config
        text = (
            "⚙️ <b>Config</b>\n"
            f"Leverage: {cfg.exchange.leverage}x\n"
            f"Risk/trade: {cfg.risk.risk_per_trade_main*100:.0f}% / {cfg.risk.risk_per_trade_scalp*100:.0f}%\n"
            f"Max daily loss: {cfg.risk.max_daily_loss_pct*100:.0f}%\n"
            f"Max positions: {cfg.risk.max_positions}\n"
            f"Top symbols: {cfg.risk.max_symbols}\n"
            f"Cooldown: {cfg.risk.cooldown_candles} candles\n"
            f"Min R:R: {cfg.risk.min_rr_trending}/{cfg.risk.min_rr_ranging}\n"
            f"Main TF: {cfg.timeframe.main_tf} | HTF: {cfg.timeframe.htf}"
        )
        await send_message(text)

    async def _cmd_swing(self):
        await send_message("🔍 <b>Running swing scan...</b>")
        try:
            await self.bot.run_swing_scan_manual()
        except Exception as e:
            await send_message(f"⚠️ Swing scan error: {e}")

    async def _cmd_chart(self, args: list[str]):
        from src.chart import generate_chart
        from src.indicators import compute_all

        # Find symbol
        query = args[0].upper() if args else ""
        symbol = None
        for sym in self.bot.symbols:
            if query and query in sym.split("/")[0]:
                symbol = sym
                break
        if symbol is None:
            symbol = self.bot.symbols[0] if self.bot.symbols else None
        if symbol is None:
            await send_message("⚠️ No symbols available")
            return

        await send_message(f"📊 Generating chart for {symbol.split('/')[0]}...")

        try:
            candles = await self.bot.exchange.fetch_candles(
                self.bot.config.timeframe.main_tf, 100, symbol=symbol,
            )
            if len(candles) < 50:
                await send_message("⚠️ Not enough data")
                return

            state = self.bot.states.get(symbol)
            regime_str = ""
            bias_str = state.bias_h1.value if state else ""

            df = compute_all(candles, self.bot.config.indicators)
            last = df.iloc[-1]
            adx = last.get("adx", 0)
            regime_str = "trending" if adx >= self.bot.config.strategy.adx_trending else "ranging"

            chart_bytes = generate_chart(
                candles, self.bot.config.indicators,
                symbol=symbol.split("/")[0] + "/USDT",
                regime=regime_str, bias=bias_str,
            )
            caption = (
                f"📊 <b>{symbol.split('/')[0]}</b> | 15m\n"
                f"💰 ${last['close']:,.1f} | {regime_str} | HTF: {bias_str}\n"
                f"RSI: {last.get('rsi', 0):.0f} | ADX: {adx:.0f}"
            )
            await send_photo(chart_bytes, caption)
        except Exception as e:
            logger.warning("Chart error: %s", e)
            await send_message(f"⚠️ Chart error: {e}")


# ── Notifications ──

async def notify_startup(symbols: list[str], balance: float):
    text = (
        "🟢 <b>FUTU Bot Started</b>\n"
        f"💰 Balance: <code>${balance:.2f}</code>\n"
        f"📊 Symbols: {len(symbols)}\n"
        f"🪙 {', '.join(s.split('/')[0] for s in symbols)}"
    )
    await send_message(text)


async def notify_signal(symbol: str, side: str, entry: float, sl: float,
                        tp: float, amount: float, rr: float, regime: str,
                        chart_bytes: bytes | None = None):
    logger.info("Sending trade notification: %s %s @ %.2f", symbol.split('/')[0], side, entry)
    emoji = "🟢" if side == "buy" else "🔴"
    text = (
        f"{emoji} <b>NEW TRADE</b>\n"
        f"🪙 <b>{symbol.split('/')[0]}</b> | {side.upper()} | {regime}\n"
        f"📍 Entry: <code>{entry:.2f}</code>\n"
        f"🎯 TP: <code>{tp:.2f}</code>\n"
        f"🛡 SL: <code>{sl:.2f}</code>\n"
        f"📦 Size: <code>${amount * entry:.0f}</code>\n"
        f"⚖️ R:R = <code>{rr:.2f}</code>"
    )
    if chart_bytes:
        await send_photo(chart_bytes, text)
    else:
        await send_message(text)


async def notify_close(symbol: str, pnl: float, reason: str, balance: float):
    emoji = "✅" if pnl >= 0 else "❌"
    text = (
        f"{emoji} <b>TRADE CLOSED</b>\n"
        f"🪙 <b>{symbol.split('/')[0]}</b>\n"
        f"💵 PnL: <code>${pnl:+.2f}</code>\n"
        f"📝 {reason}\n"
        f"💰 Balance: <code>${balance:.2f}</code>"
    )
    await send_message(text)


async def notify_daily_summary(wins: int, losses: int, pnl: float,
                                balance: float, trade_count: int):
    wr = (wins / trade_count * 100) if trade_count > 0 else 0
    text = (
        "📊 <b>Daily Summary</b>\n"
        f"📈 Trades: {trade_count} | W/L: {wins}/{losses}\n"
        f"🎯 Win Rate: <code>{wr:.1f}%</code>\n"
        f"💵 PnL: <code>${pnl:+.2f}</code>\n"
        f"💰 Balance: <code>${balance:.2f}</code>"
    )
    await send_message(text)


async def notify_error(error: str):
    text = f"⚠️ <b>ERROR</b>\n<code>{error[:500]}</code>"
    await send_message(text)


async def notify_bias_update(biases: dict[str, str]):
    em = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
    lines = ["🔄 <b>Bias Update (H1 / H4)</b>\n"]
    for sym, bias in biases.items():
        # bias format: "H1:bearish/H4:bullish"
        parts = bias.split("/")
        h1 = parts[0].split(":")[1] if len(parts) >= 1 else "?"
        h4 = parts[1].split(":")[1] if len(parts) >= 2 else "?"
        lines.append(f"  {em.get(h1, '⚪')}{em.get(h4, '⚪')} <b>{sym.split('/')[0]}</b>  {h1} / {h4}")
    await send_message("\n".join(lines))


async def notify_heartbeat(symbols_scanned: int, signals_found: int,
                           open_positions: int, balance: float):
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    text = (
        f"💓 <b>Scan {now}</b>\n"
        f"🔍 Scanned: {symbols_scanned} coins\n"
        f"📡 Signals: {signals_found}\n"
        f"📈 Positions: {open_positions}\n"
        f"💰 Balance: <code>${balance:.2f}</code>"
    )
    await send_message(text)
