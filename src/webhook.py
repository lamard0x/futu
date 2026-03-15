import logging
import time
from collections import defaultdict
from aiohttp import web

from src.config import Config
from src.strategy import Signal, SignalType, SignalSource, Regime
from src.dashboard import Dashboard

logger = logging.getLogger("futu.webhook")


class WebhookServer:
    def __init__(self, config: Config, bot):
        self.config = config
        self.bot = bot
        self.runner = None
        self.site = None
        self._rate_limits: dict[str, list[float]] = defaultdict(list)

    async def start(self):
        app = web.Application()
        app.router.add_get("/health", self._handle_health)

        # Dashboard always enabled
        dashboard = Dashboard(self.bot)
        dashboard.setup(app)

        if self.config.webhook.enabled:
            app.router.add_post("/webhook", self._handle_webhook)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        port = self.config.webhook.port or 8888
        try:
            self.site = web.TCPSite(self.runner, "localhost", port)
            await self.site.start()
            logger.info("Dashboard on http://localhost:%d", port)
        except Exception as e:
            logger.error("Dashboard start failed: %s", e)

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "running": self.bot.running,
            "positions": sum(1 for s in self.bot.states.values() if s.has_position),
        })

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        ip = request.remote or "unknown"
        if not self._check_rate_limit(ip):
            return web.json_response({"error": "rate limited"}, status=429)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        secret = data.get("secret", "") or request.headers.get("X-Webhook-Secret", "")
        if self.config.webhook.secret and secret != self.config.webhook.secret:
            return web.json_response({"error": "unauthorized"}, status=401)

        error = self._validate(data)
        if error:
            return web.json_response({"error": error}, status=400)

        try:
            signal, symbol = self._parse_signal(data)
            result = await self._execute(signal, symbol)
            return web.json_response(result)
        except Exception as e:
            logger.error("Webhook execution error: %s", e, exc_info=True)
            return web.json_response({"error": "execution failed"}, status=500)

    def _validate(self, data: dict) -> str | None:
        action = data.get("action", "")
        if action not in ("buy", "sell"):
            return "action must be 'buy' or 'sell'"

        for field in ("symbol", "price", "sl", "tp"):
            if field not in data:
                return f"missing field: {field}"

        for field in ("price", "sl", "tp"):
            try:
                val = float(data[field])
                if val <= 0:
                    return f"{field} must be positive"
            except (ValueError, TypeError):
                return f"{field} must be a number"

        return None

    def _parse_signal(self, data: dict) -> tuple[Signal, str]:
        raw_symbol = data["symbol"].upper().strip()
        if "/" not in raw_symbol:
            symbol = f"{raw_symbol.replace('USDT', '')}/USDT:USDT"
        else:
            symbol = raw_symbol

        price = float(data["price"])
        sl = float(data["sl"])
        tp = float(data["tp"])
        action = data["action"]

        signal = Signal(
            type=SignalType.LONG if action == "buy" else SignalType.SHORT,
            source=SignalSource.ALERT,
            regime=Regime.TRENDING,
            entry_price=price,
            sl_price=sl,
            tp1_price=tp,
            tp2_price=0.0,
            reason=f"[TV] {action.upper()} {raw_symbol}",
        )
        return signal, symbol

    async def _execute(self, signal: Signal, symbol: str) -> dict:
        can_trade, reason = self.bot.risk.can_trade_new()
        if not can_trade:
            return {"status": "rejected", "reason": reason}

        rr_ok, rr = self.bot.risk.check_rr(signal)
        if not rr_ok:
            return {"status": "rejected", "reason": f"R:R too low: {rr:.2f}"}

        open_pos = sum(1 for s in self.bot.states.values() if s.has_position)
        if open_pos >= self.bot.config.risk.max_positions:
            return {"status": "rejected", "reason": "max positions reached"}

        await self.bot._execute_signal(signal, symbol)
        return {"status": "executed", "symbol": symbol, "rr": round(rr, 2)}

    def _check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        window = [t for t in self._rate_limits[ip] if now - t < 60]
        self._rate_limits[ip] = window
        if len(window) >= 10:
            return False
        self._rate_limits[ip].append(now)
        return True
