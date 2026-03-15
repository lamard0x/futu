import logging
import time
import math
from pathlib import Path
from aiohttp import web

from src.indicators import compute_all

logger = logging.getLogger("futu.dashboard")

STATIC_DIR = Path(__file__).parent.parent / "static"


class Dashboard:
    def __init__(self, bot):
        self.bot = bot
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl = 60

    def setup(self, app: web.Application):
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/dashboard", self._handle_index)
        app.router.add_get("/api/candles", self._handle_candles)
        app.router.add_get("/api/latest", self._handle_latest)
        app.router.add_get("/api/markers", self._handle_markers)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get("/api/positions", self._handle_positions)
        app.router.add_get("/api/account", self._handle_account)
        app.router.add_get("/api/trades", self._handle_trades)
        logger.info("Dashboard routes registered")

    async def _handle_index(self, request: web.Request) -> web.Response:
        html_path = STATIC_DIR / "dashboard.html"
        if not html_path.exists():
            return web.Response(text="Dashboard not found", status=404)
        return web.Response(
            text=html_path.read_text(encoding="utf-8"),
            content_type="text/html",
        )

    async def _handle_candles(self, request: web.Request) -> web.Response:
        symbol = request.query.get("symbol", "")
        tf = request.query.get("tf", self.bot.config.timeframe.main_tf)

        if not symbol and self.bot.symbols:
            symbol = self.bot.symbols[0]
        if not symbol:
            return web.json_response({"error": "no symbol"}, status=400)

        cache_key = f"{symbol}:{tf}"
        now = time.time()
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if now - cached_time < self._cache_ttl:
                return web.json_response(cached_data)

        try:
            candles = await self.bot.exchange.fetch_candles(tf, 200, symbol=symbol)
            if len(candles) < 50:
                return web.json_response({"error": "not enough data"}, status=400)

            df = compute_all(candles, self.bot.config.indicators)
            data = self._serialize(df, symbol)
            self._cache[cache_key] = (now, data)
            return web.json_response(data)
        except Exception as e:
            logger.warning("Dashboard candles error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_latest(self, request: web.Request) -> web.Response:
        """Return only the last 2 candles for realtime update (low overhead)."""
        symbol = request.query.get("symbol", "")
        tf = request.query.get("tf", self.bot.config.timeframe.main_tf)
        if not symbol and self.bot.symbols:
            symbol = self.bot.symbols[0]
        if not symbol:
            return web.json_response({"error": "no symbol"}, status=400)
        try:
            candles = await self.bot.exchange.fetch_candles(tf, 3, symbol=symbol)
            def clean(v):
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    return None
                return round(v, 6) if isinstance(v, float) else v
            result = []
            for c in candles:
                ts = c["timestamp"] // 1000
                result.append({
                    "time": ts,
                    "open": clean(c["open"]),
                    "high": clean(c["high"]),
                    "low": clean(c["low"]),
                    "close": clean(c["close"]),
                    "volume": clean(c["volume"]),
                })
            return web.json_response({"candles": result})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_markers(self, request: web.Request) -> web.Response:
        """Return buy/sell markers from recent trades for chart overlay."""
        symbol = request.query.get("symbol", "")
        if not symbol:
            return web.json_response({"markers": []})
        try:
            ex = self.bot.exchange
            trades = await ex.exchange.fetch_my_trades(symbol, limit=50)
            markers = []
            for t in trades:
                ts = (t.get("timestamp", 0) or 0) // 1000
                side = t.get("side", "buy")
                price = float(t.get("price") or 0)
                amount = float(t.get("amount") or 0)
                pnl = float(t.get("info", {}).get("pnl") or 0)
                markers.append({
                    "time": ts,
                    "position": "belowBar" if side == "buy" else "aboveBar",
                    "color": "#0ecb81" if side == "buy" else "#f6465d",
                    "shape": "arrowUp" if side == "buy" else "arrowDown",
                    "text": f"B {amount}" if side == "buy" else f"S {amount}",
                    "side": side,
                    "price": price,
                    "pnl": pnl,
                })
            markers.sort(key=lambda x: x["time"])
            return web.json_response({"markers": markers})
        except Exception as e:
            logger.warning("Markers error: %s", e)
            return web.json_response({"markers": []})

    async def _handle_status(self, request: web.Request) -> web.Response:
        symbols = []
        for sym in self.bot.symbols:
            state = self.bot.states.get(sym)
            if state:
                symbols.append({
                    "symbol": sym,
                    "short": sym.split("/")[0],
                    "bias": state.bias.value,
                    "has_position": state.has_position,
                    "cooldown": state.cooldown,
                })

        return web.json_response({
            "symbols": symbols,
            "balance": self.bot.config.risk.account_balance,
            "running": self.bot.running,
            "open_positions": sum(1 for s in self.bot.states.values() if s.has_position),
            "max_positions": self.bot.config.risk.max_positions,
            "daily_pnl": self.bot.risk.daily.pnl,
            "daily_trades": self.bot.risk.daily.trade_count,
            "daily_wins": self.bot.risk.daily.wins,
            "daily_losses": self.bot.risk.daily.losses,
            "main_tf": self.bot.config.timeframe.main_tf,
        })

    async def _handle_positions(self, request: web.Request) -> web.Response:
        try:
            ex = self.bot.exchange
            positions = await ex.exchange.fetch_positions()
            result = []
            for p in positions:
                size = float(p.get("contracts", 0))
                if size <= 0:
                    continue
                entry = float(p.get("entryPrice") or 0)
                upnl = float(p.get("unrealizedPnl") or 0)
                notional = float(p.get("notional") or 0)
                liq = float(p.get("liquidationPrice") or 0)
                pnl_pct = (upnl / (entry * size) * 100) if entry * size > 0 else 0
                result.append({
                    "symbol": p["symbol"],
                    "short": p["symbol"].split("/")[0],
                    "side": p["side"],
                    "size": size,
                    "entry": entry,
                    "mark": float(p.get("markPrice") or 0),
                    "upnl": round(upnl, 4),
                    "upnl_pct": round(pnl_pct, 2),
                    "notional": round(notional, 2),
                    "liq": round(liq, 2),
                    "leverage": p.get("leverage", ex.config.leverage),
                })
            return web.json_response({"positions": result})
        except Exception as e:
            logger.warning("Positions error: %s", e)
            return web.json_response({"positions": [], "error": str(e)})

    async def _handle_account(self, request: web.Request) -> web.Response:
        try:
            ex = self.bot.exchange
            balance = await ex.exchange.fetch_balance()
            usdt = balance.get("USDT", {})

            # Get all asset balances
            assets = []
            for currency, info in balance.items():
                if isinstance(info, dict) and float(info.get("total") or 0) > 0:
                    assets.append({
                        "currency": currency,
                        "free": round(float(info.get("free") or 0), 4),
                        "used": round(float(info.get("used") or 0), 4),
                        "total": round(float(info.get("total") or 0), 4),
                    })

            return web.json_response({
                "usdt_free": round(float(usdt.get("free") or 0), 2),
                "usdt_used": round(float(usdt.get("used") or 0), 2),
                "usdt_total": round(float(usdt.get("total") or 0), 2),
                "assets": assets,
                "config_balance": self.bot.config.risk.account_balance,
            })
        except Exception as e:
            logger.warning("Account error: %s", e)
            return web.json_response({"error": str(e)})

    async def _handle_trades(self, request: web.Request) -> web.Response:
        try:
            ex = self.bot.exchange
            all_trades = []
            for sym in self.bot.symbols[:5]:
                try:
                    trades = await ex.exchange.fetch_my_trades(sym, limit=10)
                    for t in trades:
                        all_trades.append({
                            "symbol": t["symbol"],
                            "short": t["symbol"].split("/")[0],
                            "side": t["side"],
                            "amount": float(t.get("amount") or 0),
                            "price": float(t.get("price") or 0),
                            "cost": round(float(t.get("cost") or 0), 2),
                            "fee": round(float(t.get("fee", {}).get("cost") or 0), 6),
                            "time": t.get("datetime", ""),
                            "timestamp": t.get("timestamp", 0),
                            "pnl": float(t.get("info", {}).get("pnl") or 0),
                        })
                except Exception:
                    pass
            all_trades.sort(key=lambda x: x["timestamp"], reverse=True)
            return web.json_response({"trades": all_trades[:50]})
        except Exception as e:
            logger.warning("Trades error: %s", e)
            return web.json_response({"trades": [], "error": str(e)})

    def _serialize(self, df, symbol: str) -> dict:
        def clean(v):
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return None
            return round(v, 6) if isinstance(v, float) else v

        candles = []
        indicators = {
            "ema_fast": [], "ema_mid": [], "ema_slow": [],
            "bb_upper": [], "bb_lower": [], "bb_mid": [],
            "chandelier_long": [], "chandelier_short": [],
            "volume_sma": [],
        }

        cfg = self.bot.config.indicators
        ema_cols = {
            "ema_fast": f"ema_{cfg.ema_fast}",
            "ema_mid": f"ema_{cfg.ema_mid}",
            "ema_slow": f"ema_{cfg.ema_slow}",
        }

        for idx, row in df.iterrows():
            ts = int(idx.timestamp())
            candles.append({
                "time": ts,
                "open": clean(row["open"]),
                "high": clean(row["high"]),
                "low": clean(row["low"]),
                "close": clean(row["close"]),
                "volume": clean(row["volume"]),
            })

            for key, col in ema_cols.items():
                indicators[key].append({"time": ts, "value": clean(row.get(col))})

            for col in ("bb_upper", "bb_lower", "bb_mid", "chandelier_long", "chandelier_short", "volume_sma"):
                indicators[col].append({"time": ts, "value": clean(row.get(col))})

        last = df.iloc[-1]
        adx = clean(last.get("adx", 0))
        regime = "trending" if (adx or 0) >= self.bot.config.strategy.adx_trending else "ranging"
        state = self.bot.states.get(symbol)
        bias = state.bias.value if state else "neutral"

        info = {
            "symbol": symbol,
            "short": symbol.split("/")[0],
            "price": clean(last["close"]),
            "rsi": clean(last.get("rsi", 0)),
            "adx": adx,
            "atr": clean(last.get("atr", 0)),
            "volume": clean(last.get("volume", 0)),
            "volume_sma": clean(last.get("volume_sma", 0)),
            "regime": regime,
            "bias": bias,
            "bb_width": clean(last.get("bb_width", 0)),
            "plus_di": clean(last.get("plus_di", 0)),
            "minus_di": clean(last.get("minus_di", 0)),
        }

        return {"candles": candles, "indicators": indicators, "info": info}
