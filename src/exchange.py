import ccxt.async_support as ccxt
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from src.config import ExchangeConfig
from src.rate_limiter import RateLimiter
from src.error_handler import retry_api_call

logger = logging.getLogger("futu.exchange")


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    price: float
    amount: float
    status: str
    raw: dict


class Exchange:
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.exchange_name = config.exchange_name
        self.rate_limiter = RateLimiter()

    async def connect(self):
        if self.exchange_name == "okx":
            await self._connect_okx()
        else:
            await self._connect_bybit()

        await self.exchange.load_markets()
        await self._setup_leverage()

        mode = self._get_mode()
        logger.info(
            "Connected to %s %s | %s | leverage %dx",
            self.exchange_name.upper(), mode,
            self.config.symbol, self.config.leverage,
        )

    async def _connect_bybit(self):
        self.exchange = ccxt.bybit({
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "options": {"defaultType": "swap"},
        })
        if self.config.testnet == "demo":
            self.exchange.urls["api"] = {
                "public": "https://api-demo.bybit.com",
                "private": "https://api-demo.bybit.com",
            }
        elif self.config.testnet == "true":
            self.exchange.set_sandbox_mode(True)

    async def _connect_okx(self):
        opts = {
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "password": self.config.passphrase,
            "options": {"defaultType": "swap"},
        }
        self.exchange = ccxt.okx(opts)
        if self.config.testnet in ("demo", "true"):
            self.exchange.set_sandbox_mode(True)

    def _get_mode(self) -> str:
        if self.config.testnet == "demo":
            return "demo"
        if self.config.testnet == "true":
            return "testnet"
        return "live"

    async def _setup_leverage(self):
        # OKX: set account mode to single-currency margin (required for futures)
        if self.exchange_name == "okx":
            try:
                await self.exchange.private_post_account_set_account_level(
                    {"acctLv": "2"}  # 2 = single-currency margin
                )
                logger.info("OKX account mode set to single-currency margin")
            except Exception as e:
                err = str(e).lower()
                if "already" not in err and "no attribute" not in err:
                    logger.warning("Set OKX account mode: %s", e)

        try:
            await self.exchange.set_leverage(
                self.config.leverage, self.config.symbol,
            )
        except ccxt.ExchangeError as e:
            if "not modified" not in str(e).lower():
                logger.warning("Set leverage warning: %s", e)

        try:
            await self.exchange.set_margin_mode(
                self.config.margin_mode, self.config.symbol,
            )
        except ccxt.ExchangeError as e:
            if "not modified" not in str(e).lower():
                logger.warning("Set margin mode warning: %s", e)

    async def fetch_candles(self, timeframe: str, limit: int = 200, symbol: str | None = None) -> list[dict]:
        sym = symbol or self.config.symbol
        await self.rate_limiter.acquire("fetch_ohlcv")
        ohlcv = await retry_api_call(self.exchange.fetch_ohlcv, sym, timeframe, limit=limit)
        return [
            {
                "timestamp": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in ohlcv
        ]

    async def get_balance(self) -> float:
        await self.rate_limiter.acquire("fetch_balance")
        balance = await retry_api_call(self.exchange.fetch_balance)
        return float(balance.get("USDT", {}).get("free", 0))

    async def get_position(self) -> Optional[dict]:
        await self.rate_limiter.acquire("fetch_positions")
        positions = await retry_api_call(self.exchange.fetch_positions, [self.config.symbol])
        for pos in positions:
            size = float(pos.get("contracts") or 0)
            if size > 0:
                return {
                    "side": pos["side"],
                    "size": size,
                    "entry_price": float(pos.get("entryPrice") or 0),
                    "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
                    "notional": float(pos.get("notional") or 0),
                    "liquidation_price": float(pos.get("liquidationPrice") or 0),
                }
        return None

    async def place_limit_order(
        self, side: str, amount: float, price: float,
        tp_price: Optional[float] = None, sl_price: Optional[float] = None,
    ) -> OrderResult:
        params = {}
        if self.exchange_name == "okx":
            params["tdMode"] = self.config.margin_mode

        # Convert coin amount to contracts for OKX
        order_amount = amount
        if self.exchange_name == "okx" and self.config.symbol in self.exchange.markets:
            market = self.exchange.markets[self.config.symbol]
            ct_val = float(market.get("contractSize") or market.get("info", {}).get("ctVal") or 1)
            if ct_val > 0 and ct_val != 1:
                order_amount = int(amount / ct_val)
            else:
                order_amount = int(amount)
            if order_amount <= 0:
                order_amount = 1

        await self.rate_limiter.acquire("create_order")
        order = await retry_api_call(
            self.exchange.create_order,
            symbol=self.config.symbol, type="limit",
            side=side, amount=order_amount, price=price, params=params,
        )
        logger.info("Limit order: %s %s %s @ %g | status=%s id=%s",
                    side, self.config.symbol, order_amount, price,
                    order.get("status"), order.get("id"))
        return OrderResult(
            order_id=order["id"], symbol=order["symbol"],
            side=side, price=price, amount=order_amount,
            status=order["status"], raw=order,
        )

    async def place_market_order(
        self, side: str, amount: float,
        tp_price: Optional[float] = None, sl_price: Optional[float] = None,
    ) -> OrderResult:
        # Step 1: Place market order (no TP/SL — OKX rejects inline TP/SL)
        params = {}
        if self.exchange_name == "okx":
            params["tdMode"] = self.config.margin_mode

        # Convert coin amount to contract count for OKX
        order_amount = amount
        if self.exchange_name == "okx" and self.config.symbol in self.exchange.markets:
            market = self.exchange.markets[self.config.symbol]
            ct_val = float(market.get("contractSize") or market.get("info", {}).get("ctVal") or 1)
            if ct_val > 0 and ct_val != 1:
                order_amount = int(amount / ct_val)
            else:
                order_amount = int(amount)
            if order_amount <= 0:
                order_amount = 1
            logger.info("Amount: %.6f coins -> %d contracts (ctVal=%s)", amount, order_amount, ct_val)

        await self.rate_limiter.acquire("create_order")
        order = await retry_api_call(
            self.exchange.create_order,
            symbol=self.config.symbol, type="market",
            side=side, amount=order_amount, params=params,
        )
        logger.info("Market order: %s %s %s | status=%s id=%s",
                    side, self.config.symbol, order_amount, order.get("status"), order.get("id"))

        # Step 2: Set TP/SL separately after order fills
        if tp_price is not None or sl_price is not None:
            try:
                await asyncio.sleep(0.5)  # wait for fill
                await self.update_tp_sl(tp_price=tp_price, sl_price=sl_price)
            except Exception as e:
                logger.warning("Set TP/SL after order error: %s", e)

        return OrderResult(
            order_id=order["id"], symbol=order["symbol"],
            side=side, price=float(order.get("average") or order.get("price") or 0),
            amount=amount, status=order["status"], raw=order,
        )

    async def update_tp_sl(
        self, tp_price: Optional[float] = None, sl_price: Optional[float] = None,
    ):
        if self.exchange_name == "bybit":
            await self._update_tp_sl_bybit(tp_price, sl_price)
        else:
            await self._update_tp_sl_generic(tp_price, sl_price)

    async def _update_tp_sl_bybit(self, tp_price, sl_price):
        params = {"category": "linear", "positionIdx": 0}
        raw_symbol = self.config.symbol.replace("/", "").replace(":USDT", "")
        params["symbol"] = raw_symbol
        if tp_price is not None:
            params["takeProfit"] = str(tp_price)
        if sl_price is not None:
            params["stopLoss"] = str(sl_price)
        try:
            await self.exchange.private_post_v5_position_trading_stop(params)
            logger.info("TP/SL updated: TP=%.2f SL=%.2f", tp_price or 0, sl_price or 0)
        except ccxt.ExchangeError as e:
            logger.warning("Update TP/SL error: %s", e)

    async def _update_tp_sl_generic(self, tp_price, sl_price):
        """Update TP/SL on OKX using algo order API (conditional orders)."""
        try:
            position = await self.get_position()
            if position is None:
                return

            inst_id = self.config.symbol.replace("/", "-").replace(":USDT", "-SWAP")
            close_side = "sell" if position["side"] == "long" else "buy"
            size = str(int(position["size"]))

            # Cancel existing algo orders first
            await self._cancel_algo_orders()

            # Place SL as conditional algo order
            if sl_price is not None:
                await self.rate_limiter.acquire("create_order")
                await self.exchange.private_post_trade_order_algo({
                    "instId": inst_id,
                    "tdMode": self.config.margin_mode,
                    "side": close_side,
                    "ordType": "conditional",
                    "sz": size,
                    "slTriggerPx": str(sl_price),
                    "slOrdPx": "-1",  # market price
                    "reduceOnly": "true",
                })

            # Place TP as conditional algo order
            if tp_price is not None:
                await self.rate_limiter.acquire("create_order")
                await self.exchange.private_post_trade_order_algo({
                    "instId": inst_id,
                    "tdMode": self.config.margin_mode,
                    "side": close_side,
                    "ordType": "conditional",
                    "sz": size,
                    "tpTriggerPx": str(tp_price),
                    "tpOrdPx": "-1",  # market price
                    "reduceOnly": "true",
                })

            logger.info("TP/SL set: TP=%s SL=%s", tp_price or "-", sl_price or "-")
        except Exception as e:
            logger.warning("Update TP/SL error: %s", e)

    async def _cancel_algo_orders(self):
        """Cancel ALL algo (TP/SL/trigger) orders for current symbol on OKX."""
        inst_id = self.config.symbol.replace("/", "-").replace(":USDT", "-SWAP")
        cancelled = 0

        # Get all pending algo orders for this symbol
        try:
            await self.rate_limiter.acquire("fetch_open_orders")
            result = await self.exchange.private_get_trade_orders_algo_pending({
                "instId": inst_id,
                "ordType": "conditional",
            })
            orders = result.get("data", [])
            for o in orders:
                algo_id = o.get("algoId")
                if algo_id:
                    try:
                        await self.rate_limiter.acquire("cancel_order")
                        await self.exchange.private_post_trade_cancel_algos([{
                            "algoId": algo_id,
                            "instId": inst_id,
                        }])
                        cancelled += 1
                    except Exception as e:
                        logger.warning("Cancel algo %s failed: %s", algo_id, e)
        except Exception as e:
            logger.warning("Fetch algo orders failed: %s", e)

        if cancelled > 0:
            logger.info("Cancelled %d algo orders for %s", cancelled, self.config.symbol.split("/")[0])

    async def cancel_all_orders(self):
        try:
            await self.exchange.cancel_all_orders(self.config.symbol)
            logger.info("All orders cancelled")
        except ccxt.ExchangeError as e:
            logger.warning("Cancel orders error: %s", e)

    async def close_position(self):
        position = await self.get_position()
        if position is None:
            return
        # Cancel any TP/SL algo orders first to avoid orphaned orders
        if self.exchange_name == "okx":
            await self._cancel_algo_orders()
        else:
            try:
                await self.cancel_all_orders()
            except Exception:
                pass
        close_side = "sell" if position["side"] == "long" else "buy"
        await self.place_market_order(close_side, position["size"])
        logger.info("Position closed")

    async def get_closed_pnl(self, symbol: str | None = None) -> float:
        sym = symbol or self.config.symbol
        try:
            if self.exchange_name == "bybit":
                raw = sym.replace("/", "").replace(":USDT", "")
                result = await self.exchange.private_get_v5_position_closed_pnl({
                    "category": "linear", "symbol": raw, "limit": 1,
                })
                items = result.get("result", {}).get("list", [])
                if items:
                    return float(items[0].get("closedPnl", 0))
            else:
                # OKX: fetch recent trades, find the closing trade with PnL
                trades = await self.exchange.fetch_my_trades(sym, limit=10)
                for t in reversed(trades):
                    info = t.get("info", {})
                    pnl = float(info.get("fillPnl") or info.get("pnl") or 0)
                    if pnl != 0:
                        logger.info("Closed PnL for %s: $%.4f", sym.split("/")[0], pnl)
                        return pnl
        except Exception as e:
            logger.warning("Get closed PnL error: %s", e)
        return 0.0

    async def fetch_funding_rate(self, symbol: str) -> dict:
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "rate": float(funding.get("fundingRate", 0)),
                "next_time": funding.get("fundingDatetime", ""),
                "next_timestamp": funding.get("fundingTimestamp", 0),
            }
        except Exception as e:
            logger.warning("Funding rate error %s: %s", symbol, e)
            return {"symbol": symbol, "rate": 0, "next_time": "", "next_timestamp": 0}

    async def fetch_all_funding_rates(self, limit: int = 20) -> list[dict]:
        try:
            tickers = await self.exchange.fetch_tickers()
            symbols = [
                s for s in tickers
                if s.endswith(":USDT") and "/USDT" in s
            ]
            rates = []
            for sym in symbols[:limit]:
                r = await self.fetch_funding_rate(sym)
                if abs(r["rate"]) > 0:
                    rates.append(r)
                await asyncio.sleep(0.1)
            rates.sort(key=lambda x: abs(x["rate"]), reverse=True)
            return rates
        except Exception as e:
            logger.warning("Fetch all funding rates error: %s", e)
            return []

    async def get_ticker(self) -> dict:
        await self.rate_limiter.acquire("fetch_ticker")
        ticker = await retry_api_call(self.exchange.fetch_ticker, self.config.symbol)
        return {
            "bid": float(ticker.get("bid", 0)),
            "ask": float(ticker.get("ask", 0)),
            "last": float(ticker.get("last", 0)),
            "volume_24h": float(ticker.get("quoteVolume", 0)),
        }

    async def get_top_volume_symbols(self, limit: int = 10) -> list[str]:
        # Use live public API for accurate volume data
        try:
            live_ex = ccxt.okx({"options": {"defaultType": "swap"}}) if self.exchange_name == "okx" else ccxt.bybit({"options": {"defaultType": "swap"}})
            try:
                await self.rate_limiter.acquire("fetch_tickers")
                tickers = await retry_api_call(live_ex.fetch_tickers)
            finally:
                await live_ex.close()
        except Exception:
            await self.rate_limiter.acquire("fetch_tickers")
            tickers = await retry_api_call(self.exchange.fetch_tickers)

        perp_tickers = []
        for symbol, t in tickers.items():
            if not (symbol.endswith(":USDT") and "/USDT" in symbol):
                continue
            qv = float(t.get("quoteVolume") or 0)
            if qv == 0:
                bv = float(t.get("baseVolume") or 0)
                last = float(t.get("last") or 0)
                qv = bv * last
            perp_tickers.append((symbol, qv))
        perp_tickers.sort(key=lambda x: x[1], reverse=True)
        # Only include symbols that exist in our exchange's markets
        available = set(self.exchange.markets.keys()) if self.exchange.markets else set()
        top = []
        for symbol, vol in perp_tickers:
            if available and symbol not in available:
                continue
            top.append(symbol)
            if len(top) >= limit:
                break
        logger.info("Top %d volume: %s", limit, [s.split("/")[0] for s in top])
        return top

    async def setup_symbol(self, symbol: str):
        try:
            await self.exchange.set_leverage(self.config.leverage, symbol)
        except Exception:
            pass
        try:
            await self.exchange.set_margin_mode(self.config.margin_mode, symbol)
        except Exception:
            pass

    async def disconnect(self):
        if self.exchange:
            await self.exchange.close()
            logger.info("Disconnected from %s", self.exchange_name.upper())
