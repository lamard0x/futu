import ccxt.async_support as ccxt
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from src.config import ExchangeConfig

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


class BybitExchange:
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange: Optional[ccxt.bybit] = None

    async def connect(self):
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
        elif self.config.testnet:
            self.exchange.set_sandbox_mode(True)
        await self.exchange.load_markets()
        await self._setup_leverage()
        mode = "demo" if self.config.testnet == "demo" else ("testnet" if self.config.testnet else "live")
        logger.info(
            "Connected to Bybit %s | %s | leverage %dx",
            mode,
            self.config.symbol,
            self.config.leverage,
        )

    async def _setup_leverage(self):
        try:
            await self.exchange.set_leverage(
                self.config.leverage,
                self.config.symbol,
            )
        except ccxt.ExchangeError as e:
            if "leverage not modified" not in str(e).lower():
                logger.warning("Set leverage warning: %s", e)

        try:
            await self.exchange.set_margin_mode(
                self.config.margin_mode,
                self.config.symbol,
            )
        except ccxt.ExchangeError as e:
            if "not modified" not in str(e).lower():
                logger.warning("Set margin mode warning: %s", e)

    async def fetch_candles(self, timeframe: str, limit: int = 200, symbol: str | None = None) -> list[dict]:
        sym = symbol or self.config.symbol
        ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe, limit=limit)
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
        balance = await self.exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0))

    async def get_position(self) -> Optional[dict]:
        positions = await self.exchange.fetch_positions([self.config.symbol])
        for pos in positions:
            size = float(pos.get("contracts", 0))
            if size > 0:
                return {
                    "side": pos["side"],
                    "size": size,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "notional": float(pos.get("notional", 0)),
                    "liquidation_price": float(pos.get("liquidationPrice", 0)),
                }
        return None

    async def place_limit_order(
        self,
        side: str,
        amount: float,
        price: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> OrderResult:
        params = {"timeInForce": "PostOnly"}
        if tp_price is not None:
            params["takeProfit"] = tp_price
        if sl_price is not None:
            params["stopLoss"] = sl_price

        order = await self.exchange.create_order(
            symbol=self.config.symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params=params,
        )
        logger.info(
            "Order placed: %s %s %.4f @ %.2f | TP=%.2f SL=%.2f",
            side, self.config.symbol, amount, price,
            tp_price or 0, sl_price or 0,
        )
        return OrderResult(
            order_id=order["id"],
            symbol=order["symbol"],
            side=side,
            price=price,
            amount=amount,
            status=order["status"],
            raw=order,
        )

    async def place_market_order(
        self,
        side: str,
        amount: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> OrderResult:
        params = {}
        if tp_price is not None:
            params["takeProfit"] = tp_price
        if sl_price is not None:
            params["stopLoss"] = sl_price

        order = await self.exchange.create_order(
            symbol=self.config.symbol,
            type="market",
            side=side,
            amount=amount,
            params=params,
        )
        logger.info("Market order: %s %s %.4f", side, self.config.symbol, amount)
        return OrderResult(
            order_id=order["id"],
            symbol=order["symbol"],
            side=side,
            price=float(order.get("average", 0)),
            amount=amount,
            status=order["status"],
            raw=order,
        )

    async def set_trailing_stop(self, trailing_distance: float):
        try:
            position = await self.get_position()
            if position is None:
                return
            await self.exchange.set_position_mode(False, self.config.symbol)
            side = "Buy" if position["side"] == "long" else "Sell"
            await self.exchange.private_post_v5_position_trading_stop({
                "category": "linear",
                "symbol": self.config.symbol.replace("/", "").replace(":USDT", ""),
                "trailingStop": str(trailing_distance),
                "positionIdx": 0,
            })
            logger.info("Trailing stop set: %.2f", trailing_distance)
        except ccxt.ExchangeError as e:
            logger.warning("Trailing stop error: %s", e)

    async def update_tp_sl(
        self,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ):
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
        close_side = "sell" if position["side"] == "long" else "buy"
        await self.place_market_order(close_side, position["size"])
        logger.info("Position closed")

    async def get_ticker(self) -> dict:
        ticker = await self.exchange.fetch_ticker(self.config.symbol)
        return {
            "bid": float(ticker.get("bid", 0)),
            "ask": float(ticker.get("ask", 0)),
            "last": float(ticker.get("last", 0)),
            "volume_24h": float(ticker.get("quoteVolume", 0)),
        }

    async def get_top_volume_symbols(self, limit: int = 10) -> list[str]:
        tickers = await self.exchange.fetch_tickers()
        perp_tickers = [
            (symbol, float(t.get("quoteVolume", 0)))
            for symbol, t in tickers.items()
            if symbol.endswith(":USDT") and "/USDT" in symbol
        ]
        perp_tickers.sort(key=lambda x: x[1], reverse=True)
        top = [symbol for symbol, vol in perp_tickers[:limit]]
        logger.info("Top %d volume: %s", limit, [s.split("/")[0] for s in top])
        return top

    async def setup_symbol(self, symbol: str):
        try:
            await self.exchange.set_leverage(self.config.leverage, symbol)
        except ccxt.ExchangeError:
            pass
        try:
            await self.exchange.set_margin_mode(self.config.margin_mode, symbol)
        except ccxt.ExchangeError:
            pass

    async def disconnect(self):
        if self.exchange:
            await self.exchange.close()
            logger.info("Disconnected from Bybit")
