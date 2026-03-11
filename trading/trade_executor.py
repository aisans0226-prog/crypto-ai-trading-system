"""
trading/trade_executor.py — Auto trade execution via Binance & Bybit Futures APIs.

Flow:
  detect signal → validate risk → place market order → set SL (stop-market) → set TP (take-profit-market)

Supports:
  - Binance USDM Futures
  - Bybit Linear Perpetual V5
"""
import asyncio
import hashlib
import hmac
import time
from typing import Dict, Optional
import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from trading.risk_manager import RiskParameters


# ─────────────────────────────────────────────────────────────────────────────
# Binance Futures Executor
# ─────────────────────────────────────────────────────────────────────────────
class BinanceExecutor:
    BASE = "https://fapi.binance.com"
    BASE_TEST = "https://testnet.binancefuture.com"

    def __init__(self) -> None:
        self._key = settings.binance_api_key
        self._secret = settings.binance_api_secret
        self._base = self.BASE_TEST if settings.binance_testnet else self.BASE
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    def _sign(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(
            self._secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

    async def _post(self, path: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        async with self._session.post(url, headers=headers, params=params) as r:
            data = await r.json()
            if r.status != 200:
                logger.error("Binance API error {}: {}", r.status, data)
            return data

    # ── Set leverage ──────────────────────────────────────────────────────
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._post("/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage,
        })

    # ── Place market order ────────────────────────────────────────────────
    async def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> dict:
        logger.info("Binance place {} {} qty={}", symbol, side, quantity)
        return await self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        })

    # ── Stop-loss order ───────────────────────────────────────────────────
    async def place_stop_loss(
        self, symbol: str, side: str, quantity: float, stop_price: float
    ) -> dict:
        close_side = "SELL" if side == "BUY" else "BUY"
        return await self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": close_side,
            "type": "STOP_MARKET",
            "stopPrice": round(stop_price, 6),
            "quantity": quantity,
            "reduceOnly": "true",
        })

    # ── Take-profit order ─────────────────────────────────────────────────
    async def place_take_profit(
        self, symbol: str, side: str, quantity: float, tp_price: float
    ) -> dict:
        close_side = "SELL" if side == "BUY" else "BUY"
        return await self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": round(tp_price, 6),
            "quantity": quantity,
            "reduceOnly": "true",
        })

    # ── Full trade execution ──────────────────────────────────────────────
    async def execute_trade(self, params: RiskParameters) -> Dict:
        side = "BUY" if params.direction == "LONG" else "SELL"
        result = {"exchange": "binance", "symbol": params.symbol}

        try:
            await self.set_leverage(params.symbol, params.leverage)

            order = await self.place_market_order(
                params.symbol, side, params.quantity
            )
            result["order_id"] = order.get("orderId")
            result["entry_order"] = order

            sl_order = await self.place_stop_loss(
                params.symbol, side, params.quantity, params.stop_loss
            )
            result["sl_order_id"] = sl_order.get("orderId")

            tp_order = await self.place_take_profit(
                params.symbol, side, params.quantity, params.take_profit
            )
            result["tp_order_id"] = tp_order.get("orderId")

            result["status"] = "open"
            logger.info("Trade opened: {}", result)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("Trade execution failed ({}): {}", params.symbol, exc)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Bybit Futures Executor
# ─────────────────────────────────────────────────────────────────────────────
class BybitExecutor:
    BASE = "https://api.bybit.com"
    BASE_TEST = "https://api-testnet.bybit.com"

    def __init__(self) -> None:
        self._key = settings.bybit_api_key
        self._secret = settings.bybit_api_secret
        self._base = self.BASE_TEST if settings.bybit_testnet else self.BASE
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    def _gen_signature(self, timestamp: str, params_str: str) -> str:
        recv_window = "5000"
        sign_str = timestamp + self._key + recv_window + params_str
        return hmac.new(
            self._secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()

    async def _post(self, path: str, body: dict) -> dict:
        import json
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        body_str = json.dumps(body)
        sig = self._gen_signature(ts, body_str)
        headers = {
            "X-BAPI-API-KEY": self._key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": sig,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }
        url = self._base + path
        async with self._session.post(url, headers=headers, data=body_str) as r:
            data = await r.json()
            if data.get("retCode") != 0:
                logger.error("Bybit API error: {}", data)
            return data

    async def execute_trade(self, params: RiskParameters) -> Dict:
        side = "Buy" if params.direction == "LONG" else "Sell"
        result = {"exchange": "bybit", "symbol": params.symbol}

        try:
            body = {
                "category": "linear",
                "symbol": params.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(params.quantity),
                "leverage": str(params.leverage),
                "stopLoss": str(params.stop_loss),
                "takeProfit": str(params.take_profit),
                "slTriggerBy": "LastPrice",
                "tpTriggerBy": "LastPrice",
                "timeInForce": "GoodTillCancel",
            }
            resp = await self._post("/v5/order/create", body)
            result["order_id"] = resp.get("result", {}).get("orderId")
            result["status"] = "open"
            logger.info("Bybit trade opened: {}", result)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("Bybit trade execution failed ({}): {}", params.symbol, exc)

        return result
