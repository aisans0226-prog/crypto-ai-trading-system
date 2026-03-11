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

    async def _get(self, path: str, params: dict) -> any:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        async with self._session.get(url, headers=headers, params=params) as r:
            data = await r.json()
            if r.status != 200:
                logger.error("Binance API GET error {}: {}", r.status, data)
                raise Exception(f"Binance API error {r.status}: {data.get('msg', 'unknown')}")
            return data

    async def _delete(self, path: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        async with self._session.delete(url, headers=headers, params=params) as r:
            data = await r.json()
            if r.status != 200:
                logger.error("Binance API DELETE error {}: {}", r.status, data)
                raise Exception(f"Binance API error {r.status}: {data.get('msg', 'unknown')}")
            return data

    async def _post(self, path: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        async with self._session.post(url, headers=headers, params=params) as r:
            data = await r.json()
            if r.status != 200:
                logger.error("Binance API error {}: {}", r.status, data)
                raise Exception(f"Binance API error {r.status}: {data.get('msg', 'unknown')}")
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

    async def get_open_positions(self) -> list:
        """Returns list of currently open positions from Binance Futures."""
        try:
            data = await self._get("/fapi/v2/positionRisk", {})
            return [p for p in data if float(p.get("positionAmt", 0)) != 0]
        except Exception as exc:
            logger.debug("get_open_positions Binance error: {}", exc)
            return []

    async def get_account_balance(self) -> float:
        """Get available USDT balance from Binance Futures."""
        try:
            assets = await self._get("/fapi/v2/balance", {})
            for asset in assets:
                if asset.get("asset") == "USDT":
                    return float(asset.get("availableBalance", 0))
        except Exception as exc:
            logger.debug("get_account_balance Binance error: {}", exc)
        return 0.0

    async def get_recent_pnl(self, symbol: str) -> float:
        """Get total realized PnL for a symbol in the last 24 hours."""
        try:
            start_time = int((time.time() - 86400) * 1000)
            data = await self._get("/fapi/v1/income", {
                "symbol": symbol,
                "incomeType": "REALIZED_PNL",
                "startTime": start_time,
                "limit": 50,
            })
            return sum(float(r.get("income", 0)) for r in data)
        except Exception as exc:
            logger.debug("get_recent_pnl Binance error {}: {}", symbol, exc)
        return 0.0

    async def cancel_order(self, symbol: str, order_id) -> dict:
        """Cancel a specific open order by ID."""
        return await self._delete("/fapi/v1/order", {
            "symbol": symbol,
            "orderId": int(order_id),
        })

    async def close_position_market(
        self, symbol: str, direction: str, quantity: float
    ) -> dict:
        """Force-close an open position with a market order (reduceOnly)."""
        close_side = "SELL" if direction == "LONG" else "BUY"
        logger.info("Binance force-close {} {} qty={}", symbol, close_side, quantity)
        return await self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": quantity,
            "reduceOnly": "true",
        })

    async def update_stop_loss(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        new_sl_price: float,
        old_order_id: Optional[int] = None,
    ) -> dict:
        """
        Replace the existing SL order with a new STOP_MARKET at new_sl_price.
        Places new order FIRST (continuous protection), then cancels old.
        Returns the new SL order dict.
        """
        side = "BUY" if direction == "LONG" else "SELL"
        new_order = await self.place_stop_loss(symbol, side, quantity, new_sl_price)
        if old_order_id:
            try:
                await self.cancel_order(symbol, old_order_id)
            except Exception as exc:
                logger.debug("Cancel old SL order {} failed: {}", old_order_id, exc)
        return new_order


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

    async def _get(self, path: str, params: dict) -> dict:
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        params_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sig = self._gen_signature(ts, params_str)
        headers = {
            "X-BAPI-API-KEY": self._key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": sig,
            "X-BAPI-RECV-WINDOW": recv_window,
        }
        url = self._base + path
        async with self._session.get(url, headers=headers, params=params) as r:
            data = await r.json()
            if data.get("retCode") != 0:
                logger.error("Bybit API GET error: {}", data)
            return data

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
            if resp.get("retCode") != 0:
                result["status"] = "error"
                result["error"] = resp.get("retMsg", "Bybit API error")
                logger.error("Bybit order rejected ({}): {}", params.symbol, resp.get("retMsg"))
            else:
                result["order_id"] = resp.get("result", {}).get("orderId")
                result["status"] = "open"
                logger.info("Bybit trade opened: {}", result)
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("Bybit trade execution failed ({}): {}", params.symbol, exc)

        return result

    async def get_open_positions(self) -> list:
        """Returns list of currently open positions from Bybit."""
        try:
            data = await self._get("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
            positions = data.get("result", {}).get("list", [])
            return [p for p in positions if float(p.get("size", 0)) != 0]
        except Exception as exc:
            logger.debug("get_open_positions Bybit error: {}", exc)
            return []

    async def get_account_balance(self) -> float:
        """Get available USDT balance from Bybit."""
        try:
            data = await self._get("/v5/account/wallet-balance", {"accountType": "CONTRACT"})
            coins = data.get("result", {}).get("list", [{}])[0].get("coin", [])
            for coin in coins:
                if coin.get("coin") == "USDT":
                    return float(coin.get("availableToWithdraw", 0))
        except Exception as exc:
            logger.debug("get_account_balance Bybit error: {}", exc)
        return 0.0

    async def cancel_order(self, symbol: str, order_id) -> dict:
        """Cancel a specific open order by ID."""
        resp = await self._post("/v5/order/cancel", {
            "category": "linear",
            "symbol": symbol,
            "orderId": str(order_id),
        })
        if resp.get("retCode") != 0:
            logger.debug("Bybit cancel_order {}: {}", order_id, resp.get("retMsg"))
        return resp

    async def close_position_market(
        self, symbol: str, direction: str, quantity: float
    ) -> dict:
        """Force-close an open position with a market order (reduceOnly)."""
        close_side = "Sell" if direction == "LONG" else "Buy"
        logger.info("Bybit force-close {} {} qty={}", symbol, close_side, quantity)
        resp = await self._post("/v5/order/create", {
            "category": "linear",
            "symbol": symbol,
            "side": close_side,
            "orderType": "Market",
            "qty": str(quantity),
            "reduceOnly": True,
            "timeInForce": "GoodTillCancel",
        })
        if resp.get("retCode") != 0:
            logger.error("Bybit force-close failed {}: {}", symbol, resp.get("retMsg"))
        return resp

    async def update_stop_loss(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        new_sl_price: float,
        old_order_id: Optional[str] = None,     # accepted for interface compatibility, unused
    ) -> dict:
        """
        Update SL on an open Bybit position via the trading-stop endpoint.
        Bybit manages SL at the position level (not as a cancellable order),
        so old_order_id is intentionally ignored.
        """
        resp = await self._post("/v5/position/trading-stop", {
            "category": "linear",
            "symbol": symbol,
            "stopLoss": str(new_sl_price),
            "slTriggerBy": "LastPrice",
            "positionIdx": 0,
        })
        if resp.get("retCode") != 0:
            logger.debug("Bybit update_stop_loss {}: {}", symbol, resp.get("retMsg"))
        return resp
