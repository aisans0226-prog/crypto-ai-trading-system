"""
trading/trade_executor.py — Auto trade execution via Binance & Bybit Futures APIs.

Flow:
  detect signal → validate risk → place market order → set SL (stop-market) → set TP (take-profit-market)

Supports:
  - Binance USDM Futures
  - Bybit Linear Perpetual V5

Live-trading fixes applied:
  ✓ Exchange filter cache (LOT_SIZE stepSize, PRICE_FILTER tickSize, MIN_NOTIONAL)
  ✓ Quantity/price precision rounded per-symbol before every order
  ✓ recvWindow=5000 on all Binance signed requests
  ✓ 3-attempt retry with exponential back-off on transient errors
  ✓ API-key validation at startup
  ✓ Bybit: explicit set_leverage() before execute_trade()
  ✓ validate_min_notional() check before order placement
"""
import asyncio
import hashlib
import hmac
import json
import math
import time
from typing import Dict, Optional, Tuple

import aiohttp
from loguru import logger

from config import settings
from trading.risk_manager import RiskParameters


# ── Precision helpers ──────────────────────────────────────────────────────────

def _decimals(step: float) -> int:
    """Return number of decimal places implied by a step/tick size."""
    if step <= 0:
        return 8
    if step >= 1:
        return 0
    return max(0, -int(math.floor(math.log10(step))))


def _floor_to_step(value: float, step: float) -> float:
    """Floor `value` to the nearest multiple of `step`."""
    if step <= 0:
        return value
    return math.floor(value / step) * step


def _round_to_tick(value: float, tick: float) -> float:
    """Round `value` to the nearest multiple of `tick`."""
    if tick <= 0:
        return value
    return round(round(value / tick) * tick, _decimals(tick))


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
        # symbol → {step_size, tick_size, min_qty, min_notional, qty_dec, price_dec}
        self._symbol_filters: Dict[str, dict] = {}

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ── Signing ───────────────────────────────────────────────────────────────
    def _sign(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(
            self._secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

    # ── HTTP helpers (retry + recvWindow) ─────────────────────────────────────
    async def _get(self, path: str, params: dict) -> any:
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        last_exc: Exception = Exception("no attempts")
        for attempt in range(3):
            p = {**params,
                 "timestamp": int(time.time() * 1000),
                 "recvWindow": 5000}
            p["signature"] = self._sign(p)
            try:
                async with self._session.get(url, headers=headers, params=p) as r:
                    data = await r.json()
                    if r.status != 200:
                        msg = data.get("msg", "unknown") if isinstance(data, dict) else str(data)
                        logger.error("Binance GET {} error {}: {}", path, r.status, msg)
                        raise Exception(f"Binance API error {r.status}: {msg}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.debug("Binance GET retry {}/3 in {}s: {}", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
        raise last_exc

    async def _delete(self, path: str, params: dict) -> dict:
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        last_exc: Exception = Exception("no attempts")
        for attempt in range(3):
            p = {**params,
                 "timestamp": int(time.time() * 1000),
                 "recvWindow": 5000}
            p["signature"] = self._sign(p)
            try:
                async with self._session.delete(url, headers=headers, params=p) as r:
                    data = await r.json()
                    if r.status != 200:
                        msg = data.get("msg", "unknown") if isinstance(data, dict) else str(data)
                        logger.error("Binance DELETE {} error {}: {}", path, r.status, msg)
                        raise Exception(f"Binance API error {r.status}: {msg}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.debug("Binance DELETE retry {}/3 in {}s: {}", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
        raise last_exc

    async def _post(self, path: str, params: dict) -> dict:
        headers = {"X-MBX-APIKEY": self._key}
        url = self._base + path
        last_exc: Exception = Exception("no attempts")
        for attempt in range(3):
            p = {**params,
                 "timestamp": int(time.time() * 1000),
                 "recvWindow": 5000}
            p["signature"] = self._sign(p)
            try:
                async with self._session.post(url, headers=headers, params=p) as r:
                    data = await r.json()
                    if r.status != 200:
                        msg = data.get("msg", "unknown") if isinstance(data, dict) else str(data)
                        logger.error("Binance POST {} error {}: {}", path, r.status, msg)
                        raise Exception(f"Binance API error {r.status}: {msg}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.debug("Binance POST retry {}/3 in {}s: {}", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
        raise last_exc

    # ── Exchange filters ───────────────────────────────────────────────────────
    async def load_exchange_filters(self) -> None:
        """
        Fetch LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL for all USDT linear symbols.
        Must be called once after start() before placing any orders.
        """
        try:
            url = self._base + "/fapi/v1/exchangeInfo"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                data = await r.json()

            loaded = 0
            for sym_info in data.get("symbols", []):
                if sym_info.get("contractType") != "PERPETUAL":
                    continue
                symbol = sym_info["symbol"]
                filters = {f["filterType"]: f for f in sym_info.get("filters", [])}

                lot = filters.get("LOT_SIZE", {})
                prf = filters.get("PRICE_FILTER", {})
                mno = filters.get("MIN_NOTIONAL", {})

                step = float(lot.get("stepSize", "0.001"))
                tick = float(prf.get("tickSize", "0.00001"))
                min_qty = float(lot.get("minQty", "0.001"))
                min_notional = float(mno.get("notional", "5"))

                self._symbol_filters[symbol] = {
                    "step_size":    step,
                    "tick_size":    tick,
                    "min_qty":      min_qty,
                    "min_notional": min_notional,
                    "qty_dec":      _decimals(step),
                    "price_dec":    _decimals(tick),
                }
                loaded += 1

            logger.info("Binance: loaded exchange filters for {} symbols", loaded)
        except Exception as exc:
            logger.error("Binance load_exchange_filters failed: {} — using fallback precision", exc)

    def _round_qty(self, symbol: str, quantity: float) -> float:
        """Floor quantity to the symbol's LOT_SIZE step and return correctly rounded float."""
        f = self._symbol_filters.get(symbol)
        if not f:
            return round(quantity, 3)
        qty = _floor_to_step(quantity, f["step_size"])
        return round(qty, f["qty_dec"])

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to the symbol's PRICE_FILTER tick size."""
        f = self._symbol_filters.get(symbol)
        if not f:
            return round(price, 6)
        return _round_to_tick(price, f["tick_size"])

    def validate_min_notional(self, symbol: str, notional: float) -> bool:
        """Return True if notional meets the exchange minimum (default: True when unknown)."""
        f = self._symbol_filters.get(symbol)
        if not f:
            return True
        ok = notional >= f["min_notional"]
        if not ok:
            logger.warning(
                "{} notional {:.2f} USDT < minimum {:.2f} USDT — order would be rejected",
                symbol, notional, f["min_notional"],
            )
        return ok

    # ── API key validation ─────────────────────────────────────────────────────
    async def validate_api_key(self) -> bool:
        """Verify the configured API key works by fetching account balance."""
        try:
            data = await self._get("/fapi/v2/balance", {})
            if isinstance(data, list):
                logger.info("Binance API key OK ({} testnet={})",
                            "TESTNET" if settings.binance_testnet else "MAINNET",
                            settings.binance_testnet)
                return True
            logger.error("Binance API key validation failed — unexpected response: {}", data)
        except Exception as exc:
            logger.error("Binance API key validation FAILED: {}", exc)
        return False

    # ── Margin mode ───────────────────────────────────────────────────────────
    async def set_margin_mode(self, symbol: str, mode: str = "ISOLATED") -> None:
        """
        Set margin type for symbol before opening a position.
        mode = ISOLATED | CROSSED.  Code -4046 = 'No need to change margin type' → OK.
        """
        try:
            await self._post("/fapi/v1/marginType", {
                "symbol":     symbol,
                "marginType": mode,
            })
        except Exception as exc:
            # -4046: already in the requested mode — safe to ignore
            if "-4046" not in str(exc) and "No need to change" not in str(exc):
                logger.warning("set_margin_mode {}/{}: {}", symbol, mode, exc)

    # ── Set leverage ──────────────────────────────────────────────────────────
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._post("/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage,
        })

    # ── Place market order ─────────────────────────────────────────────────────
    async def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> dict:
        qty = self._round_qty(symbol, quantity)
        logger.info("Binance MARKET {} {} qty={}", symbol, side, qty)
        return await self._post("/fapi/v1/order", {
            "symbol":   symbol,
            "side":     side,
            "type":     "MARKET",
            "quantity": qty,
        })

    # ── Stop-loss order ────────────────────────────────────────────────────────
    async def place_stop_loss(
        self, symbol: str, side: str, quantity: float, stop_price: float
    ) -> dict:
        close_side = "SELL" if side == "BUY" else "BUY"
        qty = self._round_qty(symbol, quantity)
        stop = self._round_price(symbol, stop_price)
        return await self._post("/fapi/v1/order", {
            "symbol":     symbol,
            "side":       close_side,
            "type":       "STOP_MARKET",
            "stopPrice":  stop,
            "quantity":   qty,
            "reduceOnly": "true",
        })

    # ── Take-profit order ──────────────────────────────────────────────────────
    async def place_take_profit(
        self, symbol: str, side: str, quantity: float, tp_price: float
    ) -> dict:
        close_side = "SELL" if side == "BUY" else "BUY"
        qty = self._round_qty(symbol, quantity)
        tp = self._round_price(symbol, tp_price)
        return await self._post("/fapi/v1/order", {
            "symbol":     symbol,
            "side":       close_side,
            "type":       "TAKE_PROFIT_MARKET",
            "stopPrice":  tp,
            "quantity":   qty,
            "reduceOnly": "true",
        })

    # ── Full trade execution ───────────────────────────────────────────────────
    async def execute_trade(self, params: RiskParameters) -> Dict:
        side   = "BUY" if params.direction == "LONG" else "SELL"
        result = {"exchange": "binance", "symbol": params.symbol}

        # Guard: check minimum notional before touching exchange
        if not self.validate_min_notional(params.symbol, params.position_size_usdt):
            result["status"] = "error"
            result["error"]  = f"notional {params.position_size_usdt:.2f} < min_notional"
            return result

        try:
            # Enforce isolated margin before setting leverage or opening position
            await self.set_margin_mode(params.symbol, "ISOLATED")
            await self.set_leverage(params.symbol, params.leverage)

            order = await self.place_market_order(
                params.symbol, side, params.quantity
            )
            result["order_id"]    = order.get("orderId")
            result["fill_price"]  = float(order.get("avgPrice") or params.entry_price)
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
            logger.info("Binance trade opened: {}", result)
        except Exception as exc:
            result["status"] = "error"
            result["error"]  = str(exc)
            logger.error("Binance trade execution failed ({}): {}", params.symbol, exc)

        return result

    # ── Account / position queries ─────────────────────────────────────────────
    async def get_open_positions(self) -> list:
        try:
            data = await self._get("/fapi/v2/positionRisk", {})
            if not isinstance(data, list):
                return []
            return [p for p in data if float(p.get("positionAmt", 0)) != 0]
        except Exception as exc:
            logger.debug("get_open_positions Binance error: {}", exc)
            return []

    async def get_account_balance(self) -> float:
        try:
            assets = await self._get("/fapi/v2/balance", {})
            if not isinstance(assets, list):
                return 0.0
            for asset in assets:
                if asset.get("asset") == "USDT":
                    return float(asset.get("availableBalance", 0))
        except Exception as exc:
            logger.debug("get_account_balance Binance error: {}", exc)
        return 0.0

    async def get_recent_pnl(self, symbol: str) -> float:
        """Return net realized PnL for the symbol over the past 24 h.

        Fetches REALIZED_PNL (position settlement) AND COMMISSION (taker fees)
        separately then sums them, because the /income endpoint splits these into
        different incomeType entries.  COMMISSION values are already negative on
        Binance, so summing gives the true net-of-fee realized PnL.
        """
        try:
            start_time = int((time.time() - 86400) * 1000)
            total = 0.0
            for income_type in ("REALIZED_PNL", "COMMISSION"):
                data = await self._get("/fapi/v1/income", {
                    "symbol":     symbol,
                    "incomeType": income_type,
                    "startTime":  start_time,
                    "limit":      50,
                })
                if isinstance(data, list):
                    total += sum(float(r.get("income", 0)) for r in data)
            return total
        except Exception as exc:
            logger.debug("get_recent_pnl Binance error {}: {}", symbol, exc)
        return 0.0

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        qty = self._round_qty(symbol, quantity)
        prc = self._round_price(symbol, price)
        logger.info("Binance LIMIT {} {} qty={} price={}", symbol, side, qty, prc)
        return await self._post("/fapi/v1/order", {
            "symbol":      symbol,
            "side":        side,
            "type":        "LIMIT",
            "price":       prc,
            "quantity":    qty,
            "timeInForce": "GTC",
        })

    async def get_order_status(self, symbol: str, order_id: int) -> dict:
        return await self._get("/fapi/v1/order", {
            "symbol": symbol, "orderId": order_id,
        })

    async def cancel_order(self, symbol: str, order_id) -> dict:
        return await self._delete("/fapi/v1/order", {
            "symbol":  symbol,
            "orderId": int(order_id),
        })

    async def close_position_market(
        self, symbol: str, direction: str, quantity: float
    ) -> dict:
        close_side = "SELL" if direction == "LONG" else "BUY"
        qty = self._round_qty(symbol, quantity)
        logger.info("Binance force-close {} {} qty={}", symbol, close_side, qty)
        return await self._post("/fapi/v1/order", {
            "symbol":     symbol,
            "side":       close_side,
            "type":       "MARKET",
            "quantity":   qty,
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
        Replace the existing SL with a new STOP_MARKET at new_sl_price.
        Places the new order first (continuous protection), then cancels old.
        """
        side      = "BUY" if direction == "LONG" else "SELL"
        new_order = await self.place_stop_loss(symbol, side, quantity, new_sl_price)
        if old_order_id:
            try:
                await self.cancel_order(symbol, old_order_id)
            except Exception as exc:
                logger.debug("Cancel old SL {} failed (may already be filled): {}", old_order_id, exc)
        return new_order


# ─────────────────────────────────────────────────────────────────────────────
# Bybit Futures Executor
# ─────────────────────────────────────────────────────────────────────────────
class BybitExecutor:
    BASE      = "https://api.bybit.com"
    BASE_TEST = "https://api-testnet.bybit.com"

    def __init__(self) -> None:
        self._key    = settings.bybit_api_key
        self._secret = settings.bybit_api_secret
        self._base   = self.BASE_TEST if settings.bybit_testnet else self.BASE
        self._session: Optional[aiohttp.ClientSession] = None
        # symbol → {step_size, tick_size, min_qty, min_notional, qty_dec, price_dec}
        self._symbol_filters: Dict[str, dict] = {}

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    def _gen_signature(self, timestamp: str, params_str: str) -> str:
        recv_window = "5000"
        sign_str    = timestamp + self._key + recv_window + params_str
        return hmac.new(
            self._secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()

    # ── HTTP helpers with retry ────────────────────────────────────────────────
    async def _get(self, path: str, params: dict) -> dict:
        recv_window = "5000"
        url         = self._base + path
        last_exc: Exception = Exception("no attempts")
        for attempt in range(3):
            ts         = str(int(time.time() * 1000))
            params_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            sig        = self._gen_signature(ts, params_str)
            headers    = {
                "X-BAPI-API-KEY":      self._key,
                "X-BAPI-TIMESTAMP":    ts,
                "X-BAPI-SIGN":         sig,
                "X-BAPI-RECV-WINDOW":  recv_window,
            }
            try:
                async with self._session.get(url, headers=headers, params=params) as r:
                    data = await r.json()
                    if data.get("retCode") not in (0, None):
                        logger.error("Bybit GET {} error: retCode={} msg={}",
                                     path, data.get("retCode"), data.get("retMsg"))
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.debug("Bybit GET retry {}/3 in {}s: {}", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
        raise last_exc

    async def _post(self, path: str, body: dict) -> dict:
        recv_window = "5000"
        url         = self._base + path
        body_str    = json.dumps(body)
        last_exc: Exception = Exception("no attempts")
        for attempt in range(3):
            ts  = str(int(time.time() * 1000))
            sig = self._gen_signature(ts, body_str)
            headers = {
                "X-BAPI-API-KEY":      self._key,
                "X-BAPI-TIMESTAMP":    ts,
                "X-BAPI-SIGN":         sig,
                "X-BAPI-RECV-WINDOW":  recv_window,
                "Content-Type":        "application/json",
            }
            try:
                async with self._session.post(url, headers=headers, data=body_str) as r:
                    data = await r.json()
                    if data.get("retCode") not in (0, None):
                        logger.error("Bybit POST {} error: retCode={} msg={}",
                                     path, data.get("retCode"), data.get("retMsg"))
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.debug("Bybit POST retry {}/3 in {}s: {}", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
        raise last_exc

    # ── API key validation ─────────────────────────────────────────────────────
    async def validate_api_key(self) -> bool:
        """Verify the configured API key works by fetching account balance."""
        try:
            data = await self._get("/v5/account/wallet-balance", {"accountType": "CONTRACT"})
            if data.get("retCode") == 0:
                logger.info("Bybit API key OK ({})",
                            "TESTNET" if settings.bybit_testnet else "MAINNET")
                return True
            logger.error("Bybit API key validation failed: {}", data.get("retMsg"))
        except Exception as exc:
            logger.error("Bybit API key validation FAILED: {}", exc)
        return False

    # ── Exchange filters ───────────────────────────────────────────────────────
    async def load_exchange_filters(self) -> None:
        """
        Fetch LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL for all USDT linear symbols.
        Must be called once after start() before placing any orders.
        """
        try:
            data = await self._get("/v5/market/instruments-info",
                                   {"category": "linear", "limit": "1000"})
            loaded = 0
            for item in data.get("result", {}).get("list", []):
                if item.get("settleCoin") != "USDT":
                    continue
                symbol    = item["symbol"]
                lot_size  = item.get("lotSizeFilter", {})
                price_flt = item.get("priceFilter", {})

                step     = float(lot_size.get("qtyStep", "0.001"))
                tick     = float(price_flt.get("tickSize", "0.00001"))
                min_qty  = float(lot_size.get("minOrderQty", "0.001"))
                min_val  = float(lot_size.get("minOrderAmt", "5"))   # minOrderAmt = min notional

                self._symbol_filters[symbol] = {
                    "step_size":    step,
                    "tick_size":    tick,
                    "min_qty":      min_qty,
                    "min_notional": min_val,
                    "qty_dec":      _decimals(step),
                    "price_dec":    _decimals(tick),
                }
                loaded += 1
            logger.info("Bybit: loaded exchange filters for {} symbols", loaded)
        except Exception as exc:
            logger.error("Bybit load_exchange_filters failed: {} — using fallback precision", exc)

    def _round_qty(self, symbol: str, quantity: float) -> float:
        """Floor quantity to the symbol's lotSizeFilter qtyStep."""
        f = self._symbol_filters.get(symbol)
        if not f:
            return round(quantity, 3)
        qty = _floor_to_step(quantity, f["step_size"])
        return round(qty, f["qty_dec"])

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to the symbol's PRICE_FILTER tickSize."""
        f = self._symbol_filters.get(symbol)
        if not f:
            return round(price, 6)
        return _round_to_tick(price, f["tick_size"])

    def validate_min_notional(self, symbol: str, notional: float) -> bool:
        f = self._symbol_filters.get(symbol)
        if not f:
            return True
        ok = notional >= f["min_notional"]
        if not ok:
            logger.warning(
                "{} notional {:.2f} USDT < Bybit minimum {:.2f} USDT — order would be rejected",
                symbol, notional, f["min_notional"],
            )
        return ok

    # ── Margin mode ───────────────────────────────────────────────────────────
    async def set_margin_mode(self, symbol: str, mode: str = "ISOLATED_MARGIN") -> None:
        """
        Set trade mode for symbol: ISOLATED_MARGIN | REGULAR_MARGIN (cross).
        retCode 110026 = 'Position mode not modified' — treat as success.
        """
        resp = await self._post("/v5/account/set-margin-mode", {
            "setMarginMode": mode,
        })
        code = resp.get("retCode", 0)
        if code not in (0, 3400045):
            logger.warning("Bybit set_margin_mode {}/{}: retCode={} msg={}",
                           symbol, mode, code, resp.get("retMsg"))

    # ── Set leverage ───────────────────────────────────────────────────────────
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Explicitly set leverage for a symbol before opening a position.
        retCode 110043 = 'leverage not modified' — treat as success.
        """
        resp = await self._post("/v5/position/set-leverage", {
            "category":    "linear",
            "symbol":      symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        })
        if resp.get("retCode") not in (0, 110043):
            logger.warning("Bybit set_leverage {} {}x: retCode={} msg={}",
                           symbol, leverage, resp.get("retCode"), resp.get("retMsg"))

    # ── Full trade execution ───────────────────────────────────────────────────
    async def execute_trade(self, params: RiskParameters) -> Dict:
        side   = "Buy" if params.direction == "LONG" else "Sell"
        result = {"exchange": "bybit", "symbol": params.symbol}

        # Guard: check minimum notional
        if not self.validate_min_notional(params.symbol, params.position_size_usdt):
            result["status"] = "error"
            result["error"]  = f"notional {params.position_size_usdt:.2f} < min_notional"
            return result

        try:
            # Enforce isolated margin, then set leverage
            await self.set_margin_mode(params.symbol, "ISOLATED_MARGIN")
            await self.set_leverage(params.symbol, params.leverage)

            qty   = self._round_qty(params.symbol, params.quantity)
            sl    = self._round_price(params.symbol, params.stop_loss)
            tp    = self._round_price(params.symbol, params.take_profit)

            body = {
                "category":    "linear",
                "symbol":      params.symbol,
                "side":        side,
                "orderType":   "Market",
                "qty":         str(qty),
                "stopLoss":    str(sl),
                "takeProfit":  str(tp),
                "slTriggerBy": "LastPrice",
                "tpTriggerBy": "LastPrice",
                "timeInForce": "GoodTillCancel",
            }
            resp = await self._post("/v5/order/create", body)
            if resp.get("retCode") != 0:
                result["status"] = "error"
                result["error"]  = resp.get("retMsg", "Bybit API error")
                logger.error("Bybit order rejected ({}): retCode={} msg={}",
                             params.symbol, resp.get("retCode"), resp.get("retMsg"))
            else:
                result["order_id"]   = resp.get("result", {}).get("orderId")
                result["fill_price"] = params.entry_price   # market — actual fill confirmed by WS
                result["status"]     = "open"
                logger.info("Bybit trade opened: {}", result)
        except Exception as exc:
            result["status"] = "error"
            result["error"]  = str(exc)
            logger.error("Bybit trade execution failed ({}): {}", params.symbol, exc)

        return result

    # ── Account / position queries ─────────────────────────────────────────────
    async def get_open_positions(self) -> list:
        try:
            data      = await self._get("/v5/position/list",
                                        {"category": "linear", "settleCoin": "USDT"})
            positions = data.get("result", {}).get("list", [])
            return [p for p in positions if float(p.get("size", 0)) != 0]
        except Exception as exc:
            logger.debug("get_open_positions Bybit error: {}", exc)
            return []

    async def get_account_balance(self) -> float:
        try:
            data  = await self._get("/v5/account/wallet-balance", {"accountType": "CONTRACT"})
            lst   = data.get("result", {}).get("list", [])
            coins = lst[0].get("coin", []) if lst else []
            for coin in coins:
                if coin.get("coin") == "USDT":
                    return float(coin.get("availableToWithdraw", 0))
        except Exception as exc:
            logger.debug("get_account_balance Bybit error: {}", exc)
        return 0.0

    async def cancel_order(self, symbol: str, order_id) -> dict:
        resp = await self._post("/v5/order/cancel", {
            "category": "linear",
            "symbol":   symbol,
            "orderId":  str(order_id),
        })
        if resp.get("retCode") != 0:
            logger.debug("Bybit cancel_order {}: retCode={} msg={}",
                         order_id, resp.get("retCode"), resp.get("retMsg"))
        return resp

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        logger.info("Bybit LIMIT {} {} qty={} price={}", symbol, side, quantity, price)
        resp = await self._post("/v5/order/create", {
            "category":    "linear",
            "symbol":      symbol,
            "side":        side,
            "orderType":   "Limit",
            "qty":         str(quantity),
            "price":       str(round(price, 8)),
            "timeInForce": "GTC",
        })
        if resp.get("retCode") != 0:
            logger.error("Bybit LIMIT order rejected {}: {}", symbol, resp.get("retMsg"))
        return resp

    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        resp = await self._get("/v5/order/realtime", {
            "category": "linear", "symbol": symbol, "orderId": order_id,
        })
        if resp.get("result", {}).get("list"):
            return resp
        return await self._get("/v5/order/history", {
            "category": "linear", "symbol": symbol, "orderId": order_id,
        })

    async def set_position_tp_sl(
        self, symbol: str, stop_loss: float, take_profit: float
    ) -> dict:
        resp = await self._post("/v5/position/trading-stop", {
            "category":    "linear",
            "symbol":      symbol,
            "stopLoss":    str(stop_loss),
            "takeProfit":  str(take_profit),
            "slTriggerBy": "LastPrice",
            "tpTriggerBy": "LastPrice",
            "positionIdx": 0,
        })
        if resp.get("retCode") != 0:
            logger.error("Bybit set_position_tp_sl {}: {}", symbol, resp.get("retMsg"))
        return resp

    async def close_position_market(
        self, symbol: str, direction: str, quantity: float
    ) -> dict:
        close_side = "Sell" if direction == "LONG" else "Buy"
        logger.info("Bybit force-close {} {} qty={}", symbol, close_side, quantity)
        resp = await self._post("/v5/order/create", {
            "category":    "linear",
            "symbol":      symbol,
            "side":        close_side,
            "orderType":   "Market",
            "qty":         str(quantity),
            "reduceOnly":  True,
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
        old_order_id: Optional[str] = None,   # accepted for interface compatibility, unused
    ) -> dict:
        """
        Update SL on an open Bybit position via the trading-stop endpoint.
        Bybit manages SL at position level — old_order_id is intentionally ignored.
        """
        resp = await self._post("/v5/position/trading-stop", {
            "category":    "linear",
            "symbol":      symbol,
            "stopLoss":    str(new_sl_price),
            "slTriggerBy": "LastPrice",
            "positionIdx": 0,
        })
        if resp.get("retCode") != 0:
            logger.debug("Bybit update_stop_loss {} retCode={} msg={}",
                         symbol, resp.get("retCode"), resp.get("retMsg"))
        return resp
