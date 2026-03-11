"""
data_engine/websocket_feed.py — Real-time WebSocket market feeds
Handles Binance & Bybit futures streams with auto-reconnect.
"""
import asyncio
import json
import time
from typing import Callable, Dict, List, Optional
import websockets
from loguru import logger

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Shared order-book / ticker cache updated by WS feeds
# ─────────────────────────────────────────────────────────────────────────────
class MarketCache:
    def __init__(self) -> None:
        self.tickers: Dict[str, dict] = {}        # symbol → latest 24h ticker
        self.klines: Dict[str, dict] = {}          # symbol → latest closed kline
        self.order_books: Dict[str, dict] = {}     # symbol → {bids, asks}
        self.open_interest: Dict[str, float] = {}  # symbol → OI value
        self._callbacks: List[Callable] = []

    def register_callback(self, fn: Callable) -> None:
        """Register a coroutine to be called whenever new data arrives."""
        self._callbacks.append(fn)

    async def _emit(self, event: str, payload: dict) -> None:
        for cb in self._callbacks:
            try:
                await cb(event, payload)
            except Exception as exc:
                logger.error("Callback error: {}", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Binance WebSocket Feed
# ─────────────────────────────────────────────────────────────────────────────
class BinanceWebSocketFeed:
    """
    Subscribes to Binance Futures combined streams:
    - !miniTicker@arr  (all 24h mini-tickers)
    - <symbol>@kline_15m  (15-minute kline per coin)
    - <symbol>@openInterest (open-interest)
    """
    WS_BASE = "wss://fstream.binance.com"

    def __init__(self, cache: MarketCache, symbols: List[str]) -> None:
        self._cache = cache
        self._symbols = symbols
        self._running = False

    async def start(self) -> None:
        self._running = True
        await asyncio.gather(
            self._stream_mini_tickers(),
            self._stream_klines(),
        )

    async def stop(self) -> None:
        self._running = False

    # ── All-market mini-ticker (price / volume) ───────────────────────────
    async def _stream_mini_tickers(self) -> None:
        url = f"{self.WS_BASE}/stream?streams=!miniTicker@arr"
        await self._connect_with_reconnect(url, self._handle_mini_tickers)

    async def _handle_mini_tickers(self, raw: str) -> None:
        msg = json.loads(raw)
        data = msg.get("data", msg)
        for ticker in data:
            sym = ticker.get("s", "")
            if sym.endswith("USDT"):
                self._cache.tickers[sym] = {
                    "symbol": sym,
                    "price": float(ticker["c"]),
                    "volume": float(ticker["v"]),
                    "quote_volume": float(ticker["q"]),
                    "price_change_pct": float(ticker["P"]),
                    "ts": ticker["E"],
                }
        await self._cache._emit("mini_ticker_update", {})

    # ── Per-symbol 15-m kline streams ────────────────────────────────────
    async def _stream_klines(self) -> None:
        # Binance allows up to 200 streams per connection
        chunk_size = 200
        chunks = [
            self._symbols[i: i + chunk_size]
            for i in range(0, len(self._symbols), chunk_size)
        ]
        await asyncio.gather(*[self._kline_chunk(chunk) for chunk in chunks])

    async def _kline_chunk(self, symbols: List[str]) -> None:
        streams = "/".join(f"{s.lower()}@kline_15m" for s in symbols)
        url = f"{self.WS_BASE}/stream?streams={streams}"
        await self._connect_with_reconnect(url, self._handle_kline)

    async def _handle_kline(self, raw: str) -> None:
        msg = json.loads(raw)
        data = msg.get("data", {})
        if data.get("e") == "kline":
            k = data["k"]
            sym = data["s"]
            if k["x"]:          # only process closed candles
                self._cache.klines[sym] = {
                    "symbol": sym,
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                    "quote_volume": float(k["q"]),
                    "ts": k["t"],
                }
                await self._cache._emit("kline_closed", {"symbol": sym})

    # ── Generic reconnect loop ────────────────────────────────────────────
    async def _connect_with_reconnect(
        self,
        url: str,
        handler: Callable,
        max_retries: int = 0,       # 0 = infinite
    ) -> None:
        retries = 0
        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                ) as ws:
                    logger.info("WS connected: {}", url[:80])
                    retries = 0
                    async for message in ws:
                        if not self._running:
                            break
                        await handler(message)
            except (websockets.ConnectionClosed, OSError) as exc:
                retries += 1
                wait = min(2 ** retries, 60)
                logger.warning("WS disconnected ({}). Retry in {}s", exc, wait)
                await asyncio.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# Bybit WebSocket Feed
# ─────────────────────────────────────────────────────────────────────────────
class BybitWebSocketFeed:
    """Subscribes to Bybit V5 public linear WebSocket."""
    WS_URL = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, cache: MarketCache, symbols: List[str]) -> None:
        self._cache = cache
        self._symbols = symbols
        self._running = False

    async def start(self) -> None:
        self._running = True
        # chunk subscriptions (max 10 topics per message)
        await self._connect_with_reconnect()

    async def stop(self) -> None:
        self._running = False

    async def _connect_with_reconnect(self) -> None:
        retries = 0
        while self._running:
            try:
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    logger.info("Bybit WS connected")
                    retries = 0

                    # Subscribe to all kline.15 topics
                    for i in range(0, len(self._symbols), 10):
                        chunk = self._symbols[i: i + 10]
                        topics = [f"kline.15.{s}" for s in chunk]
                        await ws.send(json.dumps({"op": "subscribe", "args": topics}))
                        await asyncio.sleep(0.1)

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle(raw)
            except (websockets.ConnectionClosed, OSError) as exc:
                retries += 1
                wait = min(2 ** retries, 60)
                logger.warning("Bybit WS disconnected ({}). Retry in {}s", exc, wait)
                await asyncio.sleep(wait)

    async def _handle(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg.get("topic", "").startswith("kline"):
            topic_parts = msg["topic"].split(".")
            sym = topic_parts[-1]
            for candle in msg.get("data", []):
                if candle.get("confirm"):   # closed candle
                    self._cache.klines[sym] = {
                        "symbol": sym,
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle["volume"]),
                        "ts": candle["start"],
                    }
                    await self._cache._emit("kline_closed", {"symbol": sym})
