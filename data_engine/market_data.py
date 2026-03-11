"""
data_engine/market_data.py — Async REST market data fetcher
Supports Binance Futures (production + testnet) + Bybit Futures with rate-limit handling.
"""
import asyncio
from typing import Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


BINANCE_FAPI_PROD    = "https://fapi.binance.com"
BINANCE_FAPI_TESTNET = "https://testnet.binancefuture.com"
BYBIT_V5_BASE        = "https://api.bybit.com"

KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


class MarketDataEngine:
    """
    Fetches OHLCV data, order-book snapshots, open-interest,
    and funding-rate data from Binance / Bybit asynchronously.
    Supports testnet mode and HTTP proxy.
    """

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_semaphore = asyncio.Semaphore(20)
        self._binance_base = (
            BINANCE_FAPI_TESTNET if settings.binance_testnet else BINANCE_FAPI_PROD
        )
        self._proxy: Optional[str] = settings.http_proxy or None
        if settings.binance_testnet:
            logger.info("MarketDataEngine using Binance TESTNET ({})", self._binance_base)
        if self._proxy:
            logger.info("MarketDataEngine using HTTP proxy: {}", self._proxy)

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=15)
        self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("MarketDataEngine started")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        logger.info("MarketDataEngine stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _get(self, url: str, params: dict | None = None) -> dict | list:
        async with self._rate_limit_semaphore:
            async with self._session.get(
                url, params=params, proxy=self._proxy
            ) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning("Rate limited. Sleeping {}s", retry_after)
                    await asyncio.sleep(retry_after)
                    raise Exception("Rate limited")
                resp.raise_for_status()
                return await resp.json()

    async def get_binance_symbols(self) -> List[str]:
        data = await self._get(f"{self._binance_base}/fapi/v1/exchangeInfo")
        symbols = [
            s["symbol"] for s in data["symbols"]
            if s["status"] == "TRADING" and s["symbol"].endswith("USDT")
        ]
        logger.debug("Binance symbols fetched: {}", len(symbols))
        return symbols

    async def get_bybit_symbols(self) -> List[str]:
        data = await self._get(
            f"{BYBIT_V5_BASE}/v5/market/instruments-info",
            params={"category": "linear", "limit": 1000},
        )
        symbols = [
            s["symbol"] for s in data["result"]["list"]
            if s["symbol"].endswith("USDT") and s["status"] == "Trading"
        ]
        logger.debug("Bybit symbols fetched: {}", len(symbols))
        return symbols

    async def get_klines_binance(
        self, symbol: str, interval: str = "15m", limit: int = 200,
    ) -> pd.DataFrame:
        raw = await self._get(
            f"{self._binance_base}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume", "quote_volume"]]

    async def get_klines_bybit(
        self, symbol: str, interval: str = "15", limit: int = 200,
    ) -> pd.DataFrame:
        raw = await self._get(
            f"{BYBIT_V5_BASE}/v5/market/kline",
            params={"category": "linear", "symbol": symbol, "interval": interval, "limit": limit},
        )
        rows = raw["result"]["list"]
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    async def get_open_interest_binance(self, symbol: str) -> float:
        data = await self._get(
            f"{self._binance_base}/fapi/v1/openInterest",
            params={"symbol": symbol},
        )
        return float(data["openInterest"])

    async def get_open_interest_bybit(self, symbol: str) -> float:
        data = await self._get(
            f"{BYBIT_V5_BASE}/v5/market/open-interest",
            params={"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": 1},
        )
        rows = data["result"]["list"]
        return float(rows[0]["openInterest"]) if rows else 0.0

    async def get_funding_rate_binance(self, symbol: str) -> float:
        data = await self._get(
            f"{self._binance_base}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
        )
        return float(data["lastFundingRate"])

    async def get_order_book_binance(self, symbol: str, limit: int = 20) -> Dict:
        data = await self._get(
            f"{self._binance_base}/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
        )
        return {
            "bids": [[float(p), float(q)] for p, q in data["bids"]],
            "asks": [[float(p), float(q)] for p, q in data["asks"]],
        }

    async def get_ticker_24h_binance(self, symbol: str) -> Dict:
        return await self._get(
            f"{self._binance_base}/fapi/v1/ticker/24hr",
            params={"symbol": symbol},
        )

    async def get_all_tickers_binance(self) -> List[Dict]:
        return await self._get(f"{self._binance_base}/fapi/v1/ticker/24hr")

    async def bulk_fetch_klines(
        self,
        symbols: List[str],
        interval: str = "15m",
        limit: int = 200,
        exchange: str = "binance",
    ) -> Dict[str, pd.DataFrame]:
        semaphore = asyncio.Semaphore(50)

        async def fetch_one(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
            async with semaphore:
                try:
                    if exchange == "binance":
                        df = await self.get_klines_binance(sym, interval, limit)
                    else:
                        df = await self.get_klines_bybit(sym, interval, limit)
                    return sym, df
                except Exception as exc:
                    logger.warning("Failed to fetch {} - {}", sym, exc)
                    return sym, None

        tasks = [fetch_one(s) for s in symbols]
        results = await asyncio.gather(*tasks)
        return {sym: df for sym, df in results if df is not None}
