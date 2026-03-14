"""
data_engine/external_data.py — External contextual data sources.

Provides three supplementary data feeds consumed by the research engine:
  - Fear & Greed Index  (alternative.me, free, no key required)
  - Crypto news + sentiment  (CryptoPanic API, optional free-tier key)
  - Liquidation clusters  (Binance Futures public endpoint, no key required)

All fetchers degrade gracefully when API keys are absent or endpoints are
unreachable — callers always receive a safe default (None / [] / 0.0).
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


# ── Public endpoint constants ─────────────────────────────────────────────────
_FEAR_GREED_URL       = "https://api.alternative.me/fng/?limit=1"
_BINANCE_LIQ_URL      = "https://fapi.binance.com/fapi/v1/allForceOrders"
_CRYPTOPANIC_URL      = "https://cryptopanic.com/api/v1/posts/"

# Minimum USD value to keep a liquidation cluster (filters out tiny noise)
_LIQ_CLUSTER_MIN_USD  = 1_000.0
# Maximum clusters returned (keeps caller memory small)
_LIQ_CLUSTER_MAX      = 20


# ── Data-transfer objects ─────────────────────────────────────────────────────

@dataclass
class FearGreedData:
    """Snapshot of the Crypto Fear & Greed Index (0–100)."""
    value: int           # 0–100
    classification: str  # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
    timestamp: int       # Unix timestamp of the index reading


@dataclass
class NewsItem:
    """Single news article returned by CryptoPanic."""
    title: str
    url: str
    source: str        # domain, e.g. "cointelegraph.com"
    published_at: str  # ISO 8601 string
    sentiment: str     # "positive" | "negative" | "neutral"
    currencies: List[str]  # e.g. ["BTC", "ETH"]


@dataclass
class LiquidationCluster:
    """Aggregated forced-liquidation volume at a given price band."""
    price: float      # centre of the 0.5 % price band
    total_usd: float  # sum of liquidated notional USD in the band
    side: str         # "LONG" (was a long position) | "SHORT" (was a short position)


# ── Main fetcher class ────────────────────────────────────────────────────────

class ExternalDataFetcher:
    """
    Fetches macro and contextual data from external public sources.

    Designed to be composed alongside MarketDataEngine — it accepts an
    externally created aiohttp session so both engines share one connection
    pool, or it manages its own session when started standalone.

    Usage (standalone):
        fetcher = ExternalDataFetcher(cryptopanic_api_key="<key>")
        await fetcher.start()
        fg = await fetcher.get_fear_greed_index()
        await fetcher.stop()

    Usage (shared session):
        fetcher = ExternalDataFetcher(session=existing_session)
        fg = await fetcher.get_fear_greed_index()
        # do NOT call start() / stop() — lifecycle is owned by the caller
    """

    def __init__(
        self,
        cryptopanic_api_key: str = "",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._cryptopanic_key: str = cryptopanic_api_key
        self._session: Optional[aiohttp.ClientSession] = session
        self._own_session: bool = session is None  # True = we created it, we close it

        # Fear & Greed cache — index updates once per day; 1-hour TTL is plenty
        self._fear_greed_cache: Optional[FearGreedData] = None
        self._fear_greed_cache_ts: float = 0.0
        self._FEAR_GREED_TTL: float = 3600.0

        # News cache keyed by base coin symbol (e.g. "BTC")
        # {coin: (fetched_timestamp, list[NewsItem])}
        self._news_cache: Dict[str, Tuple[float, List[NewsItem]]] = {}
        self._NEWS_TTL: float = 1800.0  # 30 minutes

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create an aiohttp session if none was provided externally."""
        if self._own_session:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "CryptoTradingBot/2.0"},
                timeout=aiohttp.ClientTimeout(total=10),
            )
        logger.info("ExternalDataFetcher started (cryptopanic_key={})",
                    "set" if self._cryptopanic_key else "not set")

    async def stop(self) -> None:
        """Close the aiohttp session if we own it."""
        if self._own_session and self._session:
            await self._session.close()
            self._session = None
        logger.info("ExternalDataFetcher stopped")

    async def __aenter__(self) -> "ExternalDataFetcher":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    # ── Fear & Greed Index ────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=1, max=3),
        reraise=False,
    )
    async def get_fear_greed_index(self) -> Optional[FearGreedData]:
        """
        Fetch the Crypto Fear & Greed Index from alternative.me.

        Free, no API key required.  The index is updated once per day so
        responses are cached for 1 hour to avoid unnecessary network calls.

        Returns:
            FearGreedData with value 0–100 and a text classification, or None
            on network failure (stale cache is returned if available).

        Value bands:
            0–25   Extreme Fear
            26–46  Fear
            47–54  Neutral
            55–75  Greed
            76–100 Extreme Greed
        """
        now = time.time()
        if self._fear_greed_cache and (now - self._fear_greed_cache_ts) < self._FEAR_GREED_TTL:
            return self._fear_greed_cache

        try:
            async with self._session.get(_FEAR_GREED_URL) as resp:
                if resp.status != 200:
                    logger.warning("Fear & Greed API returned HTTP {}", resp.status)
                    return self._fear_greed_cache  # stale cache preferred over None
                data = await resp.json(content_type=None)
                entry = data["data"][0]
                fg = FearGreedData(
                    value=int(entry["value"]),
                    classification=entry["value_classification"],
                    timestamp=int(entry.get("timestamp", int(now))),
                )
                self._fear_greed_cache = fg
                self._fear_greed_cache_ts = now
                logger.debug("Fear & Greed: {} ({})", fg.value, fg.classification)
                return fg
        except Exception as exc:
            logger.warning("Fear & Greed fetch failed: {}", exc)
            return self._fear_greed_cache  # return stale cache if available

    # ── News sentiment (CryptoPanic) ──────────────────────────────────────────

    async def get_news_sentiment(self, symbol: str) -> List[NewsItem]:
        """
        Fetch recent news articles for a trading symbol from CryptoPanic.

        Requires CRYPTOPANIC_API_KEY in .env (free tier: ~1 000 calls/month).
        Returns an empty list immediately when no key is configured — this is
        intentional so callers do not need to guard against missing keys.

        Free tier capacity note:
            30 scans/day × 30 top symbols = 900 calls/day  → borderline.
            Recommended usage: only call for the top-10 coins by score.

        Args:
            symbol: Exchange pair symbol, e.g. "BTCUSDT" or "ETHUSDT".

        Returns:
            List of NewsItem, most recent first.  Empty list on error or if
            no API key is configured.
        """
        if not self._cryptopanic_key:
            return []  # not configured — graceful no-op

        # Strip quote currency to get the base coin (BTCUSDT → BTC)
        coin = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")

        now = time.time()
        if coin in self._news_cache:
            cached_ts, cached_items = self._news_cache[coin]
            if (now - cached_ts) < self._NEWS_TTL:
                return cached_items

        try:
            params = {
                "auth_token": self._cryptopanic_key,
                "currencies": coin,
                "filter": "hot",    # hot = high-engagement posts
                "public": "true",
                "limit": 10,
            }
            async with self._session.get(_CRYPTOPANIC_URL, params=params) as resp:
                if resp.status != 200:
                    logger.warning("CryptoPanic returned HTTP {} for {}", resp.status, symbol)
                    return []
                data = await resp.json(content_type=None)
                items: List[NewsItem] = []
                for post in data.get("results", []):
                    # Derive sentiment from community up/down votes
                    votes = post.get("votes", {})
                    bullish = (votes.get("positive") or 0) + (votes.get("liked") or 0)
                    bearish = (votes.get("negative") or 0) + (votes.get("disliked") or 0)
                    if bullish > bearish:
                        sentiment = "positive"
                    elif bearish > bullish:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                    currencies = [c["code"] for c in post.get("currencies", [])]
                    items.append(NewsItem(
                        title=post.get("title", ""),
                        url=post.get("url", ""),
                        source=post.get("domain", ""),
                        published_at=post.get("published_at", ""),
                        sentiment=sentiment,
                        currencies=currencies,
                    ))
                self._news_cache[coin] = (now, items)
                logger.debug("CryptoPanic: {} articles fetched for {}", len(items), coin)
                return items
        except Exception as exc:
            logger.warning("CryptoPanic fetch failed for {}: {}", symbol, exc)
            return []

    async def get_news_score(self, symbol: str) -> float:
        """
        Compute a normalised news sentiment score for a symbol.

        Averages the sentiment across all fetched articles:
            positive = +1.0, negative = -1.0, neutral = 0.0

        Returns:
            float in [-1.0, +1.0], or 0.0 if no news or key not configured.
        """
        items = await self.get_news_sentiment(symbol)
        if not items:
            return 0.0
        score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores = [score_map.get(item.sentiment, 0.0) for item in items]
        return sum(scores) / len(scores)

    # ── Liquidation clusters ──────────────────────────────────────────────────

    async def get_binance_liquidations(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[LiquidationCluster]:
        """
        Fetch recent forced-liquidation orders from Binance Futures (public endpoint).

        Endpoint: GET /fapi/v1/allForceOrders — no API key required.
        Returns up to 100 orders for the last 24 h, then groups them into 0.5%
        price bands to reveal where cascading liquidations are concentrated.

        These clusters are useful as:
          - LONG trade fuel: SHORT clusters above price (trapped shorts squeezed up)
          - SHORT trade fuel: LONG clusters below price (trapped longs forced down)

        Args:
            symbol: Exchange pair, e.g. "BTCUSDT".
            limit:  Number of raw liquidation orders to fetch (max 100).

        Returns:
            List of LiquidationCluster sorted by total_usd descending, or []
            on network failure.
        """
        try:
            params: Dict = {"symbol": symbol, "limit": min(limit, 100)}
            async with self._session.get(_BINANCE_LIQ_URL, params=params) as resp:
                if resp.status != 200:
                    logger.debug("Liquidation fetch HTTP {} for {}", resp.status, symbol)
                    return []
                orders = await resp.json(content_type=None)
                clusters = self._cluster_liquidations(orders)
                logger.debug(
                    "Liquidations: {} clusters for {} (raw orders: {})",
                    len(clusters), symbol, len(orders),
                )
                return clusters
        except Exception as exc:
            logger.debug("Liquidation fetch failed for {}: {}", symbol, exc)
            return []

    def _cluster_liquidations(self, orders: List[Dict]) -> List[LiquidationCluster]:
        """
        Aggregate raw liquidation orders into 0.5 % price-band clusters.

        Binance side convention:
            order side = BUY  → the closed position was SHORT  → LiquidationCluster.side = "SHORT"
            order side = SELL → the closed position was LONG   → LiquidationCluster.side = "LONG"

        Only clusters with total notional >= $1 000 are retained to remove noise.

        Args:
            orders: Raw list of order dicts from the Binance allForceOrders endpoint.

        Returns:
            Up to _LIQ_CLUSTER_MAX clusters sorted by total_usd descending.
        """
        if not orders:
            return []

        # Accumulate {(price_band, side): total_usd_notional}
        cluster_map: Dict[Tuple[float, str], float] = {}

        for order in orders:
            try:
                # Some orders carry averagePrice instead of price (already-filled)
                price = float(order.get("price") or order.get("averagePrice") or 0)
                qty = float(order.get("origQty") or 0)
                if price <= 0 or qty <= 0:
                    continue

                # "BUY" forced order = exchange bought back a SHORT position (SHORT liq)
                exchange_side = order.get("side", "BUY")
                liq_side = "SHORT" if exchange_side == "BUY" else "LONG"

                # Round to nearest 0.5 % price band for clustering
                band = round(price * 200) / 200
                usd_val = price * qty

                key = (band, liq_side)
                cluster_map[key] = cluster_map.get(key, 0.0) + usd_val
            except (ValueError, TypeError):
                continue

        # Filter small clusters and convert to dataclass list
        result: List[LiquidationCluster] = [
            LiquidationCluster(price=price, total_usd=total_usd, side=side)
            for (price, side), total_usd in cluster_map.items()
            if total_usd >= _LIQ_CLUSTER_MIN_USD
        ]

        result.sort(key=lambda x: x.total_usd, reverse=True)
        return result[:_LIQ_CLUSTER_MAX]

    def get_nearest_liquidation_level(
        self,
        current_price: float,
        clusters: List[LiquidationCluster],
        direction: str,
        max_distance_pct: float = 3.0,
    ) -> Optional[float]:
        """
        Find the nearest relevant liquidation cluster ahead of the trade direction.

        Relevance rules:
          LONG trade  → look for SHORT clusters ABOVE current price
                        (trapped shorts will be squeezed, providing momentum)
          SHORT trade → look for LONG clusters BELOW current price
                        (trapped longs will be forced out, providing momentum)

        Only clusters within max_distance_pct of current_price are considered.

        Args:
            current_price:    Latest traded price of the symbol.
            clusters:         Output of get_binance_liquidations().
            direction:        "LONG" or "SHORT".
            max_distance_pct: Clusters beyond this % distance are ignored.

        Returns:
            Distance to the nearest relevant cluster as a percentage of
            current_price, or None if no qualifying cluster exists.
        """
        if not clusters:
            return None

        target_side = "SHORT" if direction == "LONG" else "LONG"
        best_distance: Optional[float] = None

        for cluster in clusters:
            if cluster.side != target_side:
                continue

            dist_pct = abs(cluster.price - current_price) / current_price * 100
            if dist_pct > max_distance_pct:
                continue

            # Directional position check — cluster must be ahead of the trade
            if direction == "LONG" and cluster.price <= current_price:
                continue
            if direction == "SHORT" and cluster.price >= current_price:
                continue

            if best_distance is None or dist_pct < best_distance:
                best_distance = dist_pct

        return best_distance
