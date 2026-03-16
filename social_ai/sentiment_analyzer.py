"""
social_ai/sentiment_analyzer.py — Crypto social sentiment scoring.

Sources:
  - Twitter/X (via Bearer Token)
  - Reddit (public JSON API — no auth required)
  - VADER rule-based fallback

Output: sentiment score  -1.0 → +1.0
Signal contribution: +1 if score ≥ 0.3
"""
import asyncio
import re
import time
from typing import List, Optional, Tuple
import aiohttp
from loguru import logger

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from config import settings


_SENTIMENT_CACHE_TTL = 600  # reuse per-symbol result for 10 min across scan cycles
_REDDIT_CIRCUIT_BREAK_AFTER = 3   # consecutive failures before disabling Reddit
_REDDIT_COOLDOWN_SECS       = 900  # 15 min cooldown after circuit trip


class SentimentAnalyzer:
    CRYPTO_TERMS = [
        "pump", "moon", "bullish", "buy", "long",
        "all-time high", "ath", "breakout", "rally",
    ]

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self._session: Optional[aiohttp.ClientSession] = None
        # Per-symbol result cache {symbol: (timestamp, score, signals)}
        self._cache: dict = {}
        # Reddit circuit-breaker state
        self._reddit_fail_streak = 0
        self._reddit_disabled_until = 0.0

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ────────────────────────────────────────────────────────
    async def get_sentiment_score(self, symbol: str) -> Tuple[float, List[str]]:
        """Return (composite_score, list_of_signals)."""
        # Check per-symbol TTL cache to avoid repeated slow fetches across cycles
        now = time.time()
        cached = self._cache.get(symbol)
        if cached and now - cached[0] < _SENTIMENT_CACHE_TTL:
            return cached[1], cached[2]

        coin = symbol.replace("USDT", "").replace("PERP", "").lower()

        scores = []
        texts: List[str] = []

        # Twitter
        if settings.twitter_bearer_token:
            tw_texts = await self._fetch_twitter(coin)
            texts.extend(tw_texts)

        # Reddit
        reddit_texts = await self._fetch_reddit(coin)
        texts.extend(reddit_texts)

        if not texts:
            self._cache[symbol] = (now, 0.0, [])
            return 0.0, []

        for text in texts:
            s = self._score_text(text)
            scores.append(s)

        avg = sum(scores) / len(scores) if scores else 0.0

        signals = []
        if avg >= 0.3:
            signals.append("positive_sentiment")
        elif avg <= -0.3:
            signals.append("negative_sentiment")

        result_score = round(avg, 4)
        self._cache[symbol] = (now, result_score, signals)
        return result_score, signals

    # ── Twitter fetch ─────────────────────────────────────────────────────
    async def _fetch_twitter(self, coin: str) -> List[str]:
        if not self._session or not settings.twitter_bearer_token:
            return []
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {settings.twitter_bearer_token}"}
        params = {
            "query": f"#{coin} OR ${coin.upper()} -is:retweet lang:en",
            "max_results": 20,
            "tweet.fields": "text",
        }
        try:
            async with self._session.get(url, headers=headers, params=params) as r:
                if r.status == 200:
                    data = await r.json()
                    return [t["text"] for t in data.get("data", [])]
        except Exception as exc:
            logger.debug("Twitter fetch error: {}", exc)
        return []

    # ── Reddit fetch (no API key needed) ─────────────────────────────────
    async def _fetch_reddit(self, coin: str) -> List[str]:
        if not self._session:
            return []
        # Circuit breaker: skip Reddit if it has been repeatedly failing
        now = time.time()
        if now < self._reddit_disabled_until:
            return []
        url = "https://www.reddit.com/search.json"
        params = {"q": coin, "sort": "new", "limit": 15, "t": "day"}
        headers = {"User-Agent": "CryptoBot/1.0"}
        try:
            async with self._session.get(url, params=params, headers=headers) as r:
                if r.status == 200:
                    data = await r.json()
                    posts = data.get("data", {}).get("children", [])
                    self._reddit_fail_streak = 0  # reset on success
                    return [
                        p["data"].get("title", "") + " " + p["data"].get("selftext", "")
                        for p in posts
                    ]
                # Non-200 (e.g. 429, 403) treated as failure
                self._reddit_fail_streak += 1
        except Exception as exc:
            logger.debug("Reddit fetch error: {}", exc)
            self._reddit_fail_streak += 1
        if self._reddit_fail_streak >= _REDDIT_CIRCUIT_BREAK_AFTER:
            self._reddit_disabled_until = time.time() + _REDDIT_COOLDOWN_SECS
            logger.info(
                "Reddit circuit-breaker tripped after {} failures — pausing {}s",
                self._reddit_fail_streak, _REDDIT_COOLDOWN_SECS,
            )
            self._reddit_fail_streak = 0
        return []

    # ── Scoring ───────────────────────────────────────────────────────────
    def _score_text(self, text: str) -> float:
        if self._vader:
            score = self._vader.polarity_scores(text)
            return score["compound"]
        # simple heuristic fallback
        text_lower = text.lower()
        pos = sum(1 for w in ["bull", "moon", "pump", "long", "buy"] if w in text_lower)
        neg = sum(1 for w in ["bear", "dump", "short", "sell", "rekt"] if w in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total
