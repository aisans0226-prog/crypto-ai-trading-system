"""
scanners/market_scanner.py — Master scanner that coordinates all sub-scanners
Scans up to 500 coins in parallel and aggregates scores.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from config import settings
from data_engine.market_data import MarketDataEngine
from scanners.pump_detector import PumpDetector
from scanners.whale_tracker import WhaleTracker
from scanners.memecoin_scanner import MemecoinScanner


@dataclass
class SignalResult:
    symbol: str
    score: int = 0
    direction: str = "LONG"   # LONG | SHORT
    signals: List[str] = field(default_factory=list)
    price: float = 0.0
    volume_24h: float = 0.0
    price_change_pct: float = 0.0
    open_interest: float = 0.0
    ai_prediction: float = 0.0
    confidence: float = 0.0

    @property
    def is_high_probability(self) -> bool:
        return self.score >= settings.effective_signal_score_threshold


class MarketScanner:
    """
    Orchestrates pump, whale, and memecoin scanners across all available symbols.
    Produces a ranked list of SignalResult objects.
    """

    def __init__(self, data_engine: MarketDataEngine) -> None:
        self._engine = data_engine
        self._pump = PumpDetector(data_engine)
        self._whale = WhaleTracker(data_engine)
        self._meme = MemecoinScanner(data_engine)
        self._last_klines: Dict[str, "pd.DataFrame"] = {}  # full bulk-fetch cache for ranker

    def get_last_klines(self) -> Dict[str, "pd.DataFrame"]:
        """Return the klines map from the most recent scan (all volume-filtered coins)."""
        return dict(self._last_klines)

    def get_last_funding_cache(self) -> Dict[str, float]:
        """Return the bulk-fetched funding rates from the most recent scan."""
        return dict(self._pump._funding_bulk)

    # ── Public API ────────────────────────────────────────────────────────
    async def scan_all(self) -> List[SignalResult]:
        """Full market scan. Returns signals sorted by score descending."""
        logger.info("Starting full market scan …")

        symbols = await self._engine.get_binance_symbols()
        symbols = symbols[: settings.max_coins_to_scan]

        # Filter by minimum 24h volume
        tickers = await self._engine.get_all_tickers_binance()
        high_volume = {
            t["symbol"]
            for t in tickers
            if float(t.get("quoteVolume", 0)) >= settings.effective_min_volume_usdt
        }
        symbols = [s for s in symbols if s in high_volume]
        logger.info("Symbols after volume filter: {}", len(symbols))

        # Bulk OHLCV fetch + bulk OI + bulk funding rates — all in parallel
        klines_map, oi_map, funding_map = await asyncio.gather(
            self._engine.bulk_fetch_klines(symbols, interval="15m", limit=200),
            self._engine.bulk_fetch_open_interests(symbols),
            self._engine.bulk_fetch_funding_rates(),
        )
        self._last_klines = dict(klines_map)  # cache for CoinRanker

        # Prime pump-detector caches so per-symbol evaluate() skips HTTP calls
        self._pump.update_caches(oi_map=oi_map, funding_map=funding_map)

        # Parallel sub-scanner execution
        semaphore = asyncio.Semaphore(100)

        async def scan_symbol(sym: str) -> Optional[SignalResult]:
            async with semaphore:
                df = klines_map.get(sym)
                if df is None or len(df) < 50:
                    return None
                try:
                    result = await self._evaluate_symbol(sym, df, tickers)
                    return result
                except Exception as exc:
                    logger.debug("Scan error {}: {}", sym, exc)
                    return None

        tasks = [scan_symbol(s) for s in klines_map]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None and r.score > 0]
        results.sort(key=lambda x: x.score, reverse=True)

        high_prob = [r for r in results if r.is_high_probability]
        logger.info(
            "Scan complete — {} signals, {} high-probability",
            len(results), len(high_prob),
        )
        return results

    # ── Internal evaluation ───────────────────────────────────────────────
    async def _evaluate_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        tickers: List[dict],
    ) -> SignalResult:
        ticker = next((t for t in tickers if t["symbol"] == symbol), {})
        price = float(ticker.get("lastPrice", df["close"].iloc[-1]))
        volume_24h = float(ticker.get("quoteVolume", 0))
        price_change_pct = float(ticker.get("priceChangePercent", 0))

        # ── Direction from EMA stack (technical basis, not 24h price change) ──
        close = df["close"].astype(float)
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        last_close = close.iloc[-1]
        if last_close > ema20 and ema20 > ema50:
            direction = "LONG"
        elif last_close < ema20 and ema20 < ema50:
            direction = "SHORT"
        elif price_change_pct >= 0:
            direction = "LONG"   # fallback to momentum when EMA is flat
        else:
            direction = "SHORT"

        result = SignalResult(
            symbol=symbol,
            price=price,
            volume_24h=volume_24h,
            price_change_pct=price_change_pct,
            direction=direction,
        )

        # Run all 3 sub-scanners in parallel — ~3x faster than sequential
        (pump_score, pump_signals), (whale_score, whale_signals), (meme_score, meme_signals) = \
            await asyncio.gather(
                self._pump.evaluate(symbol, df, ticker, direction),
                self._whale.evaluate(symbol, df, direction),
                self._meme.evaluate(symbol, df, ticker),
            )

        result.score = pump_score + whale_score + meme_score
        result.signals = pump_signals + whale_signals + meme_signals

        return result
