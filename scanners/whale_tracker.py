"""
scanners/whale_tracker.py — Detect whale accumulation via order-book & volume.

Scoring breakdown (max 4 points):
  +1  large buy-side order book imbalance
  +2  absorption pattern (large volume, small price move = accumulation)
  +1  consecutive bullish candles  ≥ 3
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from data_engine.market_data import MarketDataEngine


class WhaleTracker:
    def __init__(self, engine: MarketDataEngine) -> None:
        self._engine = engine

    async def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Tuple[int, List[str]]:
        score = 0
        signals: List[str] = []

        try:
            # ── Order book imbalance ─────────────────────────────────────
            try:
                book = await self._engine.get_order_book_binance(symbol, limit=20)
                bid_volume = sum(q for _, q in book["bids"])
                ask_volume = sum(q for _, q in book["asks"])
                if ask_volume > 0:
                    imbalance = bid_volume / ask_volume
                    if imbalance >= 1.5:
                        score += 1
                        signals.append("buy_side_imbalance")
            except Exception:
                pass

            # ── Accumulation pattern (high volume, low price change) ──────
            recent = df.tail(5)
            avg_vol = df["volume"].rolling(20).mean().iloc[-1]
            bar_count = 0
            for _, row in recent.iterrows():
                candle_range_pct = (row["high"] - row["low"]) / row["low"] * 100
                if row["volume"] > avg_vol * 1.5 and candle_range_pct < 1.5:
                    bar_count += 1
            if bar_count >= 2:
                score += 2
                signals.append("whale_accumulation")

            # ── Consecutive bullish candles ───────────────────────────────
            closes = df["close"].tail(5).values
            opens = df["open"].tail(5).values
            bullish_streak = sum(1 for c, o in zip(closes[-3:], opens[-3:]) if c > o)
            if bullish_streak >= 3:
                score += 1
                signals.append("strong_bullish_streak")

        except Exception as exc:
            logger.debug("WhaleTracker error ({}): {}", symbol, exc)

        return score, signals
