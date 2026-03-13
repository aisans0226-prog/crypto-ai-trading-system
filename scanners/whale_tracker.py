"""
scanners/whale_tracker.py — Detect whale accumulation/distribution via order-book & volume.

Scoring breakdown (max 4 points):
  +1  order-book imbalance in trade direction (buy-side for LONG, sell-side for SHORT)
  +2  absorption pattern (large volume, small price move = smart money positioning)
  +1  consecutive candles in trade direction (3+ green for LONG, 3+ red for SHORT)
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
        direction: str = "LONG",
    ) -> Tuple[int, List[str]]:
        score = 0
        signals: List[str] = []

        try:
            # ── Order book imbalance (100 levels captures deep whale orders) ──
            try:
                book = await self._engine.get_order_book_binance(symbol, limit=100)
                bid_volume = sum(q for _, q in book["bids"])
                ask_volume = sum(q for _, q in book["asks"])
                if direction == "LONG" and ask_volume > 0:
                    # Buyers dominating: bid pressure > ask pressure
                    if bid_volume / ask_volume >= 1.5:
                        score += 1
                        signals.append("buy_side_imbalance")
                elif direction == "SHORT" and bid_volume > 0:
                    # Sellers dominating: ask pressure > bid pressure
                    if ask_volume / bid_volume >= 1.5:
                        score += 1
                        signals.append("sell_side_imbalance")
            except Exception:
                pass

            # ── Absorption pattern (high volume, low price change = smart money) ──
            # Direction-agnostic: whales accumulate (long) OR distribute (short) silently
            recent = df.tail(5)
            avg_vol = df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
            bar_count = 0
            if not pd.isna(avg_vol) and avg_vol > 0:
                for _, row in recent.iterrows():
                    low = max(float(row["low"]), 1e-10)  # guard div-by-zero on malformed data
                    candle_range_pct = (row["high"] - low) / low * 100
                    if row["volume"] > avg_vol * 1.5 and candle_range_pct < 1.5:
                        bar_count += 1
            if bar_count >= 2:
                score += 2
                signals.append("whale_absorption")

            # ── Consecutive directional candles ───────────────────────────
            closes = df["close"].tail(5).values
            opens = df["open"].tail(5).values
            if direction == "LONG":
                streak = sum(1 for c, o in zip(closes[-3:], opens[-3:]) if c > o)
                if streak >= 3:
                    score += 1
                    signals.append("bullish_streak")
            else:
                streak = sum(1 for c, o in zip(closes[-3:], opens[-3:]) if c < o)
                if streak >= 3:
                    score += 1
                    signals.append("bearish_streak")

        except Exception as exc:
            logger.debug("WhaleTracker error ({}): {}", symbol, exc)

        return score, signals
