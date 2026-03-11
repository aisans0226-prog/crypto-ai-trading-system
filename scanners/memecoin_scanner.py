"""
scanners/memecoin_scanner.py — Detect early-stage memecoin / narrative pumps.

Scoring breakdown (max 3 points):
  +1  24h price change  ≥ 10 %
  +1  volume / market-cap proxy surge
  +1  breakout above recent 48-candle high
"""
from typing import List, Tuple
import pandas as pd
from loguru import logger

from data_engine.market_data import MarketDataEngine

# Known meme / narrative coin patterns
MEME_KEYWORDS = {
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF",
    "MEME", "TURBO", "BABYDOGE", "ELON", "POGAI",
}


class MemecoinScanner:
    def __init__(self, engine: MarketDataEngine) -> None:
        self._engine = engine

    async def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        ticker: dict,
    ) -> Tuple[int, List[str]]:
        score = 0
        signals: List[str] = []

        try:
            base_asset = symbol.replace("USDT", "").replace("PERP", "")
            is_meme = any(k in base_asset for k in MEME_KEYWORDS)

            price_change_pct = float(ticker.get("priceChangePercent", 0))
            volume_24h = float(ticker.get("quoteVolume", 0))

            # ── Strong 24h move ──────────────────────────────────────────
            if abs(price_change_pct) >= 10.0:
                score += 1
                signals.append("large_24h_move")

            # ── Volume surge above 5-day average proxy ───────────────────
            avg_vol = df["volume"].rolling(96).mean().iloc[-1]  # 96 × 15m ≈ 1 day
            last_vol = df["volume"].iloc[-1]
            if avg_vol > 0 and last_vol > avg_vol * 3.0:
                score += 1
                signals.append("volume_surge")

            # ── Breakout above 48-candle high ────────────────────────────
            recent_high = df["high"].iloc[-49:-1].max()
            if df["close"].iloc[-1] > recent_high:
                score += 1
                signals.append("breakout")

            # ── Meme bonus ───────────────────────────────────────────────
            if is_meme and score >= 2:
                score += 1
                signals.append("meme_narrative")

        except Exception as exc:
            logger.debug("MemecoinScanner error ({}): {}", symbol, exc)

        return score, signals
