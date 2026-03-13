"""
scanners/pump_detector.py — Detect pump/dump signals using volume, price, and OI.

Scoring breakdown (max 6 points):
  +1  volume spike  ≥ 2× 20-bar average
  +1  strong price move  ≥ 3% in last candle (direction-aligned)
  +1  OI increasing  ≥ 0.5% change (or active on first visit)
  +1  RSI in healthy zone (LONG: ≤ 72; SHORT: ≥ 28)
  +1  price on correct side of EMA20 (direction-aligned)
  +1  funding rate supports direction (LONG: ≤ 0.001; SHORT: ≥ -0.001)
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import ta
from loguru import logger

from data_engine.market_data import MarketDataEngine


class PumpDetector:
    def __init__(self, engine: MarketDataEngine) -> None:
        self._engine = engine
        self._oi_cache: dict = {}      # persistent per-symbol OI from last cycle
        self._oi_bulk: dict = {}       # bulk pre-fetched OI for current scan
        self._funding_bulk: dict = {}  # bulk pre-fetched funding for current scan

    def update_caches(self, oi_map: Dict[str, float], funding_map: Dict[str, float]) -> None:
        """Called once per scan cycle with bulk-fetched data.
        Eliminates per-symbol HTTP calls for OI and funding rate.
        """
        self._oi_bulk = oi_map
        self._funding_bulk = funding_map

    async def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        ticker: dict,
        direction: str = "LONG",
    ) -> Tuple[int, List[str]]:
        score = 0
        signals: List[str] = []

        try:
            close = df["close"]
            volume = df["volume"]

            # ── Volume spike ─────────────────────────────────────────────
            avg_vol = volume.rolling(20).mean().iloc[-1]
            last_vol = volume.iloc[-1]
            if avg_vol > 0 and last_vol >= avg_vol * 2.0:
                score += 1
                signals.append("volume_spike")

            # ── Price momentum (direction-aligned) ───────────────────────
            pct_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            price_ok = (direction == "LONG" and pct_change >= 3.0) or \
                       (direction == "SHORT" and pct_change <= -3.0)
            if price_ok:
                score += 1
                signals.append("strong_price_move")

            # ── RSI directional gate ──────────────────────────────────────
            rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            rsi = rsi_series.iloc[-1]
            # LONG: not overbought; SHORT: not oversold; both avoid extremes
            if direction == "LONG" and rsi <= 72:
                score += 1
                signals.append("rsi_long_ok")
            elif direction == "SHORT" and rsi >= 28:
                score += 1
                signals.append("rsi_short_ok")

            # ── EMA20 alignment (direction-aware) ─────────────────────────
            ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            if direction == "LONG" and close.iloc[-1] > ema20.iloc[-1]:
                score += 1
                signals.append("above_ema20")
            elif direction == "SHORT" and close.iloc[-1] < ema20.iloc[-1]:
                score += 1
                signals.append("below_ema20")

            # ── Open interest — use bulk cache, fall back to REST ─────────
            try:
                oi_now = self._oi_bulk.get(symbol)
                if oi_now is None:
                    oi_now = await self._engine.get_open_interest_binance(symbol)
                oi_prev = self._oi_cache.get(symbol, -1)
                self._oi_cache[symbol] = oi_now
                if oi_prev < 0:
                    if oi_now > 0:
                        score += 1
                        signals.append("oi_active")
                elif oi_prev > 0 and oi_now > oi_prev:
                    oi_delta_pct = (oi_now - oi_prev) / oi_prev * 100
                    if oi_delta_pct >= 0.5:
                        score += 1
                        signals.append("oi_increase")
            except Exception:
                pass

            # ── Funding rate — use bulk cache, fall back to REST ──────────
            try:
                funding = self._funding_bulk.get(symbol)
                if funding is None:
                    funding = await self._engine.get_funding_rate_binance(symbol)
                # LONG: low funding (room to run); SHORT: negative/low funding also ok
                if direction == "LONG" and funding <= 0.001:
                    score += 1
                    signals.append("low_funding_rate")
                elif direction == "SHORT" and funding >= -0.001:
                    score += 1
                    signals.append("funding_supports_short")
            except Exception:
                pass

        except Exception as exc:
            logger.debug("PumpDetector error ({}): {}", symbol, exc)

        return score, signals
