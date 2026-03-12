"""
scanners/pump_detector.py — Detect pump signals using volume, price, and OI.

Scoring breakdown (max 6 points):
  +1  volume spike  ≥ 2× 20-bar average
  +1  strong price move  ≥ 3 % in last candle
  +1  OI increasing  ≥ 0.5 % change
  +1  RSI < 70 (not overbought on entry)
  +1  price above 20-EMA (trend confirmation)
  +1  funding rate neutral / negative (room to run long)
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
import ta
from loguru import logger

from data_engine.market_data import MarketDataEngine


class PumpDetector:
    def __init__(self, engine: MarketDataEngine) -> None:
        self._engine = engine
        self._oi_cache: dict = {}

    async def evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        ticker: dict,
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

            # ── Price momentum ───────────────────────────────────────────
            pct_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            if abs(pct_change) >= 3.0:
                score += 1
                signals.append("strong_price_move")

            # ── RSI not overbought ────────────────────────────────────────
            # Allow oversold entries (RSI < 30 is a valid reversal buy setup)
            rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            rsi = rsi_series.iloc[-1]
            if rsi <= 75:
                score += 1
                signals.append("rsi_healthy")

            # ── Above 20-EMA ─────────────────────────────────────────────
            ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            if close.iloc[-1] > ema20.iloc[-1]:
                score += 1
                signals.append("above_ema20")

            # ── Open interest increasing (real OI delta check) ────────────
            # Use -1 as sentinel to distinguish "never seen" from "seen with OI=0"
            try:
                oi_now = await self._engine.get_open_interest_binance(symbol)
                oi_prev = self._oi_cache.get(symbol, -1)
                self._oi_cache[symbol] = oi_now
                if oi_prev < 0:
                    # First visit — award point if coin has active open interest
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

            # ── Funding rate check ───────────────────────────────────────
            try:
                funding = await self._engine.get_funding_rate_binance(symbol)
                if funding <= 0.001:   # neutral or negative
                    score += 1
                    signals.append("low_funding_rate")
            except Exception:
                pass

        except Exception as exc:
            logger.debug("PumpDetector error ({}): {}", symbol, exc)

        return score, signals
