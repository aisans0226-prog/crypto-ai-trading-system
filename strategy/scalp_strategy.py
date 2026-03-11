"""
strategy/scalp_strategy.py — Bollinger Band squeeze + breakout scalp.

A "squeeze" is detected when the BB width drops below the 25th percentile of
the last 50 candles.  The very next breakout candle that closes outside the
band with volume ≥ 1.3× average is the entry signal.

SL: 0.8× ATR from entry.
TP: 1.6× ATR from entry (≥ 1:2 RR by construction).
"""
from typing import Optional

import numpy as np
import pandas as pd
import ta
from loguru import logger

from config import settings
from strategy.trend_strategy import TradeSetup


class ScalpStrategy:
    NAME = "SCALP_BB"
    SQUEEZE_WINDOW = 50      # candles used for percentile baseline
    SQUEEZE_PCT    = 25      # squeeze = below 25th-percentile width

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSetup]:
        try:
            if len(df) < self.SQUEEZE_WINDOW + 5:
                return None

            close  = df["close"]
            high   = df["high"]
            low    = df["low"]
            volume = df["volume"]

            bb    = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            upper = bb.bollinger_hband()
            lower = bb.bollinger_lband()
            width = (upper - lower) / close.replace(0, float("nan"))

            # Previous candle must have been in squeeze
            history   = width.iloc[-self.SQUEEZE_WINDOW:-1].dropna()
            if len(history) < 20:
                return None
            threshold = float(np.percentile(history, self.SQUEEZE_PCT))
            if width.iloc[-2] > threshold:
                return None

            # Current candle: volume spike required
            avg_vol  = volume.rolling(20).mean().iloc[-1]
            last_vol = volume.iloc[-1]
            if last_vol < avg_vol * 1.3:
                return None

            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range().iloc[-1]
            c = close.iloc[-1]

            # ── LONG breakout: close above upper band ─────────────────────────
            if c > upper.iloc[-1]:
                stop = c - 0.8 * atr
                tp   = c + 1.6 * atr
                rr   = (tp - c) / max(c - stop, 1e-10)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="LONG",
                        entry_price=c, stop_loss=round(stop, 6), take_profit=round(tp, 6),
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

            # ── SHORT breakout: close below lower band ─────────────────────────
            if c < lower.iloc[-1]:
                stop = c + 0.8 * atr
                tp   = c - 1.6 * atr
                rr   = (c - tp) / max(stop - c, 1e-10)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="SHORT",
                        entry_price=c, stop_loss=round(stop, 6), take_profit=round(tp, 6),
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

        except Exception as exc:
            logger.debug("ScalpStrategy error ({}): {}", symbol, exc)
        return None

    def regime_fit(self, df: pd.DataFrame) -> float:
        """Higher when BB is currently squeezed (low-volatility pre-breakout regime)."""
        try:
            close = df["close"]
            bb    = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            width = (bb.bollinger_hband() - bb.bollinger_lband()) / close.replace(0, float("nan"))
            history = width.iloc[-self.SQUEEZE_WINDOW:-1].dropna()
            if len(history) < 20:
                return 0.5
            current_w = width.iloc[-1]
            # pct_rank = fraction of history values below current width
            # Low rank → tighter → better for scalp
            pct_rank = float((history < current_w).mean())
            return round(1.0 - pct_rank, 4)
        except Exception:
            return 0.5
