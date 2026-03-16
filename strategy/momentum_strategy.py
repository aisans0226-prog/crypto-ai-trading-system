"""
strategy/momentum_strategy.py — Momentum burst strategy.

Entry: 2+ consecutive bullish candles (lowered from 3), price above all EMAs (9/20/50),
       RSI 50–80 (widened from 55–78), volume each candle ≥ 1.1× 20-period average (from 1.2×).

SL:  0.5× ATR below the first streak candle's open (capped at 2× ATR).
TP:  2.5× ATR above entry.
Direction: LONG only (burst is inherently directional upward).
"""
from typing import Optional
import pandas as pd
import ta
from loguru import logger

from config import settings
from strategy.trend_strategy import TradeSetup


class MomentumStrategy:
    NAME = "MOMENTUM"
    MIN_STREAK = 2  # lowered from 3; 2 consecutive bullish candles required

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSetup]:
        try:
            if len(df) < 60:
                return None

            close  = df["close"]
            open_  = df["open"]
            high   = df["high"]
            low    = df["low"]
            volume = df["volume"]

            # All MIN_STREAK most-recent candles must be bullish
            for i in range(-self.MIN_STREAK, 0):
                if close.iloc[i] <= open_.iloc[i]:
                    return None

            avg_vol = volume.rolling(20).mean().iloc[-1]
            for i in range(-self.MIN_STREAK, 0):
                if volume.iloc[i] < avg_vol * 1.1:  # lowered from 1.2×
                    return None

            ema9  = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
            ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
            rsi   = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            atr   = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()

            c = close.iloc[-1]

            if not (ema9.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] and c > ema9.iloc[-1]):
                return None
            if not (50 <= rsi.iloc[-1] <= 80):  # widened from 55–78
                return None

            atr_val = atr.iloc[-1]
            # SL below first streak candle's open, capped at 2× ATR from entry
            stop = open_.iloc[-self.MIN_STREAK] - 0.5 * atr_val
            stop = max(stop, c - 2.0 * atr_val)
            tp   = c + 2.5 * atr_val
            rr   = (tp - c) / max(c - stop, 1e-10)
            if rr < settings.min_risk_reward_ratio:
                return None

            return TradeSetup(
                symbol=symbol, direction="LONG",
                entry_price=c, stop_loss=round(stop, 6), take_profit=round(tp, 6),
                leverage=settings.max_leverage,
                risk_reward=round(rr, 2),
            )

        except Exception as exc:
            logger.debug("MomentumStrategy error ({}): {}", symbol, exc)
        return None

    def regime_fit(self, df: pd.DataFrame) -> float:
        """Higher when price is in strong momentum: ROC + volume surge + mild ADX."""
        try:
            close  = df["close"]
            high   = df["high"]
            low    = df["low"]
            volume = df["volume"]

            roc5 = (close.iloc[-1] - close.iloc[-6]) / max(close.iloc[-6], 1e-10)

            avg_vol    = volume.rolling(20).mean().iloc[-1]
            recent_vol = volume.iloc[-3:].mean()
            vol_ratio  = min(recent_vol / max(avg_vol, 1), 2.5) / 2.5

            adx       = ta.trend.ADXIndicator(high=high, low=low, close=close).adx().iloc[-1]
            adx_score = min(adx / 40.0, 1.0)

            roc_score = min(max(roc5 / 0.03, 0.0), 1.0)   # 3 % ROC = full score

            return round(0.35 * roc_score + 0.40 * vol_ratio + 0.25 * adx_score, 4)
        except Exception:
            return 0.5
