"""
strategy/mean_reversion_strategy.py — RSI + Bollinger Band mean reversion.

LONG:  RSI < 32 AND close ≤ BB lower band AND current candle is bullish (bounce).
SHORT: RSI > 68 AND close ≥ BB upper band AND current candle is bearish (reversal).

SL:  1.5× ATR beyond the wick extreme, capped at 3× ATR from entry.
TP:  BB middle band (SMA-20); enforced to satisfy min_risk_reward_ratio.
"""
from typing import Optional
import pandas as pd
import ta
from loguru import logger

from config import settings
from strategy.trend_strategy import TradeSetup


class MeanReversionStrategy:
    NAME = "MEAN_REVERSION"
    RSI_LONG  = 38  # widened from 32; catches broader oversold zone
    RSI_SHORT = 62  # widened from 68; catches broader overbought zone

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSetup]:
        try:
            if len(df) < 30:
                return None

            close = df["close"]
            high  = df["high"]
            low   = df["low"]

            rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            bb  = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_mid   = bb.bollinger_mavg()
            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()

            c       = close.iloc[-1]
            prev_c  = close.iloc[-2]
            atr_val = atr.iloc[-1]

            # ── LONG: oversold bounce off lower band ──────────────────────────
            if (
                rsi.iloc[-1] <= self.RSI_LONG
                and c <= bb_lower.iloc[-1] * 1.005   # within 0.5 % of lower band (from 0.2 %)
                and c > prev_c                        # bullish reversal candle
            ):
                stop = low.iloc[-1] - 1.5 * atr_val
                stop = max(stop, c - 3.0 * atr_val)
                tp   = bb_mid.iloc[-1]
                # Ensure minimum RR
                if tp - c < (c - stop) * settings.min_risk_reward_ratio:
                    tp = c + (c - stop) * settings.min_risk_reward_ratio
                rr = (tp - c) / max(c - stop, 1e-10)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="LONG",
                        entry_price=c, stop_loss=round(stop, 6), take_profit=round(tp, 6),
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

            # ── SHORT: overbought rejection off upper band ───────────────────
            if (
                rsi.iloc[-1] >= self.RSI_SHORT
                and c >= bb_upper.iloc[-1] * 0.995  # within 0.5 % of upper band (from 0.2 %)
                and c < prev_c                        # bearish reversal candle
            ):
                stop = high.iloc[-1] + 1.5 * atr_val
                stop = min(stop, c + 3.0 * atr_val)
                tp   = bb_mid.iloc[-1]
                if c - tp < (stop - c) * settings.min_risk_reward_ratio:
                    tp = c - (stop - c) * settings.min_risk_reward_ratio
                rr = (c - tp) / max(stop - c, 1e-10)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="SHORT",
                        entry_price=c, stop_loss=round(stop, 6), take_profit=round(tp, 6),
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

        except Exception as exc:
            logger.debug("MeanReversionStrategy error ({}): {}", symbol, exc)
        return None

    def regime_fit(self, df: pd.DataFrame) -> float:
        """Higher when RSI is extreme and BB is wide (strong extension from mean)."""
        try:
            close = df["close"]
            rsi   = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
            bb    = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            c     = close.iloc[-1]
            width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / max(c, 1e-10)

            # 0 = neutral RSI, 1 = at 0 or 100
            rsi_extremity = abs(rsi - 50) / 50.0
            # Scale: 3 % width = moderate extension → score 1.0
            bb_score = min(width / 0.03, 1.0)

            return round(0.60 * rsi_extremity + 0.40 * bb_score, 4)
        except Exception:
            return 0.5
