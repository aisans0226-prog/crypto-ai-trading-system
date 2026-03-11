"""
strategy/trend_strategy.py — EMA trend-following strategy.

Entry conditions (all must be true):
  - EMA9 > EMA20 > EMA50 (bullish alignment)
  - Close above EMA20
  - RSI 45–70
  - MACD histogram positive and rising
  - ADX > 25 (strong trend)

Returns: direction, entry, stop_loss, take_profit
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import ta
from loguru import logger

from config import settings


@dataclass
class TradeSetup:
    symbol: str
    direction: str          # LONG | SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    risk_reward: float


class TrendStrategy:
    NAME = "TREND_FOLLOW"

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSetup]:
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            ema9 = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
            ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
            ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
            rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            macd_obj = ta.trend.MACD(close=close)
            macd_hist = macd_obj.macd_diff()
            adx = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()
            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range()

            c = close.iloc[-1]

            # ── LONG condition ────────────────────────────────────────────
            if (
                ema9.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]
                and c > ema20.iloc[-1]
                and 45 <= rsi.iloc[-1] <= 70
                and macd_hist.iloc[-1] > 0
                and macd_hist.iloc[-1] > macd_hist.iloc[-2]
                and adx.iloc[-1] > 25
            ):
                atr_val = atr.iloc[-1]
                stop = c - 1.5 * atr_val
                tp = c + 3.0 * atr_val           # 1:2 minimum RR
                rr = (tp - c) / (c - stop)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="LONG",
                        entry_price=c, stop_loss=stop, take_profit=tp,
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

            # ── SHORT condition ───────────────────────────────────────────
            if (
                ema9.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1]
                and c < ema20.iloc[-1]
                and 30 <= rsi.iloc[-1] <= 55
                and macd_hist.iloc[-1] < 0
                and macd_hist.iloc[-1] < macd_hist.iloc[-2]
                and adx.iloc[-1] > 25
            ):
                atr_val = atr.iloc[-1]
                stop = c + 1.5 * atr_val
                tp = c - 3.0 * atr_val
                rr = (c - tp) / (stop - c)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="SHORT",
                        entry_price=c, stop_loss=stop, take_profit=tp,
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

        except Exception as exc:
            logger.debug("TrendStrategy error ({}): {}", symbol, exc)

        return None

    def regime_fit(self, df: pd.DataFrame) -> float:
        """Higher when ADX confirms a strong directional trend."""
        try:
            high  = df["high"]
            low   = df["low"]
            close = df["close"]
            adx   = ta.trend.ADXIndicator(high=high, low=low, close=close).adx().iloc[-1]
            if adx >= 35:
                return 0.95
            if adx >= 28:
                return 0.80
            if adx >= 22:
                return 0.60
            if adx >= 16:
                return 0.35
            return 0.15
        except Exception:
            return 0.5
