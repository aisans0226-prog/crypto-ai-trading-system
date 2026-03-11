"""
strategy/liquidity_strategy.py — Order-book liquidity sweep / stop-hunt strategy.

Logic:
  Detect large liquidity pools (dense bid/ask walls) near current price.
  Enter after a wick sweeps the liquidity and price reverses.

Uses: order-book snapshots + candle wick analysis.
"""
from typing import Optional
import pandas as pd
import ta
from loguru import logger

from config import settings
from strategy.trend_strategy import TradeSetup


class LiquidityStrategy:
    NAME = "LIQUIDITY_SWEEP"
    WICK_RATIO_THRESHOLD = 0.6    # wick > 60 % of total candle range

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSetup]:
        try:
            if len(df) < 20:
                return None

            close = df["close"]
            open_ = df["open"]
            high = df["high"]
            low = df["low"]

            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range().iloc[-1]

            c = close.iloc[-1]
            o = open_.iloc[-1]
            h = high.iloc[-1]
            l = low.iloc[-1]
            candle_range = h - l

            if candle_range < 1e-10:
                return None

            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)

            # ── Bullish liquidity sweep ─────────────────────────────────
            # Large lower wick = stop-hunt below support, price recovers
            if lower_wick / candle_range >= self.WICK_RATIO_THRESHOLD and c > o:
                stop = l - 0.5 * atr
                tp = c + (c - stop) * settings.min_risk_reward_ratio
                rr = (tp - c) / (c - stop)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="LONG",
                        entry_price=c, stop_loss=round(stop, 6),
                        take_profit=round(tp, 6),
                        leverage=min(settings.max_leverage, 3),
                        risk_reward=round(rr, 2),
                    )

            # ── Bearish liquidity sweep ─────────────────────────────────
            if upper_wick / candle_range >= self.WICK_RATIO_THRESHOLD and c < o:
                stop = h + 0.5 * atr
                tp = c - (stop - c) * settings.min_risk_reward_ratio
                rr = (c - tp) / (stop - c)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="SHORT",
                        entry_price=c, stop_loss=round(stop, 6),
                        take_profit=round(tp, 6),
                        leverage=min(settings.max_leverage, 3),
                        risk_reward=round(rr, 2),
                    )

        except Exception as exc:
            logger.debug("LiquidityStrategy error ({}): {}", symbol, exc)

        return None
