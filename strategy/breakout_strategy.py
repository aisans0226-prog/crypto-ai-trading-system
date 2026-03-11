"""
strategy/breakout_strategy.py — Price breakout strategy.

Entry: price closes above resistance (highest high of last N candles) with volume confirmation.
SL: below breakout candle low.
TP: SL distance × 2 (minimum 1:2 RR).
"""
from typing import Optional
import pandas as pd
import ta
from loguru import logger

from config import settings
from strategy.trend_strategy import TradeSetup


class BreakoutStrategy:
    NAME = "BREAKOUT"
    LOOKBACK = 48       # candles (~12 hours on 15m)

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSetup]:
        try:
            if len(df) < self.LOOKBACK + 5:
                return None

            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            # Resistance / support over lookback window
            resistance = high.iloc[-(self.LOOKBACK + 1):-1].max()
            support = low.iloc[-(self.LOOKBACK + 1):-1].min()

            avg_vol = volume.rolling(20).mean().iloc[-1]
            last_vol = volume.iloc[-1]
            volume_confirm = last_vol >= avg_vol * 1.5

            atr = ta.volatility.AverageTrueRange(
                high=high, low=low, close=close, window=14
            ).average_true_range().iloc[-1]

            c = close.iloc[-1]

            # ── Upside breakout ───────────────────────────────────────────
            if c > resistance * 1.001 and volume_confirm:
                stop = low.iloc[-1] - 0.5 * atr
                tp = c + (c - stop) * settings.min_risk_reward_ratio
                rr = (tp - c) / (c - stop)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="LONG",
                        entry_price=c, stop_loss=round(stop, 6),
                        take_profit=round(tp, 6),
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

            # ── Downside breakout ─────────────────────────────────────────
            if c < support * 0.999 and volume_confirm:
                stop = high.iloc[-1] + 0.5 * atr
                tp = c - (stop - c) * settings.min_risk_reward_ratio
                rr = (c - tp) / (stop - c)
                if rr >= settings.min_risk_reward_ratio:
                    return TradeSetup(
                        symbol=symbol, direction="SHORT",
                        entry_price=c, stop_loss=round(stop, 6),
                        take_profit=round(tp, 6),
                        leverage=settings.max_leverage,
                        risk_reward=round(rr, 2),
                    )

        except Exception as exc:
            logger.debug("BreakoutStrategy error ({}): {}", symbol, exc)

        return None

    def regime_fit(self, df: pd.DataFrame) -> float:
        """Higher when price is close to resistance/support with rising volume."""
        try:
            close  = df["close"]
            high   = df["high"]
            low    = df["low"]
            volume = df["volume"]

            resistance = high.iloc[-(self.LOOKBACK + 1):-1].max()
            support    = low.iloc[-(self.LOOKBACK + 1):-1].min()
            c          = close.iloc[-1]

            dist_to_res = abs(c - resistance) / max(c, 1e-10)
            dist_to_sup = abs(c - support)    / max(c, 1e-10)
            # Proximity score: 1 = at the level, 0 = >5 % away
            proximity   = 1.0 - min(min(dist_to_res, dist_to_sup), 0.05) / 0.05

            avg_vol   = volume.rolling(20).mean().iloc[-1]
            last_vol  = volume.iloc[-1]
            vol_ratio = min(last_vol / max(avg_vol, 1), 3.0) / 3.0

            return round(proximity * 0.60 + vol_ratio * 0.40, 4)
        except Exception:
            return 0.5
