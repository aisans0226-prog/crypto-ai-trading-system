"""
ml_models/coin_ranker.py - AI-powered coin potential ranking.

Ranks all scanned coins by composite score:
  - Momentum regime (ROC, RSI)
  - Volatility regime (ATR, Bollinger)
  - Volume consistency
  - BTC correlation delta
  - Historical win-rate from SelfLearningEngine
  - ML ensemble prediction confidence
  - Trend alignment (EMA stack)
"""
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import ta
from loguru import logger

from ml_models.self_learning import SelfLearningEngine


@dataclass
class CoinRank:
    symbol: str
    composite_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    volume_score: float = 0.0
    trend_score: float = 0.0
    win_rate: float = 0.0
    ml_confidence: float = 0.0
    grade: str = "C"

    @property
    def is_elite(self) -> bool:
        return self.grade in ("A+", "A")


class CoinRanker:
    def __init__(self, learner: Optional[SelfLearningEngine] = None) -> None:
        self._learner = learner
        self._btc_returns: Optional[pd.Series] = None

    def update_btc(self, btc_df: pd.DataFrame) -> None:
        self._btc_returns = btc_df["close"].pct_change().dropna()

    async def rank(
        self, klines: Dict[str, pd.DataFrame],
        ml_predictions: Optional[Dict[str, float]] = None,
    ) -> List[CoinRank]:
        loop = asyncio.get_running_loop()
        results: List[CoinRank] = []
        for symbol, df in klines.items():
            if len(df) < 50:
                continue
            try:
                rank = await loop.run_in_executor(
                    None, self._analyse, symbol, df, ml_predictions or {}
                )
                results.append(rank)
            except Exception as exc:
                logger.debug("CoinRanker error {}: {}", symbol, exc)
        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    def _analyse(self, symbol: str, df: pd.DataFrame, ml_predictions: dict) -> CoinRank:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Momentum score
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
        roc5 = close.pct_change(5).iloc[-1]
        roc20 = close.pct_change(20).iloc[-1]
        momentum_score = round(self._clamp(
            (self._norm(rsi, 30, 70) + self._clamp(roc5 * 10) + self._clamp(roc20 * 5)) / 3.0
        ), 4)

        # Volatility score
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close
        ).average_true_range().iloc[-1]
        atr_pct = atr / close.iloc[-1]
        vol_score = round(self._clamp(1 - abs(atr_pct * 100 - 3) / 5), 4)

        # Volume consistency score
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1
        vol_score_val = round(self._clamp(min(vol_ratio / 3.0, 1.0)), 4)

        # Trend alignment score
        ema9 = ta.trend.EMAIndicator(close=close, window=9).ema_indicator().iloc[-1]
        ema20v = ta.trend.EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]
        c = close.iloc[-1]
        trend_pts = sum([c > ema9, c > ema20v, c > ema50, ema9 > ema20v, ema20v > ema50])
        trend_score = round(trend_pts / 5.0, 4)

        ml_conf = ml_predictions.get(symbol, 0.5)
        win_rate = self._learner.get_coin_win_rate(symbol) if self._learner else 0.5

        # BTC correlation penalty
        corr_penalty = 0.0
        if self._btc_returns is not None and len(df) >= 50:
            coin_ret = close.pct_change().dropna().tail(50)
            btc_ret = self._btc_returns.tail(50)
            min_len = min(len(coin_ret), len(btc_ret))
            if min_len >= 20:
                a = pd.Series(coin_ret.values[-min_len:])
                b = pd.Series(btc_ret.values[-min_len:])
                # Guard: skip correlation when either series has no variance
                if a.std() > 1e-10 and b.std() > 1e-10:
                    corr = float(a.corr(b))
                    if not pd.isna(corr):
                        corr_penalty = max(0, corr - 0.5) * 0.2

        composite = (
            momentum_score * 0.25 + vol_score * 0.10 + vol_score_val * 0.15
            + trend_score * 0.25 + ml_conf * 0.15 + win_rate * 0.10
            - corr_penalty
        )
        composite = round(max(0.0, min(1.0, composite)), 4)

        if composite >= 0.80:
            grade = "A+"
        elif composite >= 0.65:
            grade = "A"
        elif composite >= 0.50:
            grade = "B"
        elif composite >= 0.35:
            grade = "C"
        else:
            grade = "D"

        return CoinRank(
            symbol=symbol, composite_score=composite,
            momentum_score=momentum_score, volatility_score=vol_score,
            volume_score=vol_score_val, trend_score=trend_score,
            win_rate=round(win_rate, 4), ml_confidence=round(ml_conf, 4),
            grade=grade,
        )

    @staticmethod
    def _norm(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.5
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))
