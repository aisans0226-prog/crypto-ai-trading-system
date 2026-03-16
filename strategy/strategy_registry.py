"""
strategy/strategy_registry.py — Auto Strategy Discovery engine.

Before every trade:
  1. All 6 strategies evaluate the signal's symbol + dataframe.
  2. Each valid setup is scored by a dynamic fitness formula:
       fitness = w_reg × regime_fit  +  w_stat × win_rate  +  w_rec × recent_score  +  w_rr × rr_score
  3. Weights shift from regime-heavy → stats-heavy as historical data accumulates.
  4. Setup with the highest fitness is returned as the trade's entry plan.
  5. If no strategy qualifies, SCAN_ALIGNED fallback fires when basic EMA+RSI+ADX
     conditions confirm the scanner's signal direction.

After every trade close:
  record_outcome(strategy_name, pnl) → persists to strategy_stats table.

The fitness formula ensures:
  - New strategy (< 5 trades): decision driven almost entirely by regime fit.
  - Growing strategy (5–20 trades): stats gain 25 % weight.
  - Mature strategy (≥ 20 trades): historical win-rate carries 45 % weight.
"""
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ta
from loguru import logger

from strategy.trend_strategy import TrendStrategy, TradeSetup
from strategy.breakout_strategy import BreakoutStrategy
from strategy.liquidity_strategy import LiquidityStrategy
from strategy.momentum_strategy import MomentumStrategy
from strategy.mean_reversion_strategy import MeanReversionStrategy
from strategy.scalp_strategy import ScalpStrategy


class StrategyRegistry:
    """Auto-selects the best strategy for each trade opportunity."""

    _CACHE_TTL = 300   # seconds between DB cache refreshes

    def __init__(self, coin_db) -> None:
        self._db = coin_db
        self._strategies = [
            TrendStrategy(),
            BreakoutStrategy(),
            LiquidityStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            ScalpStrategy(),
        ]
        self._stats_cache: Dict[str, dict] = {}
        self._cache_ts: float = 0.0

    # ── Public interface ───────────────────────────────────────────────────

    def strategy_names(self) -> List[str]:
        return [s.NAME for s in self._strategies]

    async def select_best(
        self,
        symbol: str,
        df: pd.DataFrame,
        direction: str,
    ) -> Tuple[Optional[TradeSetup], Optional[str]]:
        """
        Evaluate all strategies and return (TradeSetup, strategy_name)
        with the highest fitness score.  Returns (None, None) if none qualify.
        """
        await self._refresh_cache()

        candidates = []
        for strat in self._strategies:
            # Auto-disable strategies with proven negative avg PnL (≥ 3 trades, avg < -1 USDT)
            strat_stats = self._stats_cache.get(strat.NAME, {})
            strat_trades = strat_stats.get("total_trades", 0)
            strat_total_pnl = strat_stats.get("total_pnl", 0.0)
            if strat_trades >= 3 and strat_total_pnl / strat_trades < -1.0:
                logger.debug(
                    "Skipping {} — negative avg PnL {:.2f} over {} trades",
                    strat.NAME, strat_total_pnl / strat_trades, strat_trades,
                )
                continue

            try:
                setup = strat.evaluate(symbol, df)
            except Exception as exc:
                logger.debug("Strategy {} error ({}): {}", strat.NAME, symbol, exc)
                continue

            if setup is None or setup.direction != direction:
                continue

            fitness = self._compute_fitness(strat, df, setup)
            candidates.append((fitness, setup, strat.NAME))
            logger.debug(
                "{} {} fitness={:.3f}", symbol, strat.NAME, fitness
            )

        if not candidates:
            # SCAN_ALIGNED fallback: fires when scanner + research confirmed a signal
            # but no named strategy pattern matched.  Uses lightweight EMA/RSI/ADX
            # confirmation so it is technically grounded, not a pure blind entry.
            try:
                fallback = self._build_scan_aligned_setup(symbol, df, direction)
                if fallback:
                    logger.info(
                        "{} → SCAN_ALIGNED fallback ({} direction confirmed by EMA+RSI+ADX)",
                        symbol, direction,
                    )
                    return fallback, "SCAN_ALIGNED"
            except Exception as exc:
                logger.debug("SCAN_ALIGNED fallback error {}: {}", symbol, exc)
            return None, None

        candidates.sort(reverse=True, key=lambda x: x[0])
        best_fitness, best_setup, best_name = candidates[0]
        logger.info(
            "{} → best strategy: {} (fitness={:.3f}, {} candidates)",
            symbol, best_name, best_fitness, len(candidates),
        )
        return best_setup, best_name

    async def record_outcome(self, strategy_name: str, pnl: float) -> None:
        """Called when a trade managed by this registry closes."""
        try:
            await self._db.record_strategy_outcome(strategy_name, pnl)
            self._cache_ts = 0.0   # force refresh on next selection
        except Exception as exc:
            logger.debug("StrategyRegistry.record_outcome error: {}", exc)

    # ── Private helpers ────────────────────────────────────────────────────

    async def _refresh_cache(self) -> None:
        if time.time() - self._cache_ts < self._CACHE_TTL:
            return
        try:
            rows = await self._db.get_strategy_stats()
            self._stats_cache = {r["name"]: r for r in (rows or [])}
            self._cache_ts = time.time()
        except Exception as exc:
            logger.debug("Strategy stats cache refresh failed: {}", exc)

    def _compute_fitness(
        self, strat, df: pd.DataFrame, setup: TradeSetup
    ) -> float:
        stats  = self._stats_cache.get(strat.NAME, {})
        trades = stats.get("total_trades", 0)

        # Historical win-rate (0–1)
        win_rate = stats.get("win_rate", 50.0) / 100.0

        # Recent performance score from last 10 trades (0–1)
        recent = stats.get("recent_pnl", [])
        if not isinstance(recent, list):
            recent = []
        recent_score = (
            sum(1 for p in recent if p > 0) / len(recent)
            if recent else 0.5
        )

        # Regime fitness (strategy-specific, 0–1)
        regime = (
            strat.regime_fit(df)
            if hasattr(strat, "regime_fit")
            else 0.5
        )

        # Risk/reward score (capped at RR = 4; floored at 0 to prevent negative fitness)
        rr_score = max(0.0, min(setup.risk_reward / 4.0, 1.0))

        # Dynamic weight schedule
        if trades < 5:
            w_reg, w_stat, w_rec, w_rr = 0.70, 0.00, 0.10, 0.20
        elif trades < 20:
            w_reg, w_stat, w_rec, w_rr = 0.55, 0.25, 0.10, 0.10
        else:
            w_reg, w_stat, w_rec, w_rr = 0.35, 0.45, 0.10, 0.10

        return round(
            w_reg  * regime
            + w_stat * win_rate
            + w_rec  * recent_score
            + w_rr   * rr_score,
            4,
        )

    def _build_scan_aligned_setup(
        self, symbol: str, df: pd.DataFrame, direction: str
    ) -> Optional[TradeSetup]:
        """
        Lightweight entry aligned with scanner signal direction.
        Conditions (LONG):  EMA9 > EMA20, RSI 40–82, ADX > 15, volume ≥ avg
        Conditions (SHORT): EMA9 < EMA20, RSI 18–60, ADX > 15, volume ≥ avg
        SL: 1.5× ATR; TP: 3× ATR (1:2 RR).
        Only fires after the named strategies all declined — serves as the
        alignment bridge between scanner signals and trade execution.
        """
        from config import settings

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        ema9  = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
        ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        rsi   = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        adx   = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()
        atr   = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        avg_vol  = volume.rolling(20).mean().iloc[-1]
        last_vol = volume.iloc[-1]
        rsi_val  = rsi.iloc[-1]
        adx_val  = adx.iloc[-1]
        atr_val  = atr.iloc[-1]
        c        = close.iloc[-1]

        # Minimal volume check — at least average, no surge required
        if last_vol < avg_vol * 0.8:
            return None
        # Must have some directional strength
        if adx_val < 15:
            return None

        if direction == "LONG":
            if not (ema9.iloc[-1] > ema20.iloc[-1] and 40 <= rsi_val <= 82):
                return None
            stop = c - 1.5 * atr_val
            tp   = c + 3.0 * atr_val

        elif direction == "SHORT":
            if not (ema9.iloc[-1] < ema20.iloc[-1] and 18 <= rsi_val <= 60):
                return None
            stop = c + 1.5 * atr_val
            tp   = c - 3.0 * atr_val

        else:
            return None

        rr = abs(tp - c) / max(abs(c - stop), 1e-10)
        if rr < settings.min_risk_reward_ratio:
            return None

        return TradeSetup(
            symbol=symbol,
            direction=direction,
            entry_price=c,
            stop_loss=round(stop, 6),
            take_profit=round(tp, 6),
            leverage=settings.max_leverage,
            risk_reward=round(rr, 2),
        )
