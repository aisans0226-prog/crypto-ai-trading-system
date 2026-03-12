"""
scanners/research_engine.py — Deep multi-timeframe research before trade execution.

For each watchlist candidate, fetches 15m / 1h / 4h klines and evaluates:
  1. EMA trend alignment (price > EMA20 > EMA50 vs full bull-stack EMA20>50>200)
  2. RSI zone — avoids entering overbought (>70) or oversold (<30) conditions
  3. Volume confirmation — last bar >= 130% of 20-bar rolling average
  4. Momentum — 3-candle consecutive in trade direction
  5. Historical win-rate bonus sourced from CoinDatabase

Scoring: each timeframe contributes a weighted score (15m×1, 1h×2, 4h×3).
Final score is normalised to 0–10. Passes when score >= config threshold
AND at least 2/3 timeframes agree with the trade direction.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from config import settings
from data_engine.coin_database import CoinDatabase
import ai_engine.llm_analyzer as llm_analyzer


@dataclass
class ResearchResult:
    symbol:         str
    passed:         bool
    confidence:     float           # 0.0 – 1.0
    score:          float           # normalised 0–10
    mtf_alignment:  float           # fraction of TFs agreeing (0–1)
    reasons:        List[str] = field(default_factory=list)
    failed_reasons: List[str] = field(default_factory=list)
    ai_analysis:    Optional[object] = None   # LLMAnalysis | None


class ResearchEngine:
    # Timeframes and their score weights (higher TF = more weight)
    _TIMEFRAMES = [("15m", 1.0), ("1h", 2.0), ("4h", 3.0)]
    _MAX_RAW = sum(w * 3.5 for _, w in _TIMEFRAMES)   # 3.5 pts max per TF

    def __init__(self, data_engine, coin_db: CoinDatabase) -> None:
        self._data = data_engine
        self._db   = coin_db

    async def research(
        self,
        symbol: str,
        direction: str,
        initial_score: int,
    ) -> ResearchResult:
        """
        Run full multi-timeframe deep research.
        Returns ResearchResult — caller checks .passed before executing trade.
        """
        raw_score = 0.0
        reasons: List[str] = []
        failed: List[str] = []
        tf_agrees: List[bool] = []

        for tf, weight in self._TIMEFRAMES:
            try:
                df = await self._data.get_klines_binance(symbol, tf, 200)
                tf_score, agrees, tf_ok, tf_bad = self._analyze_timeframe(df, direction, tf)
                raw_score += tf_score * weight
                tf_agrees.append(agrees)
                reasons.extend(tf_ok)
                failed.extend(tf_bad)
            except Exception as exc:
                logger.debug("Research {}/{} error: {}", symbol, tf, exc)
                tf_agrees.append(False)
                failed.append(f"{tf}:data_error")

        # Normalise to 0–10
        score = min(10.0, (raw_score / self._MAX_RAW) * 10.0)
        mtf_alignment = sum(tf_agrees) / len(tf_agrees) if tf_agrees else 0.0

        # Historical win-rate bonus/penalty (up to ±1 pt) when >=3 trades exist
        stats = await self._db.get_coin_stats(symbol)
        if stats and stats["times_executed"] >= 3:
            hist_wr = stats["win_rate"] / 100.0
            bonus = (hist_wr - 0.5) * 2.0          # −1 … +1
            score = max(0.0, min(10.0, score + bonus))
            tag = f"hist_wr_{stats['win_rate']:.0f}pct"
            (reasons if bonus >= 0 else failed).append(tag)

        passed = (
            score >= settings.effective_research_min_score
            and mtf_alignment >= settings.effective_research_min_mtf_alignment
        )

        # ── LLM analysis (only active when ai_analysis_enabled=True) ─────────
        ai_result = None
        if llm_analyzer.is_enabled():
            try:
                tf_tags = self._parse_tf_tags(reasons)
                ai_result = await llm_analyzer.LLMAnalyzer().analyze(
                    symbol=symbol,
                    direction=direction,
                    signal_score=initial_score,
                    research_score=score,
                    mtf_alignment=mtf_alignment,
                    tf_tags=tf_tags,
                )
                if ai_result.enabled:
                    # Apply score adjustment (clamped to 0–10)
                    score = max(0.0, min(10.0, score + ai_result.score_delta))
                    tag = f"ai:{ai_result.recommendation}(δ{ai_result.score_delta:+.1f})"
                    if ai_result.recommendation == "avoid":
                        # LLM strongly disagrees — override to failed
                        passed = False
                        failed.append(f"ai:avoid — {ai_result.reasoning[:100]}")
                    else:
                        reasons.append(tag)
                        # Re-evaluate pass with adjusted score
                        passed = (
                            score >= settings.effective_research_min_score
                            and mtf_alignment >= settings.effective_research_min_mtf_alignment
                        )
            except Exception as exc:
                logger.debug("LLM research integration error: {}", exc)

        return ResearchResult(
            symbol=symbol,
            passed=passed,
            confidence=round(score / 10.0, 3),
            score=round(score, 2),
            mtf_alignment=round(mtf_alignment, 2),
            reasons=reasons,
            failed_reasons=failed,
            ai_analysis=ai_result,
        )

    # ── Single-timeframe analysis ─────────────────────────────────────────
    @staticmethod
    def _analyze_timeframe(
        df: pd.DataFrame,
        direction: str,
        label: str,
    ) -> Tuple[float, bool, List[str], List[str]]:
        """
        Score one timeframe. Returns (score 0–3.5, agrees_with_direction, ok_tags, fail_tags).
        """
        score  = 0.0
        ok: List[str]   = []
        bad: List[str]  = []

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)
        last   = close.iloc[-1]

        # ── 1. EMA alignment (max 1.5) ─────────────────────────────────────
        ema20  = close.ewm(span=20).mean()
        ema50  = close.ewm(span=50).mean()
        ema200 = close.ewm(span=200).mean()

        if direction == "LONG":
            full_stack = (last > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1])
            partial    = (last > ema20.iloc[-1])
        else:
            full_stack = (last < ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1])
            partial    = (last < ema20.iloc[-1])

        if full_stack:
            score += 1.5
            ok.append(f"{label}:ema_stack_full")
        elif partial:
            score += 0.5
            ok.append(f"{label}:ema_partial")
        else:
            bad.append(f"{label}:ema_against_direction")

        # ── 2. RSI zone (max 1.0) ──────────────────────────────────────────
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi    = 100 - (100 / (1 + gain / (loss + 1e-9)))
        rsi_v  = float(rsi.iloc[-1])

        if direction == "LONG" and 40 <= rsi_v <= 65:
            score += 1.0
            ok.append(f"{label}:rsi_ideal_{rsi_v:.0f}")
        elif direction == "SHORT" and 35 <= rsi_v <= 60:
            score += 1.0
            ok.append(f"{label}:rsi_ideal_{rsi_v:.0f}")
        elif rsi_v > 72 or rsi_v < 28:
            bad.append(f"{label}:rsi_extreme_{rsi_v:.0f}")
        else:
            score += 0.3
            ok.append(f"{label}:rsi_ok_{rsi_v:.0f}")

        # ── 3. Volume confirmation (max 0.5) ───────────────────────────────
        vol_avg  = volume.rolling(20).mean().iloc[-1]
        vol_last = volume.iloc[-1]
        if vol_avg > 0 and vol_last >= vol_avg * 1.3:
            score += 0.5
            ok.append(f"{label}:vol_{vol_last/vol_avg:.1f}x")
        else:
            bad.append(f"{label}:vol_weak")

        # ── 4. Momentum — 3 consecutive candles in direction (max 0.5) ────
        recent = close.iloc[-4:].values
        if direction == "LONG" and all(recent[i] < recent[i+1] for i in range(3)):
            score += 0.5
            ok.append(f"{label}:momentum_bullish")
        elif direction == "SHORT" and all(recent[i] > recent[i+1] for i in range(3)):
            score += 0.5
            ok.append(f"{label}:momentum_bearish")

        # Agreement = EMA at least partial + RSI not extreme
        agrees = partial and not (rsi_v > 72 or rsi_v < 28)

        return score, agrees, ok, bad

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_tf_tags(reasons: List[str]) -> Dict[str, List[str]]:
        """Extract per-timeframe tag lists from the flat reasons list."""
        tags: Dict[str, List[str]] = {"15m": [], "1h": [], "4h": []}
        for r in reasons:
            for tf in tags:
                if r.startswith(tf + ":"):
                    tags[tf].append(r[len(tf) + 1:])
                    break
        return tags
