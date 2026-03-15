"""
scanners/research_engine.py — Deep multi-timeframe research before trade execution.

For each watchlist candidate, fetches 15m / 1h / 4h klines IN PARALLEL and evaluates:
  1. EMA trend alignment (price > EMA20 > EMA50 vs full bull-stack EMA20>50>200)
  2. RSI zone — ideal zone + oversold/overbought recovery bonus
  3. Volume confirmation — last bar >= 130% of 20-bar rolling average
  4. Momentum — 3-candle consecutive in trade direction
  5. MACD histogram direction confirmation
  6. Bollinger Band position — avoid extreme over-extension
  7. Taker buy/sell volume delta — confirms institutional flow direction
  8. Historical win-rate bonus from CoinDatabase (global, not per-TF)
  9. Funding rate gate — high positive funding penalises LONG setups

Scoring: each timeframe contributes a weighted score (15m×1, 1h×2, 4h×3).
Final score is normalised to 0–10. Passes when score >= config threshold
AND at least 2/3 timeframes agree with the trade direction.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import ta
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
    # Max per-TF: EMA(1.5) + RSI(1.0) + Vol(0.5) + Mom(0.5) + MACD(0.5) + BB(0.5) + Delta(0.5) = 5.0
    _MAX_RAW = sum(w * 5.0 for _, w in _TIMEFRAMES)   # 30.0

    def __init__(self, data_engine, coin_db: CoinDatabase) -> None:
        self._data = data_engine
        self._db   = coin_db
        self._funding_cache: Dict[str, float] = {}   # updated each scan cycle

    def update_funding_cache(self, funding_map: Dict[str, float]) -> None:
        """Pre-populate funding rates fetched in bulk during market scan."""
        self._funding_cache = funding_map

    async def research(
        self,
        symbol: str,
        direction: str,
        initial_score: int,
        context: dict = None,
    ) -> ResearchResult:
        """
        Run full multi-timeframe deep research.
        Timeframe fetches execute in parallel for maximum speed.
        Returns ResearchResult — caller checks .passed before executing trade.

        context: optional dict with pre-fetched external data:
            fear_greed_value  — int 0-100 (Crypto Fear & Greed Index)
            oi_delta          — float % change in open interest since last cycle
            liq_distance_pct  — float % distance from nearest liquidation cluster
        """
        context = context or {}
        raw_score = 0.0
        reasons: List[str] = []
        failed: List[str] = []
        tf_agree_map: Dict[str, bool] = {}  # keyed by TF label for weighted alignment

        # ── Fetch all timeframes IN PARALLEL (3x faster than sequential) ──
        tf_list = [tf for tf, _ in self._TIMEFRAMES]
        tf_weights = {tf: w for tf, w in self._TIMEFRAMES}

        # Also fetch BTC 4h for trend filter (skip if symbol IS BTC)
        fetch_btc = symbol != "BTCUSDT"
        fetch_tasks = [self._data.get_klines_binance(symbol, tf, 200) for tf in tf_list]
        if fetch_btc:
            fetch_tasks.append(self._data.get_klines_binance("BTCUSDT", "4h", 60))

        all_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        df_results = all_results[:len(tf_list)]
        btc_df_4h = all_results[len(tf_list)] if fetch_btc else None

        for tf, df_or_exc in zip(tf_list, df_results):
            weight = tf_weights[tf]
            if isinstance(df_or_exc, Exception):
                logger.debug("Research {}/{} error: {}", symbol, tf, df_or_exc)
                tf_agree_map[tf] = False
                failed.append(f"{tf}:data_error")
                continue
            try:
                tf_score, agrees, tf_ok, tf_bad = self._analyze_timeframe(
                    df_or_exc, direction, tf
                )
                raw_score += tf_score * weight
                tf_agree_map[tf] = agrees
                reasons.extend(tf_ok)
                failed.extend(tf_bad)
            except Exception as exc:
                logger.debug("Research {}/{} analysis error: {}", symbol, tf, exc)
                tf_agree_map[tf] = False
                failed.append(f"{tf}:analysis_error")

        # Normalise to 0–10
        score = min(10.0, (raw_score / self._MAX_RAW) * 10.0)

        # ── Weighted MTF alignment (Fix 1) ───────────────────────────────────
        # 4h carries 3× the weight of 15m, matching the per-TF score weights.
        # If 4h disagrees the alignment drops to (1+2)/6=0.50 — well below the
        # 0.67 threshold.  If only 15m disagrees: (2+3)/6=0.83 — passes correctly.
        _TF_W = {"15m": 1, "1h": 2, "4h": 3}
        _w_agree = sum(_TF_W.get(tf, 1) for tf, a in tf_agree_map.items() if a)
        _w_total = sum(_TF_W.get(tf, 1) for tf in tf_agree_map) or 1
        mtf_alignment = round(_w_agree / _w_total, 2)

        # ── Funding rate gate (global, not per-TF) ────────────────────────
        funding = self._funding_cache.get(symbol)
        if funding is not None:
            if direction == "LONG" and funding > 0.005:
                # Very high positive funding = overheated LONG, likely to get squeezed
                penalty = min(1.5, (funding - 0.005) * 300)
                score = max(0.0, score - penalty)
                failed.append(f"high_funding_{funding:.4f}")
            elif direction == "SHORT" and funding < -0.005:
                # Very negative funding benefits SHORT squeeze risk too
                penalty = min(1.0, abs(funding + 0.005) * 200)
                score = max(0.0, score - penalty)
                failed.append(f"very_neg_funding_{funding:.4f}")

        # ── OI trend confirmation (global bonus) ─────────────────────────────
        # Rising OI on a LONG entry = real buying pressure (not just short covering).
        # Falling OI on a SHORT entry = real selling pressure.
        oi_delta = context.get("oi_delta")
        if oi_delta is not None:
            if direction == "LONG" and oi_delta > 0.3:
                score = min(10.0, score + 0.3)
                reasons.append(f"oi_rising_{oi_delta:.2f}pct")
                logger.debug("OI rising {:.2f}% → +0.3 research bonus", oi_delta)
            elif direction == "SHORT" and oi_delta < -0.3:
                score = min(10.0, score + 0.3)
                reasons.append(f"oi_falling_{oi_delta:.2f}pct")
                logger.debug("OI falling {:.2f}% → +0.3 research bonus", oi_delta)

        # ── Fear & Greed macro filter (if enabled and data available) ─────────
        if getattr(settings, "fear_greed_enabled", False):
            try:
                fg_value = context.get("fear_greed_value")
                if fg_value is not None:
                    if direction == "LONG" and fg_value <= 20:
                        # Extreme Fear: market in panic, dangerous to enter long
                        score = max(0.0, score - 0.5)
                        failed.append(f"fear_greed_{fg_value}_extreme_fear")
                        logger.debug("Fear&Greed={} (Extreme Fear) → -0.5 LONG penalty", fg_value)
                    elif direction == "LONG" and fg_value >= 85:
                        # Extreme Greed: market overheated, likely pullback soon
                        score = max(0.0, score - 0.3)
                        failed.append(f"fear_greed_{fg_value}_extreme_greed")
                        logger.debug("Fear&Greed={} (Extreme Greed) → -0.3 LONG penalty", fg_value)
                    elif direction == "SHORT" and fg_value >= 80:
                        # Greed extreme: good macro tailwind for a short
                        score = min(10.0, score + 0.3)
                        reasons.append(f"fear_greed_{fg_value}_greed_short")
                        logger.debug("Fear&Greed={} (Extreme Greed) → +0.3 SHORT bonus", fg_value)
            except Exception as _fg_err:
                logger.debug("Fear & Greed filter skipped: {}", _fg_err)

        # ── Liquidation cluster proximity bonus ───────────────────────────────
        # A liquidation cluster 0.2–1.5% away adds forced buying/selling pressure
        # that amplifies the move; the closer, the bigger the bonus (max +0.5).
        if getattr(settings, "liquidation_data_enabled", False):
            try:
                liq_distance = context.get("liq_distance_pct")
                if liq_distance is not None and 0.2 <= liq_distance <= 1.5:
                    # Linear bonus: 0.5 at 0.2% distance, 0.0 at 1.5% distance
                    bonus = round((1.5 - liq_distance) / 1.5 * 0.5, 2)
                    score = min(10.0, score + bonus)
                    reasons.append(f"liq_cluster_{liq_distance:.2f}pct_away")
                    logger.debug("Liquidation cluster at {:.2f}% → +{} bonus", liq_distance, bonus)
            except Exception as _liq_err:
                logger.debug("Liquidation bonus skipped: {}", _liq_err)

        # ── Historical win-rate bonus/penalty (±1 pt) when ≥3 trades ─────
        stats = await self._db.get_coin_stats(symbol)
        if stats and stats["times_executed"] >= 3:
            hist_wr = stats["win_rate"] / 100.0
            bonus = (hist_wr - 0.5) * 2.0          # −1 … +1
            score = max(0.0, min(10.0, score + bonus))
            tag = f"hist_wr_{stats['win_rate']:.0f}pct"
            (reasons if bonus >= 0 else failed).append(tag)

        # ── BTC 4h trend filter — block LONG in bear market (LONG WR 38% vs SHORT 51%) ──
        if direction == "LONG" and btc_df_4h is not None and not isinstance(btc_df_4h, Exception):
            try:
                btc_close = btc_df_4h["close"].astype(float)
                btc_ema20 = float(btc_close.ewm(span=20).mean().iloc[-1])
                btc_ema50 = float(btc_close.ewm(span=50).mean().iloc[-1])
                if btc_ema20 < btc_ema50:
                    score = max(0.0, score - 2.5)
                    failed.append(f"btc_4h_bearish_ema20={btc_ema20:.0f}<ema50={btc_ema50:.0f}")
                    logger.debug(
                        "{} LONG -2.5 penalty: BTC 4h bearish (EMA20={:.0f} < EMA50={:.0f})",
                        symbol, btc_ema20, btc_ema50,
                    )
            except Exception as _btc_err:
                logger.debug("BTC trend filter error: {}", _btc_err)

        # ── Score 9+ LONG exhaustion gate — high score = pump already done (WR 8.3%) ──
        if initial_score >= 9 and direction == "LONG":
            try:
                df_15m = df_results[0]  # 15m is index 0 in _TIMEFRAMES
                if not isinstance(df_15m, Exception) and df_15m is not None:
                    rsi_15m = float(
                        ta.momentum.RSIIndicator(
                            close=df_15m["close"].astype(float), window=14
                        ).rsi().iloc[-1]
                    )
                    if rsi_15m > 60:
                        score = max(0.0, score - 2.5)
                        failed.append(f"exhaustion_long_score{initial_score}_rsi15m={rsi_15m:.0f}")
                        logger.debug(
                            "{} score={} LONG RSI15m={:.0f}>60 exhaustion penalty -2.5",
                            symbol, initial_score, rsi_15m,
                        )
            except Exception as _rsi_err:
                logger.debug("Score9+ LONG RSI gate error: {}", _rsi_err)

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
                    score = max(0.0, min(10.0, score + ai_result.score_delta))
                    tag = f"ai:{ai_result.recommendation}(δ{ai_result.score_delta:+.1f})"
                    if ai_result.recommendation == "avoid":
                        passed = False
                        failed.append(f"ai:avoid — {ai_result.reasoning[:100]}")
                    else:
                        reasons.append(tag)
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
        Score one timeframe. Returns (score 0–5.0, agrees_with_direction, ok_tags, fail_tags).
        Max per TF: EMA(1.5)+RSI(1.0)+Vol(0.5)+Mom(0.5)+MACD(0.5)+BB(0.5)+Delta(0.5) = 5.0
        """
        score  = 0.0
        ok: List[str]   = []
        bad: List[str]  = []

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
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

        # ── 2. RSI zone — nuanced scoring (max 1.0) ────────────────────────
        # Use Wilder's EMA (via ta library) to match pump_detector.py and avoid
        # systematic 5-10pt discrepancy from the simpler SMA formula.
        rsi_v  = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1])

        if direction == "LONG":
            if 40 <= rsi_v <= 65:
                score += 1.0                              # ideal momentum zone
                ok.append(f"{label}:rsi_ideal_{rsi_v:.0f}")
            elif 28 <= rsi_v < 40:
                score += 0.7                              # oversold recovery — valid entry
                ok.append(f"{label}:rsi_recovery_{rsi_v:.0f}")
            elif rsi_v > 78:
                bad.append(f"{label}:rsi_overbought_{rsi_v:.0f}")
            else:
                score += 0.3
                ok.append(f"{label}:rsi_ok_{rsi_v:.0f}")
        else:
            if 35 <= rsi_v <= 60:
                score += 1.0                              # ideal short zone
                ok.append(f"{label}:rsi_ideal_{rsi_v:.0f}")
            elif 60 < rsi_v <= 72:
                score += 0.7                              # overbought reversal — valid short
                ok.append(f"{label}:rsi_overbought_short_{rsi_v:.0f}")
            elif rsi_v < 22:
                bad.append(f"{label}:rsi_oversold_{rsi_v:.0f}")
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

        # ── 5. MACD histogram direction (max 0.5) ─────────────────────────
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        hist = (macd_line - signal_line).iloc[-1]
        if (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
            score += 0.5
            ok.append(f"{label}:macd_aligned")
        elif abs(hist) < 1e-8:
            pass  # flat histogram — neutral
        else:
            bad.append(f"{label}:macd_against")

        # ── 6. Bollinger Band position — avoid over-extension (max 0.5) ───
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_range = (bb_upper - bb_lower).iloc[-1]
        if bb_range > 0:
            bb_pct = (last - bb_lower.iloc[-1]) / bb_range   # 0=lower band, 1=upper band
            if direction == "LONG" and 0.25 <= bb_pct <= 0.80:
                score += 0.5
                ok.append(f"{label}:bb_position_ok")
            elif direction == "SHORT" and 0.20 <= bb_pct <= 0.75:
                score += 0.5
                ok.append(f"{label}:bb_position_ok")
            elif (direction == "LONG" and bb_pct > 0.95) or \
                 (direction == "SHORT" and bb_pct < 0.05):
                bad.append(f"{label}:bb_overextended")

        # ── 7. Taker buy/sell delta (max 0.5) ─────────────────────────────
        # Binance provides taker_buy_base; Bybit may provide buy_ratio directly.
        # If neither column exists, award a neutral 0.25 (half credit) so Bybit
        # symbols are not systematically under-scored vs Binance symbols.
        if "taker_buy_base" in df.columns:
            taker_buy  = df["taker_buy_base"].astype(float).iloc[-3:].sum()
            total_vol  = volume.iloc[-3:].sum()
            if total_vol > 0:
                buy_ratio = taker_buy / total_vol
                if direction == "LONG" and buy_ratio >= 0.55:
                    score += 0.5
                    ok.append(f"{label}:taker_buy_{buy_ratio:.2f}")
                elif direction == "SHORT" and buy_ratio <= 0.45:
                    score += 0.5
                    ok.append(f"{label}:taker_sell_{1-buy_ratio:.2f}")
        elif "buy_ratio" in df.columns:
            # Bybit implementations may pre-compute this column
            buy_ratio = float(df["buy_ratio"].iloc[-1])
            if direction == "LONG" and buy_ratio >= 0.55:
                score += 0.5
                ok.append(f"{label}:taker_buy_{buy_ratio:.2f}")
            elif direction == "SHORT" and buy_ratio <= 0.45:
                score += 0.5
                ok.append(f"{label}:taker_sell_{1-buy_ratio:.2f}")
        else:
            # No taker data available — award neutral half-credit, not 0
            score += 0.25
            ok.append(f"{label}:taker_neutral")

        # Agreement = EMA at least partial + RSI not extreme
        agrees = partial and not (
            (direction == "LONG" and rsi_v > 78) or
            (direction == "SHORT" and rsi_v < 22)
        )

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
