"""
data_engine/coin_database.py — Per-coin self-learning research database.

Tables:
  coin_stats          — tracks win rate, avg signal score, trade frequency per coin
  research_log        — logs every deep-research decision and its eventual outcome
  strategy_stats      — per-strategy win-rate and PnL tracking
  ml_training_samples — XGBoost feature vectors + labels (replaces training_samples.pkl)

The system populates these tables automatically as it scans and trades,
building a growing knowledge base used by ResearchEngine to bias future
decisions toward historically profitable setups.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text,
    select, update, func, delete,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────
class CoinStats(Base):
    """One row per symbol. Updated on every signal hit and every trade close."""
    __tablename__ = "coin_stats"

    symbol         = Column(String(20), primary_key=True)
    total_signals  = Column(Integer, default=0)     # how many times scanner flagged it
    times_researched = Column(Integer, default=0)   # how many deep-research runs
    times_executed = Column(Integer, default=0)     # actual trades opened
    wins           = Column(Integer, default=0)
    losses         = Column(Integer, default=0)
    total_pnl      = Column(Float, default=0.0)
    avg_score      = Column(Float, default=0.0)     # rolling average scanner score
    last_signal_ts = Column(DateTime, nullable=True)
    updated_at     = Column(DateTime, default=datetime.utcnow)


class ResearchLog(Base):
    """One row per research decision. outcome_pnl filled when trade closes."""
    __tablename__ = "research_log"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    symbol         = Column(String(20), nullable=False, index=True)
    ts             = Column(DateTime, default=datetime.utcnow)
    direction      = Column(String(5), default="LONG")
    initial_score  = Column(Integer, default=0)
    research_score = Column(Float, default=0.0)     # 0–10
    mtf_alignment  = Column(Float, default=0.0)     # 0–1
    passed         = Column(Boolean, default=False)
    executed       = Column(Boolean, default=False)
    trade_id       = Column(Integer, nullable=True)  # FK → trades.id
    outcome_pnl    = Column(Float, nullable=True)    # set when trade closes
    # Detailed analysis tags — used for pattern mining and bot research
    reasons_json       = Column(Text, nullable=True)        # JSON list: tags that passed
    failed_reasons_json = Column(Text, nullable=True)       # JSON list: tags that failed
    ai_analysis_json   = Column(Text, nullable=True)        # JSON: LLM response when available


class StrategyStats(Base):
    """One row per strategy. Win-rate, PnL, and recent-performance tracking."""
    __tablename__ = "strategy_stats"

    name             = Column(String(50), primary_key=True)
    total_trades     = Column(Integer, default=0)
    wins             = Column(Integer, default=0)
    losses           = Column(Integer, default=0)
    total_pnl        = Column(Float,   default=0.0)
    avg_pnl          = Column(Float,   default=0.0)   # rolling average PnL per trade
    recent_pnl_json  = Column(String(500), default="[]")  # JSON list of last 10 PnL values
    updated_at       = Column(DateTime, default=datetime.utcnow)


class MLTrainingSample(Base):
    """One row per trade signal that reached ML prediction stage.
    label=NULL means trade is still open (pending); 0=loss, 1=win after close.
    Replaces training_samples.pkl — data survives VPS resets and bot restarts.
    """
    __tablename__ = "ml_training_samples"

    id         = Column(Integer,  primary_key=True, autoincrement=True)
    trade_id   = Column(String(80), unique=True, nullable=False, index=True)
    symbol     = Column(String(20), nullable=False, index=True)
    features   = Column(Text,     nullable=False)   # JSON-encoded float32 list
    label      = Column(Integer,  nullable=True)    # None=pending, 0=loss, 1=win
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# CoinDatabase
# ─────────────────────────────────────────────────────────────────────────────
class CoinDatabase:
    def __init__(self) -> None:
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._sf = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def start(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("CoinDatabase started")

    # ── Signal hit ────────────────────────────────────────────────────────
    async def record_signal(self, symbol: str, score: int) -> None:
        """Called every scan cycle when a coin passes the score threshold."""
        async with self._sf() as session:
            result = await session.execute(
                select(CoinStats).where(CoinStats.symbol == symbol)
            )
            row = result.scalar_one_or_none()
            now = datetime.utcnow()
            if row:
                new_avg = (row.avg_score * row.total_signals + score) / (row.total_signals + 1)
                await session.execute(
                    update(CoinStats).where(CoinStats.symbol == symbol).values(
                        total_signals=row.total_signals + 1,
                        avg_score=round(new_avg, 2),
                        last_signal_ts=now,
                        updated_at=now,
                    )
                )
            else:
                session.add(CoinStats(
                    symbol=symbol, total_signals=1, avg_score=float(score),
                    last_signal_ts=now, updated_at=now,
                ))
            await session.commit()

    # ── Research decision ─────────────────────────────────────────────────
    async def record_research(
        self,
        symbol: str,
        direction: str,
        initial_score: int,
        research_score: float,
        mtf_alignment: float,
        passed: bool,
        reasons: Optional[List[str]] = None,
        failed_reasons: Optional[List[str]] = None,
        ai_analysis: Optional[dict] = None,
    ) -> int:
        """Log a research decision. Returns research_log row ID."""
        async with self._sf() as session:
            log = ResearchLog(
                symbol=symbol,
                direction=direction,
                initial_score=initial_score,
                research_score=research_score,
                mtf_alignment=mtf_alignment,
                passed=passed,
                reasons_json=json.dumps(reasons) if reasons else None,
                failed_reasons_json=json.dumps(failed_reasons) if failed_reasons else None,
                ai_analysis_json=json.dumps(ai_analysis) if ai_analysis else None,
            )
            session.add(log)
            await session.flush()   # populate log.id before commit
            log_id = log.id

            # Upsert coin_stats: create row if first time this symbol is researched
            res = await session.execute(select(CoinStats).where(CoinStats.symbol == symbol))
            row = res.scalar_one_or_none()
            now = datetime.utcnow()
            if row is None:
                session.add(CoinStats(
                    symbol=symbol, times_researched=1, updated_at=now,
                ))
            else:
                await session.execute(
                    update(CoinStats).where(CoinStats.symbol == symbol).values(
                        times_researched=CoinStats.times_researched + 1,
                        updated_at=now,
                    )
                )
            await session.commit()
        return log_id

    async def mark_research_executed(self, research_id: int, trade_id: int) -> None:
        """Called when a trade is actually opened after research passed."""
        async with self._sf() as session:
            res = await session.execute(
                select(ResearchLog).where(ResearchLog.id == research_id)
            )
            log = res.scalar_one_or_none()
            if not log:
                return
            await session.execute(
                update(ResearchLog).where(ResearchLog.id == research_id).values(
                    executed=True, trade_id=trade_id,
                )
            )
            await session.execute(
                update(CoinStats).where(CoinStats.symbol == log.symbol).values(
                    times_executed=CoinStats.times_executed + 1,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

    async def update_research_outcome(self, research_id: int, pnl: float) -> None:
        """Called when the trade connected to a research decision closes."""
        async with self._sf() as session:
            res = await session.execute(
                select(ResearchLog).where(ResearchLog.id == research_id)
            )
            log = res.scalar_one_or_none()
            if not log:
                return
            await session.execute(
                update(ResearchLog).where(ResearchLog.id == research_id).values(
                    outcome_pnl=pnl,
                )
            )
            await session.execute(
                update(CoinStats).where(CoinStats.symbol == log.symbol).values(
                    wins=CoinStats.wins + (1 if pnl >= 0.3 else 0),
                    losses=CoinStats.losses + (1 if pnl < 0.3 else 0),
                    total_pnl=CoinStats.total_pnl + pnl,
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()

    # ── Query helpers ─────────────────────────────────────────────────────
    async def get_coin_stats(self, symbol: str) -> Optional[dict]:
        async with self._sf() as session:
            result = await session.execute(
                select(CoinStats).where(CoinStats.symbol == symbol)
            )
            row = result.scalar_one_or_none()
        if not row:
            return None
        total = row.wins + row.losses
        return {
            "symbol":           row.symbol,
            "total_signals":    row.total_signals,
            "times_researched": row.times_researched,
            "times_executed":   row.times_executed,
            "wins":             row.wins,
            "losses":           row.losses,
            "win_rate":         round(row.wins / total * 100, 1) if total > 0 else 0.0,
            "total_pnl":        round(row.total_pnl, 2),
            "avg_score":        row.avg_score,
            "last_signal_ts":   row.last_signal_ts.isoformat() if row.last_signal_ts else None,
        }

    async def get_all_coin_stats(self, limit: int = 200) -> List[dict]:
        """All tracked coins with any signals, sorted by times_executed then total_signals."""
        async with self._sf() as session:
            result = await session.execute(
                select(CoinStats)
                .where(CoinStats.total_signals > 0)
                .order_by(
                    CoinStats.times_executed.desc(),
                    CoinStats.total_signals.desc(),
                )
                .limit(limit)
            )
            rows = result.scalars().all()
        out = []
        for r in rows:
            total = r.wins + r.losses
            out.append({
                "symbol":           r.symbol,
                "total_signals":    r.total_signals,
                "times_researched": r.times_researched,
                "times_executed":   r.times_executed,
                "wins":             r.wins,
                "losses":           r.losses,
                "win_rate":         round(r.wins / total * 100, 1) if total > 0 else None,
                "total_pnl":        round(r.total_pnl, 2),
                "avg_score":        r.avg_score,
            })
        return out

    async def get_top_coins(self, limit: int = 20) -> List[dict]:
        """Coins with >= 1 executed trade, sorted by win rate."""
        async with self._sf() as session:
            result = await session.execute(
                select(CoinStats)
                .where(CoinStats.times_executed >= 1)
                .order_by(
                    (CoinStats.wins / func.greatest(CoinStats.wins + CoinStats.losses, 1)).desc()
                )
                .limit(limit)
            )
            rows = result.scalars().all()
        out = []
        for r in rows:
            total = r.wins + r.losses
            out.append({
                "symbol":         r.symbol,
                "win_rate":       round(r.wins / total * 100, 1) if total > 0 else 0.0,
                "times_executed": r.times_executed,
                "total_pnl":      round(r.total_pnl, 2),
                "avg_score":      r.avg_score,
            })
        return out

    async def get_recent_research(self, limit: int = 50) -> List[dict]:
        """Last N research decisions for the dashboard."""
        async with self._sf() as session:
            result = await session.execute(
                select(ResearchLog)
                .order_by(ResearchLog.ts.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            {
                "id":               r.id,
                "symbol":           r.symbol,
                "ts":               r.ts.isoformat() if r.ts else None,
                "direction":        r.direction,
                "initial_score":    r.initial_score,
                "research_score":   round(r.research_score, 2),
                "mtf_alignment":    round(r.mtf_alignment, 2),
                "passed":           r.passed,
                "executed":         r.executed,
                "outcome_pnl":      r.outcome_pnl,
                "reasons":          json.loads(r.reasons_json) if r.reasons_json else [],
                "failed_reasons":   json.loads(r.failed_reasons_json) if r.failed_reasons_json else [],
                "ai_analysis":      json.loads(r.ai_analysis_json) if r.ai_analysis_json else None,
            }
            for r in rows
        ]

    # ── Strategy performance ───────────────────────────────────────────────
    async def record_strategy_outcome(self, name: str, pnl: float) -> None:
        """Called every time a trade managed by a named strategy closes."""
        async with self._sf() as session:
            res = await session.execute(
                select(StrategyStats).where(StrategyStats.name == name)
            )
            row = res.scalar_one_or_none()
            now = datetime.utcnow()
            if row:
                recent = json.loads(row.recent_pnl_json or "[]")
                recent.append(pnl)
                recent = recent[-10:]   # keep only last 10
                total  = row.total_trades + 1
                new_avg = round(((row.avg_pnl * row.total_trades) + pnl) / total, 4)
                await session.execute(
                    update(StrategyStats).where(StrategyStats.name == name).values(
                        total_trades    = total,
                        wins            = row.wins   + (1 if pnl >= 0.3 else 0),
                        losses          = row.losses + (1 if pnl < 0.3 else 0),
                        total_pnl       = round(row.total_pnl + pnl, 4),
                        avg_pnl         = new_avg,
                        recent_pnl_json = json.dumps(recent),
                        updated_at      = now,
                    )
                )
            else:
                session.add(StrategyStats(
                    name            = name,
                    total_trades    = 1,
                    wins            = 1 if pnl >= 0.3 else 0,
                    losses          = 1 if pnl < 0.3 else 0,
                    total_pnl       = round(pnl, 4),
                    avg_pnl         = round(pnl, 4),
                    recent_pnl_json = json.dumps([pnl]),
                    updated_at      = now,
                ))
            await session.commit()

    async def get_strategy_stats(self) -> List[dict]:
        """All strategy rows sorted by win-rate descending."""
        async with self._sf() as session:
            result = await session.execute(
                select(StrategyStats).order_by(StrategyStats.total_trades.desc())
            )
            rows = result.scalars().all()
        out = []
        for r in rows:
            total  = r.total_trades or 0
            wins   = r.wins or 0
            recent = json.loads(r.recent_pnl_json or "[]")
            out.append({
                "name":         r.name,
                "total_trades": total,
                "wins":         wins,
                "losses":       r.losses or 0,
                "win_rate":     round(wins / total * 100, 1) if total > 0 else 0.0,
                "total_pnl":    round(r.total_pnl or 0.0, 2),
                "avg_pnl":      round(r.avg_pnl or 0.0, 2),
                "recent_pnl":   recent,
                "updated_at":   r.updated_at.isoformat() if r.updated_at else None,
            })
        return out

    # ── ML training samples ────────────────────────────────────────────────
    async def save_ml_sample(self, trade_id: str, symbol: str, features: list) -> None:
        """Persist a pending feature vector when ML prediction fires.
        Call this as a background task from record_prediction().
        """
        async with self._sf() as session:
            # Upsert: ignore if trade_id already exists (idempotent)
            existing = await session.execute(
                select(MLTrainingSample).where(MLTrainingSample.trade_id == trade_id)
            )
            if existing.scalar_one_or_none() is None:
                session.add(MLTrainingSample(
                    trade_id=trade_id,
                    symbol=symbol,
                    features=json.dumps([float(v) for v in features]),
                ))
                await session.commit()

    async def label_ml_sample(self, trade_id: str, label: int) -> None:
        """Set win(1)/loss(0) label once trade closes.
        Call this as a background task from label_trade().
        """
        async with self._sf() as session:
            await session.execute(
                update(MLTrainingSample)
                .where(MLTrainingSample.trade_id == trade_id)
                .values(label=label)
            )
            await session.commit()

    async def load_ml_samples(self) -> Tuple[List[np.ndarray], List[int], Dict[str, dict]]:
        """Return (X_list, y_list, coin_stats) for ALL labeled samples.
        Used by SelfLearningEngine.initialize() to restore state on startup.
        """
        async with self._sf() as session:
            result = await session.execute(
                select(MLTrainingSample)
                .where(MLTrainingSample.label.isnot(None))
                .order_by(MLTrainingSample.id.asc())
            )
            rows = result.scalars().all()

        X, y, coin_stats = [], [], {}
        for r in rows:
            try:
                fvec = np.array(json.loads(r.features), dtype=np.float32)
                X.append(fvec)
                y.append(int(r.label))
                stat = coin_stats.setdefault(r.symbol, {"wins": 0, "losses": 0, "total": 0})
                stat["total"] += 1
                if r.label == 1:
                    stat["wins"] += 1
                else:
                    stat["losses"] += 1
            except Exception as exc:
                logger.debug("load_ml_samples: skip row {}, error: {}", r.id, exc)

        for stat in coin_stats.values():
            stat["win_rate"] = stat["wins"] / stat["total"] if stat["total"] > 0 else 0.5

        logger.info("load_ml_samples: {} labeled samples, {} coins", len(y), len(coin_stats))
        return X, y, coin_stats

    async def load_pending_ml_samples(self) -> Dict[str, dict]:
        """Return {trade_id: {symbol, features}} for unlabeled (open-trade) samples.
        Used on startup to restore _pending dict so in-flight trades can still be labeled.
        """
        async with self._sf() as session:
            result = await session.execute(
                select(MLTrainingSample).where(MLTrainingSample.label.is_(None))
            )
            rows = result.scalars().all()

        pending = {}
        for r in rows:
            try:
                fvec = np.array(json.loads(r.features), dtype=np.float32)
                pending[r.trade_id] = {
                    "symbol":   r.symbol,
                    "features": fvec,
                    "ts":       r.created_at.isoformat() if r.created_at else "",
                }
            except Exception as exc:
                logger.debug("load_pending_ml_samples: skip row {}: {}", r.id, exc)

        if pending:
            logger.info("load_pending_ml_samples: recovered {} pending predictions", len(pending))
        return pending

    async def cleanup_stale_ml_samples(self, hours: int = 24) -> int:
        """Delete pending (unlabeled) ML samples older than `hours`.
        Called on startup to clear zombie records from past scan cycles.
        Returns count of deleted rows.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        async with self._sf() as session:
            result = await session.execute(
                delete(MLTrainingSample)
                .where(MLTrainingSample.label.is_(None))
                .where(MLTrainingSample.created_at < cutoff)
            )
            await session.commit()
            deleted = result.rowcount
        if deleted:
            logger.info("ML cleanup: removed {} stale pending samples (>{} h old)", deleted, hours)
        return deleted

    async def load_labels_timeline(self, limit: int = 100) -> List[dict]:
        """Return last `limit` labeled trades as timeline dicts for rebuilding
        SelfLearningEngine._labels_timeline after a restart.

        Returns entries in chronological order (oldest first).
        NOTE: `pnl` is None here (not stored in ml_training_samples) — the
        dashboard renders it as '--', which is correct behaviour.
        """
        async with self._sf() as session:
            res = await session.execute(
                select(MLTrainingSample)
                .where(MLTrainingSample.label.isnot(None))
                .order_by(MLTrainingSample.created_at.desc())
                .limit(limit)
            )
            rows = res.scalars().all()
        return [
            {
                "ts":     r.created_at.isoformat() if r.created_at else None,
                "symbol": r.symbol,
                "label":  r.label,
                "pnl":    None,  # not stored at ML-sample level; shows '--' in dashboard
            }
            for r in reversed(rows)   # restore chronological order
        ]
