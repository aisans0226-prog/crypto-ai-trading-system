"""
data_engine/coin_database.py — Per-coin self-learning research database.

Two tables:
  coin_stats     — tracks win rate, avg signal score, trade frequency per coin
  research_log   — logs every deep-research decision and its eventual outcome

The system populates these tables automatically as it scans and trades,
building a growing knowledge base used by ResearchEngine to bias future
decisions toward historically profitable setups.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    select, update, func,
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
                    wins=CoinStats.wins + (1 if pnl > 0 else 0),
                    losses=CoinStats.losses + (1 if pnl <= 0 else 0),
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

    async def get_top_coins(self, limit: int = 20) -> List[dict]:
        """Coins with >= 2 executed trades, sorted by win rate."""
        async with self._sf() as session:
            result = await session.execute(
                select(CoinStats)
                .where(CoinStats.times_executed >= 2)
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
                "id":             r.id,
                "symbol":         r.symbol,
                "ts":             r.ts.isoformat() if r.ts else None,
                "direction":      r.direction,
                "initial_score":  r.initial_score,
                "research_score": round(r.research_score, 2),
                "mtf_alignment":  round(r.mtf_alignment, 2),
                "passed":         r.passed,
                "executed":       r.executed,
                "outcome_pnl":    r.outcome_pnl,
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
                        wins            = row.wins   + (1 if pnl > 0 else 0),
                        losses          = row.losses + (1 if pnl <= 0 else 0),
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
                    wins            = 1 if pnl > 0 else 0,
                    losses          = 1 if pnl <= 0 else 0,
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
