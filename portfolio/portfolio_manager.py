"""
portfolio/portfolio_manager.py — Track balance, open positions, PnL, and risk exposure.

Stores data to PostgreSQL via SQLAlchemy async.
Also caches real-time position data in Redis.
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, text
import aioredis

from config import settings

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────
class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(5), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    pnl_usdt = Column(Float, default=0.0)
    funding_fee_usdt = Column(Float, default=0.0)   # estimated funding fees paid during hold
    status = Column(String(10), default="open")   # open | closed | cancelled
    exchange = Column(String(10), default="binance")
    order_id = Column(String(50), nullable=True)
    signal_score = Column(Integer, default=0)
    strategy_name = Column(String(50), nullable=True)       # which strategy triggered
    signal_confidence = Column(Float, nullable=True)        # ML ensemble probability (0–1)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)


class PerformanceSnapshot(Base):
    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(DateTime, default=datetime.utcnow)
    balance = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Manager
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioManager:
    def __init__(self) -> None:
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._redis: Optional[aioredis.Redis] = None
        self._balance = settings.account_balance_usdt
        self._manual_balance_override: bool = False   # True = ignore live exchange updates
        self._open_positions: Dict[str, dict] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────
    async def start(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Idempotent migration: add funding_fee_usdt if it doesn't exist yet
            await conn.execute(text(
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS funding_fee_usdt FLOAT DEFAULT 0.0"
            ))
        _redis_tmp = None
        try:
            _redis_tmp = await aioredis.from_url(settings.redis_url)
            await _redis_tmp.ping()
            self._redis = _redis_tmp
            logger.info("PortfolioManager started (Redis connected)")
            # Restore manual balance override if it was set before restart
            manual = await self._redis.get("portfolio:manual_balance")
            if manual:
                self._manual_balance_override = True
                # Use portfolio:balance (includes accumulated PnL), not the original set amount
                current = await self._redis.get("portfolio:balance")
                self._balance = float(current) if current else float(manual)
                logger.info("Restored manual balance override: ${:.2f}", self._balance)
            else:
                # Non-manual mode: restore last known balance from Redis so closed-trade
                # PnL accumulates correctly across bot restarts (not reset to config default).
                current = await self._redis.get("portfolio:balance")
                if current and float(current) > 0:
                    self._balance = float(current)
                    logger.info("Restored portfolio balance from Redis: ${:.2f}", self._balance)
        except Exception as exc:
            logger.warning("Redis unavailable ({}), running in-memory mode", exc)
            # Close the partially-created connection so no sockets are leaked
            if _redis_tmp is not None:
                try:
                    await _redis_tmp.aclose()
                except Exception:
                    pass
            self._redis = None
        # Re-sync in-memory + Redis from DB — prevents stale state across restarts
        await self._reload_open_positions()

    async def _reload_open_positions(self) -> None:
        """Load open trades from DB into memory + Redis on startup (DB is source of truth)."""
        async with self._session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TradeRecord).where(TradeRecord.status == "open")
            )
            trades = result.scalars().all()

        self._open_positions = {
            t.symbol: {
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "quantity": t.quantity,
                "leverage": t.leverage,
                "position_size_usdt": round(t.quantity * t.entry_price, 2),
                "exchange": t.exchange,
                "order_id": t.order_id or "",
                "signal_score": t.signal_score,
                # Persist open timestamp so _position_meta can be restored after restart
                "opened_at": t.opened_at.timestamp() if t.opened_at else None,
            }
            for t in trades
        }
        if self._redis:
            await self._redis.set("portfolio:positions", json.dumps(self._open_positions))
        logger.info("Reloaded {} open positions from DB on startup", len(self._open_positions))

    async def stop(self) -> None:
        if self._redis:
            await self._redis.close()

    # ── Balance management ────────────────────────────────────────────────
    async def get_balance(self) -> float:
        if self._redis:
            val = await self._redis.get("portfolio:balance")
            if val:
                return float(val)
        return self._balance

    async def update_balance(self, balance: float) -> None:
        if self._manual_balance_override:
            return  # Never overwrite user-set balance with live exchange data
        self._balance = balance
        if self._redis:
            await self._redis.set("portfolio:balance", balance)

    async def set_manual_balance(self, amount: float) -> None:
        """Set a manual starting balance. PnL from every closed trade compounds on top."""
        self._balance = amount
        self._manual_balance_override = True
        if self._redis:
            await self._redis.set("portfolio:balance", amount)
            await self._redis.set("portfolio:manual_balance", amount)
        logger.info("Manual balance set: ${:.2f}", amount)

    async def clear_manual_balance(self) -> None:
        """Remove manual override — balance reverts to live exchange data."""
        self._manual_balance_override = False
        if self._redis:
            await self._redis.delete("portfolio:manual_balance")
        logger.info("Manual balance override cleared")

    def is_manual_balance_active(self) -> bool:
        return self._manual_balance_override

    # ── Open positions ────────────────────────────────────────────────────
    async def get_open_positions(self) -> Dict[str, dict]:
        if self._redis:
            raw = await self._redis.get("portfolio:positions")
            if raw:
                return json.loads(raw)
        return self._open_positions

    async def open_position(self, trade_data: dict) -> int:
        async with self._session_factory() as session:
            record = TradeRecord(**{
                k: v for k, v in trade_data.items()
                if k in TradeRecord.__table__.columns.keys() and k != "id"
            })
            session.add(record)
            await session.commit()
            await session.refresh(record)
            trade_id = record.id

        self._open_positions[trade_data["symbol"]] = {**trade_data, "id": trade_id}
        if self._redis:
            await self._redis.set(
                "portfolio:positions", json.dumps(self._open_positions)
            )
        return trade_id

    async def update_position_field(self, symbol: str, field: str, value) -> None:
        """Update a single field on an in-memory position (e.g. stop_loss after trailing).
        Does NOT write to DB — the correct SL is tracked on the exchange side."""
        if symbol in self._open_positions:
            self._open_positions[symbol][field] = value
            if self._redis:
                await self._redis.set(
                    "portfolio:positions", json.dumps(self._open_positions)
                )

    async def close_position(
        self, symbol: str, exit_price: float, pnl: float, funding_fee: float = 0.0
    ) -> None:
        # Use the specific trade_id from in-memory position to avoid closing duplicate
        # DB records if the same symbol was opened twice (can happen in training mode).
        trade_id = self._open_positions.get(symbol, {}).get("id")
        async with self._session_factory() as session:
            from sqlalchemy import update
            where_clause = (
                [TradeRecord.id == trade_id, TradeRecord.status == "open"]
                if trade_id
                else [TradeRecord.symbol == symbol, TradeRecord.status == "open"]
            )
            await session.execute(
                update(TradeRecord)
                .where(*where_clause)
                .values(
                    exit_price=exit_price,
                    pnl_usdt=pnl,
                    funding_fee_usdt=round(funding_fee, 4),
                    status="closed",
                    closed_at=datetime.utcnow(),
                )
            )
            await session.commit()

        self._open_positions.pop(symbol, None)
        if self._redis:
            await self._redis.set(
                "portfolio:positions", json.dumps(self._open_positions)
            )
        # Update equity balance immediately so metrics reflect realized PnL
        self._balance = max(0.0, self._balance + pnl)
        if self._redis:
            await self._redis.set("portfolio:balance", self._balance)
        logger.info("Position closed: {} PnL={:.2f} | balance={:.2f}", symbol, pnl, self._balance)
        # Persist daily performance snapshot so /api/performance-history has fresh data
        try:
            await self.save_performance_snapshot()
        except Exception as exc:
            logger.debug("save_performance_snapshot error (non-fatal): {}", exc)

    async def get_closed_trades_pnl(self, trade_ids: List[int]) -> Dict[int, float]:
        """Batch-fetch pnl_usdt for a list of closed trade DB ids.
        Returns {trade_id: pnl} — only includes ids that are actually closed."""
        if not trade_ids:
            return {}
        from sqlalchemy import select
        async with self._session_factory() as session:
            result = await session.execute(
                select(TradeRecord.id, TradeRecord.pnl_usdt)
                .where(TradeRecord.id.in_(trade_ids))
                .where(TradeRecord.status == "closed")
            )
            rows = result.fetchall()
        return {row[0]: float(row[1]) for row in rows if row[1] is not None}

    # ── Performance metrics ───────────────────────────────────────────────
    async def calculate_metrics(self) -> dict:
        async with self._session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TradeRecord).where(TradeRecord.status == "closed")
            )
            trades = result.scalars().all()

        # Always return base metrics so dashboard and WS broadcast work from startup
        base: dict = {
            "total_trades": 0,
            "open_trades": len(self._open_positions),
            "wins_count": 0,
            "losses_count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "balance": self._balance,
            "pnl_trend": [],
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "total_funding_fees": 0.0,
            "manual_balance_active": self._manual_balance_override,
        }
        if not trades:
            return base

        pnls = [t.pnl_usdt for t in trades if t.pnl_usdt is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]   # exclude break-even (pnl=0) from loss count

        win_rate = len(wins) / len(pnls) * 100
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        # Cap at 99.99 when no losses (avoids JSON serialization of float('inf'))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.99 if gross_profit > 0 else 0.0)

        # Max drawdown — equity curve starts from initial balance, not 0
        # This correctly captures drawdown even when all trades are losing
        initial_balance = settings.account_balance_usdt
        cumulative = [initial_balance + sum(pnls[:i+1]) for i in range(len(pnls))]
        peak = initial_balance
        max_dd = 0.0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Use UTC date to match UTC-stored closed_at timestamps
        today_utc = datetime.utcnow().date()
        daily_pnl = sum(
            t.pnl_usdt for t in trades
            if t.closed_at and t.closed_at.date() == today_utc
        )

        # Last 30 closed trades sorted by close time — feeds the PnL trend chart
        recent_30 = sorted(
            [t for t in trades if t.closed_at],
            key=lambda t: t.closed_at
        )[-30:]
        pnl_trend = [
            {"label": t.symbol, "pnl": round(t.pnl_usdt, 2)}
            for t in recent_30
        ]

        return {
            "total_trades": len(trades),
            "open_trades": len(self._open_positions),
            "wins_count": len(wins),
            "losses_count": len(losses),
            "win_rate": round(win_rate, 2),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_dd, 2),
            "daily_pnl": round(daily_pnl, 2),
            "total_pnl": round(sum(pnls), 2),
            "balance": self._balance,
            "pnl_trend": pnl_trend,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "total_funding_fees": round(sum(t.funding_fee_usdt or 0.0 for t in trades), 4),
            "manual_balance_active": self._manual_balance_override,
        }

    async def save_performance_snapshot(self) -> None:
        """Persist today's performance to performance_snapshots (one row per UTC day).
        Called after every trade closes so the table has a full daily history."""
        m = await self.calculate_metrics()
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        from sqlalchemy import select
        async with self._session_factory() as session:
            # Upsert — update today's row if it exists, otherwise insert
            result = await session.execute(
                select(PerformanceSnapshot).where(
                    PerformanceSnapshot.snapshot_date == today
                )
            )
            row = result.scalar_one_or_none()
            total = m.get("wins_count", 0) + m.get("losses_count", 0)
            pf = m.get("profit_factor", 0.0)
            if row:
                from sqlalchemy import update as _upd
                await session.execute(
                    _upd(PerformanceSnapshot)
                    .where(PerformanceSnapshot.snapshot_date == today)
                    .values(
                        balance=m.get("balance", 0.0),
                        daily_pnl=m.get("daily_pnl", 0.0),
                        total_trades=total,
                        winning_trades=m.get("wins_count", 0),
                        losing_trades=m.get("losses_count", 0),
                        win_rate=m.get("win_rate", 0.0),
                        profit_factor=pf,
                        max_drawdown=m.get("max_drawdown_pct", 0.0),
                    )
                )
            else:
                session.add(PerformanceSnapshot(
                    snapshot_date=today,
                    balance=m.get("balance", 0.0),
                    daily_pnl=m.get("daily_pnl", 0.0),
                    total_trades=total,
                    winning_trades=m.get("wins_count", 0),
                    losing_trades=m.get("losses_count", 0),
                    win_rate=m.get("win_rate", 0.0),
                    profit_factor=pf,
                    max_drawdown=m.get("max_drawdown_pct", 0.0),
                ))
            await session.commit()

    async def get_performance_history(self, days: int = 90) -> list:
        """Return daily performance snapshots sorted oldest-first (for charts)."""
        from sqlalchemy import select
        async with self._session_factory() as session:
            result = await session.execute(
                select(PerformanceSnapshot)
                .order_by(PerformanceSnapshot.snapshot_date.desc())
                .limit(days)
            )
            rows = result.scalars().all()
        return [
            {
                "date":           r.snapshot_date.strftime("%Y-%m-%d"),
                "balance":        round(r.balance, 2),
                "daily_pnl":      round(r.daily_pnl, 2),
                "total_trades":   r.total_trades,
                "winning_trades": r.winning_trades,
                "losing_trades":  r.losing_trades,
                "win_rate":       round(r.win_rate, 2),
                "profit_factor":  round(r.profit_factor, 2),
                "max_drawdown":   round(r.max_drawdown, 2),
            }
            for r in reversed(rows)
        ]

    async def get_risk_exposure(self) -> dict:
        positions = await self.get_open_positions()
        total_notional = sum(
            p.get("position_size_usdt", 0) for p in positions.values()
        )
        exposure_pct = total_notional / self._balance * 100 if self._balance else 0
        return {
            "open_trades": len(positions),
            "max_open_trades": settings.effective_max_open_trades,
            "total_notional_usdt": round(total_notional, 2),
            "exposure_pct": round(exposure_pct, 2),
        }

    async def reset_session(self, backup_dir: str = "backups") -> dict:
        """Backup + wipe trade history and performance data for a fresh session.

        Preserved (NOT touched): ml_training_samples, coin_stats, research_log.
        Open trade records are kept intact — they mirror live exchange positions.

        Returns dict with backup filename and counts.
        """
        import os
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"session_backup_{ts}.json")

        # Snapshot everything before deletion
        async with self._session_factory() as session:
            from sqlalchemy import select
            trades_res = await session.execute(select(TradeRecord))
            all_trades = trades_res.scalars().all()
            snaps_res = await session.execute(select(PerformanceSnapshot))
            all_snaps = snaps_res.scalars().all()

        def _ser(v):
            return v.isoformat() if isinstance(v, datetime) else v

        backup_data = {
            "timestamp": ts,
            "trades": [
                {c.name: _ser(getattr(t, c.name)) for c in TradeRecord.__table__.columns}
                for t in all_trades
            ],
            "performance_snapshots": [
                {c.name: _ser(getattr(s, c.name)) for c in PerformanceSnapshot.__table__.columns}
                for s in all_snaps
            ],
        }
        backup_str = json.dumps(backup_data, indent=2)
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: open(backup_file, "w", encoding="utf-8").write(backup_str)
        )

        # Delete closed/cancelled trades + all performance snapshots
        async with self._session_factory() as session:
            await session.execute(text("DELETE FROM performance_snapshots"))
            await session.execute(
                text("DELETE FROM trades WHERE status IN ('closed', 'cancelled')")
            )
            await session.commit()

        # Clear any cached aggregate metrics from Redis (balance + positions keys are kept)
        if self._redis:
            try:
                await self._redis.delete("portfolio:metrics")
            except Exception:
                pass

        n_trades = len(backup_data["trades"])
        n_snaps = len(backup_data["performance_snapshots"])
        logger.info(
            "Session reset: backed up {} trades + {} snapshots → {}", n_trades, n_snaps, backup_file
        )
        return {"backup_file": backup_file, "trades_backed_up": n_trades, "snapshots_backed_up": n_snaps}
