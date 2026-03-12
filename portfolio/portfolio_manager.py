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
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
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
    status = Column(String(10), default="open")   # open | closed | cancelled
    exchange = Column(String(10), default="binance")
    order_id = Column(String(50), nullable=True)
    signal_score = Column(Integer, default=0)
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
        self._open_positions: Dict[str, dict] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────
    async def start(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            self._redis = await aioredis.from_url(settings.redis_url)
            await self._redis.ping()
            logger.info("PortfolioManager started (Redis connected)")
        except Exception as exc:
            logger.warning("Redis unavailable ({}), running in-memory mode", exc)
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
        self._balance = balance
        if self._redis:
            await self._redis.set("portfolio:balance", balance)

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
                if k in TradeRecord.__table__.columns.keys()
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
        self, symbol: str, exit_price: float, pnl: float
    ) -> None:
        async with self._session_factory() as session:
            from sqlalchemy import update
            await session.execute(
                update(TradeRecord)
                .where(
                    TradeRecord.symbol == symbol,
                    TradeRecord.status == "open",
                )
                .values(
                    exit_price=exit_price,
                    pnl_usdt=pnl,
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
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "balance": self._balance,
            "pnl_trend": [],
        }
        if not trades:
            return base

        pnls = [t.pnl_usdt for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

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
            "win_rate": round(win_rate, 2),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_dd, 2),
            "daily_pnl": round(daily_pnl, 2),
            "total_pnl": round(sum(pnls), 2),
            "balance": self._balance,
            "pnl_trend": pnl_trend,
        }

    async def get_risk_exposure(self) -> dict:
        positions = await self.get_open_positions()
        total_notional = sum(
            p.get("position_size_usdt", 0) for p in positions.values()
        )
        exposure_pct = total_notional / self._balance * 100 if self._balance else 0
        return {
            "open_trades": len(positions),
            "total_notional_usdt": round(total_notional, 2),
            "exposure_pct": round(exposure_pct, 2),
        }
