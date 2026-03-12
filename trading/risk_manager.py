"""
trading/risk_manager.py — Strict risk management engine.

Rules enforced:
  ✓ Risk per trade = 1 % account balance
  ✓ Maximum leverage = 3×
  ✓ Maximum open trades = 5
  ✓ Maximum trades per day = 3 (configurable)
  ✓ Minimum risk-reward = 1:2
  ✓ Auto stop-loss and take-profit calculation
  ✓ Position-size calculation (USDT notional)
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional
from loguru import logger

from config import settings


@dataclass
class RiskParameters:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_usdt: float      # notional USDT per trade
    quantity: float                # base-asset quantity
    leverage: int
    risk_amount_usdt: float        # $ at risk
    risk_reward_ratio: float


class RiskManager:
    """
    Validates a trade setup and computes safe position sizing.
    """

    def __init__(self) -> None:
        self._open_trades: int = 0
        self._account_balance: float = settings.account_balance_usdt
        self._daily_trades: int = 0
        self._daily_date: date = date.today()
        self._daily_pnl: float = 0.0      # realized PnL for today (resets each UTC day)

    # ── Public ────────────────────────────────────────────────────────────
    def update_balance(self, balance: float) -> None:
        self._account_balance = balance

    def update_open_trades(self, count: int) -> None:
        self._open_trades = count

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if today != self._daily_date:
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._daily_date = today

    def record_trade_opened(self) -> None:
        """Call once per trade successfully opened."""
        self._reset_daily_if_needed()
        self._daily_trades += 1

    def record_trade_closed(self, pnl: float) -> None:
        """Call every time a position closes to track daily realized PnL."""
        self._reset_daily_if_needed()
        self._daily_pnl += pnl

    @property
    def daily_loss_exceeded(self) -> bool:
        """True when today's realized loss has hit the configured limit."""
        self._reset_daily_if_needed()
        max_loss_pct = 0.0 if settings.training_mode else settings.max_daily_loss_pct
        if max_loss_pct <= 0:
            return False
        loss_limit = self._account_balance * (max_loss_pct / 100.0)
        return self._daily_pnl <= -loss_limit

    @property
    def daily_trades_remaining(self) -> int:
        self._reset_daily_if_needed()
        return max(0, settings.effective_max_daily_trades - self._daily_trades)

    def can_open_trade(self) -> bool:
        self._reset_daily_if_needed()
        if self._open_trades >= settings.effective_max_open_trades:
            logger.warning(
                "Max open trades reached ({}/{})",
                self._open_trades, settings.effective_max_open_trades,
            )
            return False
        if self._daily_trades >= settings.effective_max_daily_trades:
            logger.warning(
                "Daily trade limit reached ({}/{}). Resuming tomorrow.",
                self._daily_trades, settings.effective_max_daily_trades,
            )
            return False
        if self.daily_loss_exceeded:
            logger.warning(
                "Daily loss limit reached ({:.2f} USDT = {:.1f}% of balance). No new trades today.",
                abs(self._daily_pnl), settings.max_daily_loss_pct,
            )
            return False
        return True

    def calculate_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[RiskParameters]:
        """
        Returns validated RiskParameters or None if rules are violated.
        """
        # ── Validate stop-loss direction ──────────────────────────────────
        if direction == "LONG" and stop_loss >= entry_price:
            logger.warning("{} LONG stop ({}) must be below entry ({})",
                           symbol, stop_loss, entry_price)
            return None
        if direction == "SHORT" and stop_loss <= entry_price:
            logger.warning("{} SHORT stop ({}) must be above entry ({})",
                           symbol, stop_loss, entry_price)
            return None

        # ── Stop-loss cap ─────────────────────────────────────────────────────
        # Enforce maximum SL distance so the strategy can never risk more than
        # max_stop_loss_pct % of entry, even if a technical level is further away.
        if settings.max_stop_loss_pct > 0:
            max_sl_dist = entry_price * (settings.max_stop_loss_pct / 100.0)
            if direction == "LONG":
                stop_loss = max(stop_loss, entry_price - max_sl_dist)
            else:
                stop_loss = min(stop_loss, entry_price + max_sl_dist)

        # ── Risk-reward check ─────────────────────────────────────────────
        if direction == "LONG":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        if risk <= 0:
            logger.warning("{} zero risk — skip", symbol)
            return None

        rr = reward / risk
        if rr < settings.min_risk_reward_ratio:
            logger.debug(
                "{} RR {:.2f} < minimum {:.2f} — skip",
                symbol, rr, settings.min_risk_reward_ratio,
            )
            return None

        # ── Position sizing ───────────────────────────────────────────────
        risk_usdt = self._account_balance * (settings.risk_per_trade_pct / 100.0)
        leverage = settings.max_leverage

        # Position size in quote currency
        risk_pct_of_entry = risk / entry_price
        position_size_usdt = risk_usdt / risk_pct_of_entry
        position_size_usdt = min(position_size_usdt, self._account_balance * leverage)

        quantity = position_size_usdt / entry_price

        params = RiskParameters(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            position_size_usdt=round(position_size_usdt, 2),
            quantity=round(quantity, 6),
            leverage=leverage,
            risk_amount_usdt=round(risk_usdt, 2),
            risk_reward_ratio=round(rr, 2),
        )

        logger.info(
            "RiskParams | {} {} | entry={} SL={} TP={} qty={} lev={} RR={}",
            symbol, direction, entry_price, stop_loss, take_profit,
            quantity, leverage, round(rr, 2),
        )
        return params

    def adjust_sl_to_breakeven(self, entry: float) -> float:
        """Move SL to entry after partial profit realised."""
        return entry
