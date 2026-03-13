"""
trading/risk_manager.py — Strict risk management engine.

Rules enforced:
  ✓ Risk per trade = 1 % of FREE balance (total balance minus margin in open positions)
  ✓ Maximum leverage = 3×
  ✓ Maximum open trades = 5
  ✓ Maximum trades per day = 3 (configurable)
  ✓ Minimum risk-reward = 1:2
  ✓ Auto stop-loss and take-profit calculation
  ✓ Position-size calculation (USDT notional)
  ✓ Margin tracking: prevents over-allocation across concurrent positions
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
        self._used_margin: float = 0.0    # margin committed to open/pending positions
        self._daily_trades: int = 0
        self._daily_date: date = date.today()
        self._daily_pnl: float = 0.0      # realized PnL for today (resets each UTC day)

    # ── Public ────────────────────────────────────────────────────────────
    def update_balance(self, balance: float) -> None:
        self._account_balance = balance
        # NOTE: do NOT auto-reset _used_margin here — dry-run relies on it for
        # accurate free-balance tracking. Live callers that fetch availableBalance
        # from the exchange should call reset_margin() explicitly right after.

    def reset_margin(self) -> None:
        """Reset in-memory margin tracking.
        Call only in live mode right after updating balance from exchange
        availableBalance — that value already excludes committed margin, so
        keeping _used_margin would cause double-counting."""
        self._used_margin = 0.0

    def reset_all_state(self) -> None:
        """Reset all runtime counters for a clean session start.
        Called by the dashboard Reset Session action.
        Does NOT change _account_balance — that is re-synced from the exchange."""
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_date = date.today()
        self._used_margin = 0.0
        self._open_trades = 0
        logger.info("RiskManager: all runtime counters reset")

    def add_margin(self, margin_usdt: float) -> None:
        """Reserve margin when a position opens or a LIMIT order is placed."""
        self._used_margin += max(0.0, margin_usdt)

    def release_margin(self, margin_usdt: float) -> None:
        """Free margin when a position closes or a LIMIT order is cancelled."""
        self._used_margin = max(0.0, self._used_margin - margin_usdt)

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

    def can_open_trade(self, silent: bool = False) -> bool:
        self._reset_daily_if_needed()
        if self._open_trades >= settings.effective_max_open_trades:
            if not silent:
                logger.warning(
                    "Max open trades reached ({}/{})",
                    self._open_trades, settings.effective_max_open_trades,
                )
            return False
        if self._daily_trades >= settings.effective_max_daily_trades:
            if not silent:
                logger.warning(
                    "Daily trade limit reached ({}/{}). Resuming tomorrow.",
                    self._daily_trades, settings.effective_max_daily_trades,
                )
            return False
        if self.daily_loss_exceeded:
            if not silent:
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

        # ── Minimum SL distance (prevents ultra-tight SL → oversized position) ──
        # A SL too close to entry forces a huge notional to achieve the 1% risk budget,
        # which in turn consumes most/all of the available balance as margin.
        if settings.min_stop_loss_pct > 0:
            min_sl_dist = entry_price * (settings.min_stop_loss_pct / 100.0)
            if direction == "LONG" and (entry_price - stop_loss) < min_sl_dist:
                stop_loss = entry_price - min_sl_dist
                logger.debug("{} LONG SL too tight — widened to {:.6f} ({:.2f}% from entry)",
                             symbol, stop_loss, settings.min_stop_loss_pct)
            elif direction == "SHORT" and (stop_loss - entry_price) < min_sl_dist:
                stop_loss = entry_price + min_sl_dist
                logger.debug("{} SHORT SL too tight — widened to {:.6f} ({:.2f}% from entry)",
                             symbol, stop_loss, settings.min_stop_loss_pct)

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
        if round(rr, 4) < settings.min_risk_reward_ratio:
            logger.info(
                "{} RR {:.2f} < min {:.2f} — skip",
                symbol, rr, settings.min_risk_reward_ratio,
            )
            return None

        # ── Position sizing ───────────────────────────────────────────────
        # Step 1: Reserve balance_reserve_pct% of total balance as a permanent
        # anti-liquidation buffer.  Even if all 5 trade slots are filled, this
        # fraction of the account is NEVER touched by margin.
        # Purpose: covers unrealized losses, funding fees, maintenance margin,
        # and prevents cross-margin cascades.
        # Example: balance=1000, reserve=50% → tradeable=500 USDT.
        #   5 trades × 10% cap × 500 = 250 USDT max cumulative margin (25% of total).
        tradeable_balance = self._account_balance * (
            1.0 - settings.balance_reserve_pct / 100.0
        )

        # Step 2: Further subtract already-committed margin from in-flight positions.
        # This prevents concurrent signals within the same scan cycle from all
        # sizing against the same tradeable balance.
        free_balance = max(0.0, tradeable_balance - self._used_margin)

        if free_balance < 10.0:
            logger.warning(
                "{} insufficient free balance: {:.2f} USDT free "
                "({:.2f} used margin | {:.0f}%% reserve on {:.2f} USDT total) — skip",
                symbol, free_balance, self._used_margin,
                settings.balance_reserve_pct, self._account_balance,
            )
            return None

        risk_usdt = free_balance * (settings.risk_per_trade_pct / 100.0)
        leverage = settings.max_leverage

        # Position size in quote currency (risk-based)
        risk_pct_of_entry = risk / entry_price
        position_size_usdt = risk_usdt / risk_pct_of_entry

        # ── Hard cap: max margin per trade = max_position_size_pct % of free tradeable balance ──
        # Prevents oversized positions when SL is tight.
        # Example: reserve=50%, balance=1000, cap=10%, leverage=3
        #   → tradeable=500, max notional = 500 × 10% × 3 = 150 USDT, max margin = 50 USDT
        max_notional = free_balance * (settings.max_position_size_pct / 100.0) * leverage
        if position_size_usdt > max_notional:
            logger.debug(
                "{} position capped: {:.2f} → {:.2f} USDT notional "
                "(max_position_size_pct={:.0f}%% of {:.2f} USDT free)",
                symbol, position_size_usdt, max_notional,
                settings.max_position_size_pct, free_balance,
            )
            position_size_usdt = max_notional

        # ── Fee awareness ─────────────────────────────────────────────────
        # A. Taker round-trip: entry + exit taker fee × notional.
        round_trip_fee_usdt = position_size_usdt * (settings.taker_fee_pct / 100.0) * 2.0

        # B. Funding fee estimate: worst-case rate × notional × number of 8-h periods.
        # A position held up to max_position_hold_hours pays ceil(hold/8) funding
        # settlements.  We budget for funding_periods_estimate (default 3) periods
        # at max_funding_rate_pct (default 0.10%) per period.
        # This prevents the case where funding silently drains more than the risk budget.
        estimated_funding_usdt = (
            position_size_usdt
            * (settings.max_funding_rate_pct / 100.0)
            * settings.funding_periods_estimate
        )

        total_fees_usdt = round_trip_fee_usdt + estimated_funding_usdt
        fee_ratio = total_fees_usdt / risk_usdt if risk_usdt > 0 else 0.0
        if fee_ratio > 0.20:
            logger.warning(
                "{} total estimated fees {:.4f} USDT "
                "(taker={:.4f} + funding_est={:.4f}) = {:.1f}%% of {:.4f} USDT risk — "
                "SL may be too tight or balance too small",
                symbol, total_fees_usdt,
                round_trip_fee_usdt, estimated_funding_usdt,
                fee_ratio * 100, risk_usdt,
            )

        # ── Final margin safety check ─────────────────────────────────────
        # Ensure margin + all estimated fees do not exceed 95% of free tradeable balance.
        # The 5% micro-buffer guards against exchange fee rounding and funding
        # settlements that arrive slightly before a position closes.
        margin_required = position_size_usdt / leverage
        total_committed = margin_required + total_fees_usdt
        if total_committed > free_balance * 0.95:
            logger.warning(
                "{} margin ({:.2f}) + fees ({:.4f}) = {:.2f} USDT > 95%% of free "
                "tradeable balance ({:.2f} USDT) — position reduced",
                symbol, margin_required, total_fees_usdt,
                total_committed, free_balance,
            )
            safe_margin = free_balance * 0.95 - total_fees_usdt
            if safe_margin <= 0:
                logger.warning("{} not enough free balance after fee reserve — skip", symbol)
                return None
            position_size_usdt = safe_margin * leverage

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
            "RiskParams | {} {} | entry={} SL={} TP={} qty={} lev={} RR={} | "
            "risk=${:.2f}  free=${:.2f} (reserve={:.0f}%% of ${:.2f})  "
            "notional=${:.2f}  margin=${:.2f}  taker≈${:.4f}  funding_est≈${:.4f}",
            symbol, direction, entry_price, stop_loss, take_profit,
            quantity, leverage, round(rr, 2),
            risk_usdt, free_balance,
            settings.balance_reserve_pct, self._account_balance,
            round(position_size_usdt, 2),
            round(position_size_usdt / leverage, 2),
            round(round_trip_fee_usdt, 4),
            round(estimated_funding_usdt, 4),
        )
        return params

    def adjust_sl_to_breakeven(self, entry: float) -> float:
        """Move SL to entry after partial profit realised."""
        return entry
