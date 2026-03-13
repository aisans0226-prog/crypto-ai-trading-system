"""
main.py — Crypto AI Trading System entry point.

Starts all subsystems concurrently:
  ✓ Market data engine (REST + WebSocket)
  ✓ Market scanner (500 coins, 60-second cycle)
  ✓ ML prediction engine
  ✓ Risk manager + trade executor
  ✓ Portfolio manager
  ✓ Alert system
  ✓ FastAPI dashboard

Trade execution flow (new):
  1. Scanner detects high-score signal → added to watchlist → Telegram alert
  2. Signal must persist N scan cycles (watchlist_confirmations) → confirmation
  3. After confirmation → ResearchEngine runs 15m/1h/4h multi-timeframe analysis
  4. Research result sent to Telegram (passed/failed)
  5. If passed → trade entry notification → execute trade
  6. After trade close → CoinDatabase updated (win/loss stats for future learning)

Usage:
    python main.py
    python main.py --exchange binance          # default
    python main.py --exchange bybit
    python main.py --dry-run                   # paper trading (no real orders)
"""
import asyncio
import argparse
import sys
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import uvicorn
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from utils.logger import setup_logger

from data_engine.market_data import MarketDataEngine
from data_engine.websocket_feed import BinanceWebSocketFeed, MarketCache
from data_engine.coin_database import CoinDatabase

from scanners.market_scanner import MarketScanner
from scanners.research_engine import ResearchEngine
from ml_models.prediction_model import EnsemblePredictor
from ml_models.self_learning import SelfLearningEngine
from ml_models.coin_ranker import CoinRanker

from strategy.strategy_registry import StrategyRegistry
from strategy.trend_strategy import TradeSetup

from trading.risk_manager import RiskManager
from trading.trade_executor import BinanceExecutor, BybitExecutor
from trading.arbitrage_engine import ArbitrageEngine

from portfolio.portfolio_manager import PortfolioManager
from alerts.telegram_bot import AlertSystem
from social_ai.sentiment_analyzer import SentimentAnalyzer
from updater.auto_updater import AutoUpdater

from dashboard.api_server import app, state, broadcast


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist entry — tracks a candidate signal awaiting research clearance
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WatchlistEntry:
    signal: object          # SignalResult
    first_seen: float       # time.time() when first detected
    confirmations: int = 0  # number of consecutive scan cycles signal remained high


@dataclass
class PendingEntry:
    """In-flight LIMIT order awaiting exchange fill (or cancel + retry)."""
    symbol: str
    order_id: str
    placed_at: float          # time.time() when the LIMIT order was placed
    risk_params: object       # RiskParameters
    signal: object            # SignalResult
    retry_count: int          # how many times we've re-placed this entry
    research_id: Optional[int]
    research_score: float
    strategy_name: Optional[str]
    limit_price: float = 0.0  # target fill price (used for dry-run simulation)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
class TradingSystem:
    def __init__(self, exchange: str = "binance", dry_run: bool = False) -> None:
        self.exchange = exchange
        self.dry_run = dry_run
        self._running = False

        # Core components
        self._data_engine = MarketDataEngine()
        self._cache = MarketCache()
        self._scanner = MarketScanner(self._data_engine)
        self._ml = EnsemblePredictor()
        self._risk = RiskManager()
        self._portfolio = PortfolioManager()
        self._alerts = AlertSystem()
        self._sentiment = SentimentAnalyzer()
        self._arb = ArbitrageEngine(self._data_engine)

        # AI self-learning and ranking
        self._learner = SelfLearningEngine()
        self._ranker = CoinRanker(self._learner)
        self._learner.set_predictor(self._ml)   # enable hot-reload after retrain
        self._updater = AutoUpdater()

        # Self-learning tracking
        self._pending_labels: Dict[str, str] = {}     # symbol -> trade_id
        self._prev_open_positions: Dict[str, dict] = {}  # symbol -> position snapshot
        self._klines_cache: Dict[str, object] = {}   # symbol -> DataFrame for ranking
        self._ml_predictions_cache: Dict[str, float] = {}  # symbol -> ml confidence
        self._signal_cooldowns: Dict[str, float] = {}      # symbol -> last decision timestamp

        # Per-position runtime metadata for smart exit management
        # Stored in-memory only (not persisted); keyed by symbol
        self._position_meta: Dict[str, dict] = {}    # symbol -> {opened_at, peak_price, sl_order_id, ...}

        # Watchlist + research phase
        self._watchlist: Dict[str, WatchlistEntry] = {}    # symbol -> entry awaiting research
        self._pending_research_ids: Dict[str, int] = {}    # symbol -> research_log id (for outcome)
        self._coin_db = CoinDatabase()
        self._research: Optional[ResearchEngine] = None    # initialised in start()

        # Pending LIMIT entry orders (LIMIT mode only): tracked until fill or cancel
        self._pending_entry_orders: Dict[str, PendingEntry] = {}  # symbol -> PendingEntry

        # WS feed handle (populated in start())
        self._ws_feed = None

        # Bot on/off control (toggled via Telegram commands)
        self._bot_paused: bool = False

        # Strategy registry — auto-selects best strategy per trade (set in start())
        self._strategy_registry: Optional[StrategyRegistry] = None

        # Exchange executors
        self._binance_exec = BinanceExecutor()
        self._bybit_exec = BybitExecutor()

        # Inject into dashboard state
        state.portfolio_manager = self._portfolio
        state.market_scanner = self._scanner
        state.self_learner = self._learner
        state.coin_ranker = self._ranker
        state.auto_updater = self._updater
        state.coin_database = self._coin_db
        state.strategy_registry = None     # set after StrategyRegistry is initialised in start()

    # ── Lifecycle ─────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._running = True
        logger.info(
            "Starting Crypto AI Trading System | exchange={} dry_run={}",
            self.exchange, self.dry_run,
        )

        # Start subsystems
        await self._data_engine.start()
        await self._portfolio.start()
        await self._coin_db.start()
        await self._learner.initialize(self._coin_db)   # load ML training data from PostgreSQL
        await self._recover_ml_labels_after_restart()   # re-link & auto-label SIGKILL orphans
        await self._alerts.start()
        self._alerts.set_command_handler(self._handle_telegram_command)
        await self._sentiment.start()
        await self._binance_exec.start()
        await self._bybit_exec.start()
        self._research = ResearchEngine(self._data_engine, self._coin_db)
        self._strategy_registry = StrategyRegistry(self._coin_db)
        state.strategy_registry = self._strategy_registry
        state.pending_entry_orders = self._pending_entry_orders  # live dict ref — updated in-place

        # Fetch symbol list — may fail if exchange API is geo-blocked
        try:
            symbols = await self._data_engine.get_binance_symbols()
            symbols = symbols[: settings.max_coins_to_scan]
        except Exception as exc:
            logger.warning(
                "Cannot reach Binance API ({}). "
                "Running portfolio sync + updater only — no scanning.",
                exc,
            )
            # Keep portfolio sync and auto-updater alive; skip exchange loops
            await asyncio.gather(
                self._portfolio_sync_loop(),
                self._updater.start_polling(),
                return_exceptions=True,
            )
            return

        self._ws_feed = BinanceWebSocketFeed(self._cache, symbols)
        self._cache.register_callback(self._on_ws_event)

        logger.info("System ready — scanning {} coins", len(symbols))
        online_msg = (
            f"🟢 Trading system ONLINE\n"
            f"Exchange: {self.exchange.upper()}\n"
            f"Dry run: {self.dry_run}\n"
            f"Scanning: {len(symbols)} coins"
        )
        if settings.training_mode:
            online_msg += (
                "\n\n⚠️ *TRAINING MODE ACTIVE*\n"
                f"Score gate: ≥{settings.effective_signal_score_threshold} "
                f"(normal={settings.signal_score_threshold})\n"
                f"Max daily trades: {settings.effective_max_daily_trades}\n"
                f"Max open: {settings.effective_max_open_trades}\n"
                f"Hold limit: {settings.effective_max_position_hold_hours}h\n"
                "All thresholds relaxed for ML data collection."
            )
            logger.warning(
                "⚠️  TRAINING MODE — relaxed thresholds: score≥{} confirmations={} "
                "cooldown={}min daily_max={} open_max={} hold={}h volume≥{}",
                settings.effective_signal_score_threshold,
                settings.effective_watchlist_confirmations,
                settings.effective_signal_cooldown_minutes,
                settings.effective_max_daily_trades,
                settings.effective_max_open_trades,
                settings.effective_max_position_hold_hours,
                settings.effective_min_volume_usdt,
            )
        await self._alerts.send_text(online_msg)

        # Run all async tasks
        await asyncio.gather(
            self._ws_feed.start(),
            self._scan_loop(),
            self._arbitrage_loop(symbols),
            self._portfolio_sync_loop(),
            self._position_monitor_loop(),
            self._limit_order_monitor_loop(),
            self._coin_ranking_loop(),
            self._updater.start_polling(),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        self._running = False
        if self._ws_feed:
            await self._ws_feed.stop()
        await self._data_engine.stop()
        await self._alerts.stop()
        await self._sentiment.stop()
        await self._binance_exec.stop()
        await self._bybit_exec.stop()
        await self._portfolio.stop()
        await self._updater.stop()
        logger.info("System stopped cleanly")

    # ── WS event callback ─────────────────────────────────────────────────
    async def _on_ws_event(self, event: str, payload: dict) -> None:
        if event == "kline_closed":
            sym = payload.get("symbol", "")
            kline = self._cache.klines.get(sym)
            if kline:
                await broadcast("kline", kline)

    # ── Telegram command handler ───────────────────────────────────────────
    async def _handle_telegram_command(self, cmd: str) -> None:
        """Handle /pause, /resume, /mute, /unmute, /status, /help commands from Telegram."""
        if cmd in ("pause", "stop_trading"):
            self._bot_paused = True
            await self._alerts.send_text(
                "⏸ *Bot trading PAUSED*\n\n"
                "Scanning continues but no new trades will be opened.\n"
                "Send /resume to re-enable trading."
            )
            logger.info("Bot trading paused via Telegram command")

        elif cmd == "resume":
            self._bot_paused = False
            await self._alerts.send_text(
                "▶️ *Bot trading RESUMED*\n\n"
                "Scanning and trade execution are active again."
            )
            logger.info("Bot trading resumed via Telegram command")

        elif cmd == "mute":
            # Confirm BEFORE muting so the user receives this last message
            await self._alerts.send_text(
                "🔕 *Notifications MUTED*\n\n"
                "Bot continues scanning and trading normally.\n"
                "No further alerts will be sent until you send /unmute."
            )
            self._alerts.mute()

        elif cmd == "unmute":
            self._alerts.unmute()
            await self._alerts.send_text(
                "🔔 *Notifications RESTORED*\n\n"
                "All trade alerts are active again."
            )

        elif cmd == "status":
            positions = await self._portfolio.get_open_positions()
            trade_label = "⏸ PAUSED (no new trades)" if self._bot_paused else "▶️ RUNNING"
            alert_label = "🔕 MUTED" if self._alerts.is_muted else "🔔 ON"
            await self._alerts.send_text(
                f"📊 *Bot Status*\n\n"
                f"Trading: `{trade_label}`\n"
                f"Alerts: `{alert_label}`\n"
                f"Open positions: `{len(positions)}`\n"
                f"Daily trades left: `{self._risk.daily_trades_remaining}`\n"
                f"Watchlist: `{len(self._watchlist)}` symbols\n"
                f"Exchange: `{self.exchange.upper()}`\n"
                f"Dry run: `{self.dry_run}`"
            )

        elif cmd in ("help", "start"):
            await self._alerts.send_text(
                "🤖 *Bot Commands*\n\n"
                "/status — show current bot status\n"
                "/pause — pause trading (no new trades)\n"
                "/resume — resume trading\n"
                "/mute — silence all notifications (trading continues)\n"
                "/unmute — restore notifications\n"
                "/help — show this message"
            )

        else:
            await self._alerts.send_text(
                f"❓ Unknown command: `/{cmd}`\n"
                "Send /help for available commands."
            )

    # ── ML label recovery after SIGKILL restart ───────────────────────────
    async def _recover_ml_labels_after_restart(self) -> None:
        """Re-link open positions to their ML keys and auto-label orphaned closed ones.

        After a SIGKILL restart, _pending_labels is empty but the DB still holds
        1000s of pending ML samples.  This method:
          1. Restores _pending_labels for symbols that are still open.
          2. Batch-labels closed-trade ML samples so self-training can proceed.
        """
        pending = self._learner._pending
        if not pending:
            return

        open_positions = await self._portfolio.get_open_positions()

        # Build {trade_db_id: ml_key} for samples we can parse
        parseable: Dict[int, str] = {}
        for ml_key in pending:
            try:
                db_id = int(ml_key.rsplit("_", 1)[-1])
                parseable[db_id] = ml_key
            except (ValueError, IndexError):
                pass

        # Batch-query which of those trade_ids are closed (and their PnL)
        closed_pnl: Dict[int, float] = {}
        if parseable:
            try:
                closed_pnl = await self._portfolio.get_closed_trades_pnl(
                    list(parseable.keys())
                )
            except Exception as exc:
                logger.warning("ML label recovery DB query failed: {}", exc)

        restored = labeled = 0
        for ml_key, entry in list(pending.items()):
            symbol = entry["symbol"]

            # Case 1: position still open — restore pending_labels so it labels on close
            if symbol in open_positions:
                if symbol not in self._pending_labels:
                    current_trade_id = open_positions[symbol].get("id")
                    expected_key = f"{symbol}_{current_trade_id}" if current_trade_id else None
                    if expected_key and expected_key == ml_key:
                        self._pending_labels[symbol] = ml_key
                        restored += 1
                continue

            # Case 2: trade is closed — auto-label with DB pnl
            try:
                db_id = int(ml_key.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                continue
            if db_id in closed_pnl:
                self._learner.label_trade(ml_key, closed_pnl[db_id])
                labeled += 1

        if restored or labeled:
            logger.info(
                "ML restart recovery: {} labels restored for open, {} orphans auto-labeled",
                restored, labeled,
            )

    # ── Main scan loop ─────────────────────────────────────────────────────
    async def _scan_loop(self) -> None:
        while self._running:
            try:
                logger.info("── Scan cycle starting ──")
                signals = await self._scanner.scan_all()
                now = time.time()

                # Update full klines cache from bulk scan so CoinRanker can rank
                # ALL volume-filtered coins, not just the handful that hit threshold
                self._klines_cache.update(self._scanner.get_last_klines())

                # Sync funding cache to ResearchEngine so it can apply funding-rate gate
                if self._research:
                    self._research.update_funding_cache(self._scanner.get_last_funding_cache())

                # Re-sort: primary = signal score DESC, secondary = AI rank DESC.
                # When multiple signals tie on score, coins with higher AI grade
                # get processed first and take the limited daily trade slots.
                if state.coin_rankings:
                    _rank_map = {r["symbol"]: r["composite_score"] for r in state.coin_rankings}
                    signals.sort(
                        key=lambda s: (s.score, _rank_map.get(s.symbol, 0.0)),
                        reverse=True,
                    )

                # ── Cap signals per scan to keep each cycle under ~60 seconds.
                # Watchlist coins are always included first (needed for confirmation).
                # Only new signals are capped — top scoring ones take priority.
                _cap = settings.effective_max_signals_per_scan
                _wl_signals  = [s for s in signals if s.symbol in self._watchlist]
                _new_signals = [s for s in signals if s.symbol not in self._watchlist]
                signals = _wl_signals + _new_signals[:_cap]

                # Track which symbols are currently qualifying (for watchlist expiry)
                live_qualifying: set = set()

                for signal in signals:
                    # Pre-filter: skip only signals that can't reach threshold even with
                    # max ML (+3) + sentiment (+1) boost. This allows scanner scores of
                    # (threshold - 4) or above to proceed to ML/sentiment evaluation.
                    pre_filter_score = max(0, settings.effective_signal_score_threshold - 4)
                    if signal.score < pre_filter_score:
                        continue

                    # Cooldown applies only when adding NEW entries to watchlist
                    in_watchlist = signal.symbol in self._watchlist
                    if not in_watchlist:
                        cooldown_secs = settings.effective_signal_cooldown_minutes * 60
                        if now - self._signal_cooldowns.get(signal.symbol, 0) < cooldown_secs:
                            continue

                    # ML prediction boost
                    try:
                        df = self._klines_cache.get(signal.symbol)
                        if df is None or len(df) < 50:
                            df = await self._data_engine.get_klines_binance(
                                signal.symbol, "15m", 200
                            )
                            self._klines_cache[signal.symbol] = df
                        ml_score, confidence = self._ml.predict(df)
                        signal.score += ml_score
                        signal.ai_prediction = confidence
                        signal.confidence = confidence
                        self._ml_predictions_cache[signal.symbol] = confidence
                    except Exception as exc:
                        logger.debug("ML predict error {}: {}", signal.symbol, exc)

                    # Sentiment boost
                    try:
                        sent_score, sent_signals = \
                            await self._sentiment.get_sentiment_score(signal.symbol)
                        if sent_score >= 0.3:
                            signal.score += 1
                            signal.signals.extend(sent_signals)
                    except Exception:
                        pass

                    # Re-check threshold after ML + sentiment
                    if not signal.is_high_probability:
                        continue

                    # ML confidence gate — only enforced when model is trained (not default 0.5)
                    ml_trained = abs(signal.ai_prediction - 0.5) > 0.05
                    if ml_trained and signal.ai_prediction < settings.effective_min_ml_confidence:
                        logger.debug(
                            "{} ML confidence {:.2f} < gate {:.2f} — skip",
                            signal.symbol, signal.ai_prediction, settings.effective_min_ml_confidence,
                        )
                        continue

                    live_qualifying.add(signal.symbol)

                    # Log to coin database for self-learning
                    await self._coin_db.record_signal(signal.symbol, signal.score)

                    # Update dashboard state
                    state.recent_signals.append({
                        "symbol":          signal.symbol,
                        "score":           signal.score,
                        "direction":       signal.direction,
                        "signals":         signal.signals,
                        "price":           signal.price,
                        "volume_24h":      signal.volume_24h,
                        "price_change_pct": signal.price_change_pct,
                        "ai_prediction":   signal.ai_prediction,
                    })
                    if len(state.recent_signals) > 500:
                        state.recent_signals = state.recent_signals[-500:]

                    # Skip watchlist and trading while bot is paused
                    if self._bot_paused:
                        continue

                    if not in_watchlist:
                        # ── Stage 1: Add to watchlist ──────────────────────
                        self._watchlist[signal.symbol] = WatchlistEntry(
                            signal=signal, first_seen=now
                        )
                        await self._alerts.send_watchlist_alert(
                            signal, settings.effective_watchlist_confirmations
                        )
                        await broadcast("signal", {
                            "symbol": signal.symbol, "score": signal.score,
                            "stage": "watchlist",
                        })
                        logger.info(
                            "📡 Watchlist add: {} | score={} | needs {} more scan(s)",
                            signal.symbol, signal.score, settings.effective_watchlist_confirmations,
                        )
                    else:
                        # ── Stage 2: Increment confirmation count ──────────
                        entry = self._watchlist[signal.symbol]
                        entry.signal = signal        # refresh with latest scores
                        entry.confirmations += 1
                        logger.info(
                            "📡 Watchlist confirm: {} | {}/{} | score={}",
                            signal.symbol, entry.confirmations,
                            settings.effective_watchlist_confirmations, signal.score,
                        )

                        if entry.confirmations >= settings.effective_watchlist_confirmations:
                            # ── Stage 3: Deep research ─────────────────────
                            logger.info(
                                "🔬 Research start: {} | score={}", signal.symbol, signal.score
                            )
                            result = await self._research.research(
                                signal.symbol, signal.direction, signal.score
                            )
                            research_id = await self._coin_db.record_research(
                                symbol=signal.symbol,
                                direction=signal.direction,
                                initial_score=signal.score,
                                research_score=result.score,
                                mtf_alignment=result.mtf_alignment,
                                passed=result.passed,
                            )

                            # Set cooldown and clear watchlist slot
                            self._signal_cooldowns[signal.symbol] = now
                            del self._watchlist[signal.symbol]

                            # Notify research result
                            await self._alerts.send_research_result(signal, result)

                            if result.passed:
                                logger.info(
                                    "✅ Research PASSED: {} | score={:.1f} | mtf={:.0f}%",
                                    signal.symbol, result.score, result.mtf_alignment * 100,
                                )
                                await self._try_execute_trade(
                                    signal,
                                    research_id=research_id,
                                    research_score=result.score,
                                )
                            else:
                                logger.info(
                                    "❌ Research FAILED: {} | score={:.1f} | mtf={:.0f}% — skip",
                                    signal.symbol, result.score, result.mtf_alignment * 100,
                                )

                # Expire watchlist entries whose signal dropped below threshold
                stale = [s for s in list(self._watchlist) if s not in live_qualifying]
                for sym in stale:
                    del self._watchlist[sym]
                    logger.debug("Watchlist expired (signal lost): {}", sym)

                logger.info(
                    "── Scan done | {} | watchlist={} | daily_trades_left={} | next in {}s ──",
                    "PAUSED" if self._bot_paused else "RUNNING",
                    len(self._watchlist),
                    self._risk.daily_trades_remaining,
                    settings.scan_interval_seconds,
                )
                await asyncio.sleep(settings.scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Scan loop error: {}", exc)
                await asyncio.sleep(10)

    # ── Trade execution ────────────────────────────────────────────────────
    async def _try_execute_trade(
        self,
        signal,
        research_id: Optional[int] = None,
        research_score: float = 0.0,
    ) -> None:
        # Check open trade + daily limits
        await self._sync_open_trades()
        if not self._risk.can_open_trade():
            return

        # Auto Strategy Discovery — evaluate all strategies and pick the best fit
        setup = None
        strategy_name = None
        try:
            df = await self._data_engine.get_klines_binance(
                signal.symbol, "15m", 200
            )
            setup, strategy_name = await self._strategy_registry.select_best(
                signal.symbol, df, signal.direction
            )
        except Exception as exc:
            logger.debug("Strategy eval error {}: {}", signal.symbol, exc)

        if not setup:
            # Fallback: percentage-based SL/TP when no strategy matches direction
            price = signal.price
            if not price or price <= 0:
                logger.debug("{} — no strategy and no valid price, skip", signal.symbol)
                return
            sl_pct = min(1.5, settings.max_stop_loss_pct * 0.6)
            tp_pct = sl_pct * settings.min_risk_reward_ratio
            if signal.direction == "LONG":
                sl = round(price * (1 - sl_pct / 100), 8)
                tp = round(price * (1 + tp_pct / 100), 8)
            else:
                sl = round(price * (1 + sl_pct / 100), 8)
                tp = round(price * (1 - tp_pct / 100), 8)
            setup = TradeSetup(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                leverage=settings.max_leverage,
                risk_reward=settings.min_risk_reward_ratio,
            )
            strategy_name = "FALLBACK"
            logger.info("{} — FALLBACK setup: {} sl={:.6f} tp={:.6f}", signal.symbol, signal.direction, sl, tp)

        # Re-fetch available balance from exchange right before sizing.
        # Prevents stale balance issues when multiple trades open within the same
        # scan cycle (each trade sees the true remaining free margin).
        if not self.dry_run:
            try:
                _bal_exec = self._bybit_exec if self.exchange == "bybit" else self._binance_exec
                _live_bal = await _bal_exec.get_account_balance()
                if _live_bal > 0:
                    self._risk.update_balance(_live_bal)
                    self._risk.reset_margin()  # exchange balance already excludes committed margin
            except Exception as _bal_exc:
                logger.debug("Pre-trade balance refresh error: {}", _bal_exc)

        # Risk validation & position sizing
        risk_params = self._risk.calculate_position(
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
        )
        if not risk_params:
            return

        # ── Entry price deviation guard ────────────────────────────────────
        # Abort if the live price has drifted too far from the strategy price
        # since the signal was confirmed (prevents chasing after a big spike).
        if settings.entry_max_deviation_pct > 0:
            live_ticker = self._cache.tickers.get(risk_params.symbol, {})
            live_price = float(live_ticker.get("c", 0))
            if live_price > 0:
                deviation_pct = abs(live_price - risk_params.entry_price) / risk_params.entry_price * 100
                if deviation_pct > settings.entry_max_deviation_pct:
                    logger.info(
                        "{} entry aborted — price drifted {:.2f}% from signal (max {:.1f}%)",
                        risk_params.symbol, deviation_pct, settings.entry_max_deviation_pct,
                    )
                    await self._alerts.send_text(
                        f"⚠️ *{risk_params.symbol}* entry cancelled\n"
                        f"Price drifted {deviation_pct:.2f}% from signal price\n"
                        f"Signal: {risk_params.entry_price:.4f} | Live: {live_price:.4f}"
                    )
                    return

        # ── LIMIT order path (live + dry-run simulation) ────────────────────
        # Live:    places a real GTC limit on exchange; fill confirmed in monitor loop.
        # Dry-run: simulates limit by watching if live price reaches limit_price
        #          within the timeout window, then executes the paper trade.
        if settings.entry_order_type == "LIMIT":
            live_ticker = self._cache.tickers.get(risk_params.symbol, {})
            live_price = float(live_ticker.get("c", 0)) or risk_params.entry_price
            offset = settings.limit_entry_offset_pct / 100
            if risk_params.direction == "LONG":
                limit_price = round(live_price * (1 - offset), 8)
            else:
                limit_price = round(live_price * (1 + offset), 8)

            if self.dry_run:
                # Dry-run: no exchange call — register simulated pending entry
                order_id = f"DRY-{int(time.time())}"
            else:
                executor = self._bybit_exec if self.exchange == "bybit" else self._binance_exec
                try:
                    if self.exchange == "bybit":
                        result = await executor.place_limit_order(
                            risk_params.symbol,
                            "Buy" if risk_params.direction == "LONG" else "Sell",
                            risk_params.quantity, limit_price,
                        )
                        order_id = result.get("result", {}).get("orderId")
                    else:
                        result = await executor.place_limit_order(
                            risk_params.symbol,
                            "BUY" if risk_params.direction == "LONG" else "SELL",
                            risk_params.quantity, limit_price,
                        )
                        order_id = result.get("orderId")
                except Exception as exc:
                    logger.error("Limit order placement failed {}: {}", risk_params.symbol, exc)
                    return

                if not order_id:
                    logger.error("Limit order rejected — no order_id returned for {}", risk_params.symbol)
                    return

            self._pending_entry_orders[risk_params.symbol] = PendingEntry(
                symbol=risk_params.symbol,
                order_id=str(order_id),
                placed_at=time.time(),
                risk_params=risk_params,
                signal=signal,
                retry_count=0,
                research_id=research_id,
                research_score=research_score,
                strategy_name=strategy_name,
                limit_price=limit_price,
            )
            # Reserve margin immediately so concurrent signals don't size against
            # the same free balance. Released when this order fills (teardown) or
            # is finally abandoned after max retries.
            self._risk.add_margin(risk_params.position_size_usdt / max(risk_params.leverage, 1))
            dry_tag = " [SIM]" if self.dry_run else ""
            logger.info(
                "⏳ LIMIT entry placed{}: {} {} @ {:.6f} | order_id={} | timeout={}s",
                dry_tag, risk_params.symbol, risk_params.direction, limit_price,
                order_id, settings.limit_order_timeout_seconds,
            )
            await self._alerts.send_text(
                f"⏳ *{risk_params.symbol}* LIMIT order{' (sim)' if self.dry_run else ''} placed\n"
                f"Direction: {risk_params.direction}\n"
                f"Limit price: `{limit_price:.6f}`\n"
                f"SL: `{risk_params.stop_loss:.6f}` | TP: `{risk_params.take_profit:.6f}`\n"
                f"Cancels in {settings.limit_order_timeout_seconds}s if not filled"
            )
            return   # position opened when order fills (or sim-fills)

        # ── MARKET order path (or dry-run) ─────────────────────────────────
        # Notify entry intent now — market orders fill immediately.
        await self._alerts.send_trade_entry(
            symbol=risk_params.symbol,
            direction=risk_params.direction,
            entry=risk_params.entry_price,
            sl=risk_params.stop_loss,
            tp=risk_params.take_profit,
            rr=risk_params.risk_reward_ratio,
            signal_score=signal.score,
            research_score=research_score,
            daily_remaining=max(0, self._risk.daily_trades_remaining - 1),
        )
        self._risk.record_trade_opened()
        trade_id = None

        if self.dry_run:
            logger.info("DRY RUN — would open: {} {}", setup.symbol, setup.direction)
            trade_id = await self._portfolio.open_position({
                "symbol":           risk_params.symbol,
                "direction":        risk_params.direction,
                "entry_price":      risk_params.entry_price,
                "stop_loss":        risk_params.stop_loss,
                "take_profit":      risk_params.take_profit,
                "quantity":         risk_params.quantity,
                "leverage":         risk_params.leverage,
                "position_size_usdt": risk_params.position_size_usdt,
                "exchange":         "dry_run",
                "signal_score":     signal.score,
            })
            if trade_id:
                meta = self._make_position_meta(risk_params.entry_price)
                meta["strategy_name"] = strategy_name
                meta["margin_used"] = risk_params.position_size_usdt / max(risk_params.leverage, 1)
                self._position_meta[risk_params.symbol] = meta
        else:
            # Live execution
            if self.exchange == "bybit":
                result = await self._bybit_exec.execute_trade(risk_params)
            else:
                result = await self._binance_exec.execute_trade(risk_params)

            if result.get("status") == "open":
                trade_id = await self._portfolio.open_position({
                    "symbol":           risk_params.symbol,
                    "direction":        risk_params.direction,
                    "entry_price":      risk_params.entry_price,
                    "stop_loss":        risk_params.stop_loss,
                    "take_profit":      risk_params.take_profit,
                    "quantity":         risk_params.quantity,
                    "leverage":         risk_params.leverage,
                    "position_size_usdt": risk_params.position_size_usdt,
                    "exchange":         self.exchange,
                    "order_id":         str(result.get("order_id", "")),
                    "signal_score":     signal.score,
                })
                if trade_id:
                    meta = self._make_position_meta(
                        risk_params.entry_price,
                        sl_order_id=result.get("sl_order_id"),
                        tp_order_id=result.get("tp_order_id"),
                    )
                    meta["strategy_name"] = strategy_name
                    meta["margin_used"] = risk_params.position_size_usdt / max(risk_params.leverage, 1)
                    self._position_meta[risk_params.symbol] = meta

        # Record ML feature vector only when a real trade is opened (not on every signal)
        if trade_id:
            df_cached = self._klines_cache.get(risk_params.symbol)
            if df_cached is not None and len(df_cached) >= 50:
                ml_key = f"{risk_params.symbol}_{trade_id}"
                self._learner.record_prediction(ml_key, risk_params.symbol, df_cached)
                self._pending_labels[risk_params.symbol] = ml_key

        if trade_id and research_id:
            # Link research decision to the actual trade for outcome tracking
            await self._coin_db.mark_research_executed(research_id, trade_id)
            self._pending_research_ids[risk_params.symbol] = research_id

        if trade_id:
            # Reserve margin in risk manager (prevents concurrent trades from
            # over-sizing against the same balance within the same scan cycle).
            self._risk.add_margin(risk_params.position_size_usdt / max(risk_params.leverage, 1))
            self._risk.update_open_trades(
                len(await self._portfolio.get_open_positions())
            )

    # ── Coin ranking loop ──────────────────────────────────────────────────
    async def _coin_ranking_loop(self) -> None:
        while self._running:
            try:
                if self._klines_cache:
                    # Update BTC baseline for correlation penalty
                    try:
                        btc_df = await self._data_engine.get_klines_binance(
                            "BTCUSDT", "1h", 100
                        )
                        self._ranker.update_btc(btc_df)
                    except Exception:
                        pass

                    rankings = await self._ranker.rank(
                        self._klines_cache,
                        ml_predictions=dict(self._ml_predictions_cache),
                    )
                    state.coin_rankings = [
                        {
                            "symbol": r.symbol,
                            "composite_score": r.composite_score,
                            "momentum_score": r.momentum_score,
                            "volume_score": r.volume_score,
                            "trend_score": r.trend_score,
                            "win_rate": r.win_rate,
                            "ml_confidence": r.ml_confidence,
                            "grade": r.grade,
                        }
                        for r in rankings
                    ]
                    logger.info(
                        "CoinRanker: ranked {} coins, top={} grade={}",
                        len(rankings),
                        rankings[0].symbol if rankings else "-",
                        rankings[0].grade if rankings else "-",
                    )
                    await broadcast("rankings_update", {"count": len(rankings)})
            except Exception as exc:
                logger.debug("Coin ranking loop error: {}", exc)
            await asyncio.sleep(300)    # re-rank every 5 minutes

    # ── Arbitrage loop ─────────────────────────────────────────────────────
    async def _arbitrage_loop(self, symbols) -> None:
        while self._running:
            try:
                opps = await self._arb.scan(symbols[:100])
                if opps:
                    top = opps[0]
                    if top.spread_pct >= 0.3:
                        await self._alerts.send_text(
                            f"⚡ ARB OPPORTUNITY\n"
                            f"{top.symbol}: "
                            f"Binance={top.binance_price} "
                            f"Bybit={top.bybit_price} "
                            f"Spread={top.spread_pct:.3f}%"
                        )
            except Exception as exc:
                logger.debug("Arb loop error: {}", exc)
            await asyncio.sleep(300)      # check every 5 min

    # ── Portfolio sync ─────────────────────────────────────────────────────
    async def _portfolio_sync_loop(self) -> None:
        while self._running:
            try:
                # Fetch actual balance from exchange (not config default)
                if not self.dry_run:
                    try:
                        executor = self._bybit_exec if self.exchange == "bybit" else self._binance_exec
                        exchange_balance = await executor.get_account_balance()
                        if exchange_balance > 0:
                            await self._portfolio.update_balance(exchange_balance)
                            self._risk.update_balance(exchange_balance)
                    except Exception as exc:
                        logger.debug("Balance fetch error: {}", exc)

                positions = await self._portfolio.get_open_positions()
                current_open = set(positions.keys())
                prev_open = set(self._prev_open_positions.keys())

                # Detect newly closed positions and label for self-learning
                closed_symbols = prev_open - current_open
                for sym in closed_symbols:
                    trade_id = self._pending_labels.pop(sym, None)
                    if trade_id:
                        # Fallback: teardown via _position_monitor_loop already consumed
                        # the label normally. If somehow it was missed, label with 0.0
                        # (in-memory position snapshot does not track pnl_usdt).
                        prev_data = self._prev_open_positions.get(sym, {})
                        pnl = prev_data.get("pnl_usdt", 0.0)
                        self._learner.label_trade(trade_id, pnl)

                # Save current positions as snapshot for next cycle
                self._prev_open_positions = dict(positions)

                metrics = await self._portfolio.calculate_metrics()
                balance = metrics.get("balance", settings.account_balance_usdt)
                self._risk.update_balance(balance)
                await broadcast("metrics", metrics)
            except Exception as exc:
                logger.debug("Portfolio sync error: {}", exc)
            await asyncio.sleep(30)

    async def _sync_open_trades(self) -> None:
        positions = await self._portfolio.get_open_positions()
        self._risk.update_open_trades(len(positions))

    # ── Position monitor ───────────────────────────────────────────────────
    async def _position_monitor_loop(self) -> None:
        """Poll exchange every 30s to detect SL/TP fills and run smart position management."""
        while self._running:
            try:
                if self.dry_run:
                    await self._check_dryrun_positions()
                else:
                    executor = self._bybit_exec if self.exchange == "bybit" else self._binance_exec
                    exchange_positions = await executor.get_open_positions()
                    exchange_symbols = {
                        p.get("symbol") for p in exchange_positions
                    }
                    portfolio_positions = await self._portfolio.get_open_positions()

                    for symbol, pos_data in list(portfolio_positions.items()):
                        if symbol not in exchange_symbols:
                            # Position no longer on exchange — SL or TP was triggered
                            pnl = 0.0
                            if hasattr(executor, "get_recent_pnl"):
                                try:
                                    pnl = await executor.get_recent_pnl(symbol)
                                except Exception:
                                    pass

                            await self._teardown_closed_position(symbol, 0.0, pnl)
                            outcome = "✅ TP hit" if pnl > 0 else "🛑 SL hit"
                            await self._alerts.send_text(
                                f"📊 Position closed: {symbol}\n"
                                f"{outcome} | PnL: {pnl:+.2f} USDT"
                            )
                            logger.info("Position monitor: closed {} PnL={:.2f}", symbol, pnl)
                        else:
                            # Position still open — run smart exit checks
                            ticker = self._cache.tickers.get(symbol, {})
                            current_price = float(ticker.get("c", 0))
                            if current_price > 0:
                                await self._manage_open_position(
                                    symbol, pos_data, executor, current_price
                                )
            except Exception as exc:
                logger.debug("Position monitor error: {}", exc)
            await asyncio.sleep(30)

    async def _check_dryrun_positions(self) -> None:
        """Dry-run: simulate position closing when current price hits SL or TP,
        then run smart exit checks on still-open positions."""
        portfolio_positions = await self._portfolio.get_open_positions()
        for symbol, pos_data in list(portfolio_positions.items()):
            try:
                ticker = self._cache.tickers.get(symbol, {})
                current_price = float(ticker.get("c", 0))

                # WS ticker cache miss — fall back to klines last close, then REST
                if current_price <= 0:
                    df = self._klines_cache.get(symbol)
                    if df is not None and len(df) > 0:
                        current_price = float(df["close"].iloc[-1])
                if current_price <= 0:
                    try:
                        kl = await self._data_engine.get_klines_binance(symbol, "1m", 2)
                        current_price = float(kl["close"].iloc[-1])
                        self._klines_cache[symbol] = kl   # prime cache for next cycle
                    except Exception:
                        pass
                if current_price <= 0:
                    continue

                direction = pos_data.get("direction", "LONG")
                entry = float(pos_data.get("entry_price", 0))
                sl = float(pos_data.get("stop_loss", 0))
                tp = float(pos_data.get("take_profit", 0))
                qty = float(pos_data.get("quantity", 0))

                hit_price = None
                if direction == "LONG":
                    if current_price <= sl:
                        hit_price = sl
                    elif current_price >= tp:
                        hit_price = tp
                else:
                    if current_price >= sl:
                        hit_price = sl
                    elif current_price <= tp:
                        hit_price = tp

                if hit_price:
                    pnl = (hit_price - entry) * qty if direction == "LONG" else (entry - hit_price) * qty
                    await self._teardown_closed_position(symbol, hit_price, pnl)
                    outcome = "✅ TP hit" if pnl > 0 else "🛑 SL hit"
                    logger.info("[DRY RUN] {} {} PnL={:.2f} USDT", symbol, outcome, pnl)
                else:
                    # Still open — apply smart management (no API calls in dry-run)
                    await self._manage_open_position(symbol, pos_data, None, current_price)
            except Exception as exc:
                logger.debug("Dry run position check error {}: {}", symbol, exc)

    # ── LIMIT order monitor ────────────────────────────────────────────────
    async def _limit_order_monitor_loop(self) -> None:
        """
        Poll pending LIMIT entry orders every 10 s.
        • Filled  → place SL/TP, open portfolio position, send entry alert.
        • Timeout → cancel, retry at refreshed price up to limit_order_max_retries.
        • Max retries exceeded → abandon and notify.
        No-op in dry-run mode or when entry_order_type == MARKET.
        """
        while self._running:
            try:
                if not self._pending_entry_orders:
                    await asyncio.sleep(10)
                    continue

                executor = self._bybit_exec if self.exchange == "bybit" else self._binance_exec
                now = time.time()

                for symbol in list(self._pending_entry_orders.keys()):
                    pending = self._pending_entry_orders.get(symbol)
                    if not pending:
                        continue

                    elapsed = now - pending.placed_at

                    # ── Query fill status from exchange (or simulate for dry-run) ──
                    try:
                        if self.dry_run:
                            # Simulate fill: check if live price has reached limit_price
                            live_ticker = self._cache.tickers.get(symbol, {})
                            live_price = float(live_ticker.get("c", 0))
                            if live_price > 0 and pending.limit_price > 0:
                                if pending.risk_params.direction == "LONG":
                                    filled = live_price <= pending.limit_price
                                else:
                                    filled = live_price >= pending.limit_price
                            else:
                                filled = False
                        elif self.exchange == "bybit":
                            status_data = await executor.get_order_status(
                                symbol, pending.order_id
                            )
                            order_info = status_data.get("result", {}).get("list", [{}])[0]
                            filled = order_info.get("orderStatus", "") == "Filled"
                        else:
                            status_data = await executor.get_order_status(
                                symbol, int(pending.order_id)
                            )
                            filled = status_data.get("status", "") == "FILLED"
                    except Exception as exc:
                        logger.debug("Limit order status error {}: {}", symbol, exc)
                        continue

                    if filled:
                        # ── Fill confirmed: place SL/TP + open position ────
                        del self._pending_entry_orders[symbol]
                        logger.info("✅ LIMIT order filled{}: {} — placing SL/TP",
                                    " [SIM]" if self.dry_run else "", symbol)
                        rp = pending.risk_params
                        sl_order_id = tp_order_id = None
                        if not self.dry_run:
                            try:
                                if self.exchange == "bybit":
                                    await executor.set_position_tp_sl(
                                        symbol, rp.stop_loss, rp.take_profit
                                    )
                                else:
                                    side = "BUY" if rp.direction == "LONG" else "SELL"
                                    sl_r = await executor.place_stop_loss(
                                        symbol, side, rp.quantity, rp.stop_loss
                                    )
                                    tp_r = await executor.place_take_profit(
                                        symbol, side, rp.quantity, rp.take_profit
                                    )
                                    sl_order_id = sl_r.get("orderId")
                                    tp_order_id = tp_r.get("orderId")
                            except Exception as exc:
                                logger.error("SL/TP placement after LIMIT fill {}: {}", symbol, exc)

                        self._risk.record_trade_opened()
                        await self._alerts.send_trade_entry(
                            symbol=rp.symbol,
                            direction=rp.direction,
                            entry=rp.entry_price,
                            sl=rp.stop_loss,
                            tp=rp.take_profit,
                            rr=rp.risk_reward_ratio,
                            signal_score=pending.signal.score,
                            research_score=pending.research_score,
                            daily_remaining=max(0, self._risk.daily_trades_remaining - 1),
                        )
                        trade_id = await self._portfolio.open_position({
                            "symbol":             rp.symbol,
                            "direction":          rp.direction,
                            "entry_price":        pending.limit_price if pending.limit_price > 0 else rp.entry_price,
                            "stop_loss":          rp.stop_loss,
                            "take_profit":        rp.take_profit,
                            "quantity":           rp.quantity,
                            "leverage":           rp.leverage,
                            "position_size_usdt": rp.position_size_usdt,
                            "exchange":           "dry_run" if self.dry_run else self.exchange,
                            "order_id":           pending.order_id,
                            "signal_score":       pending.signal.score,
                        })
                        if trade_id:
                            meta = self._make_position_meta(
                                rp.entry_price,
                                sl_order_id=sl_order_id,
                                tp_order_id=tp_order_id,
                            )
                            meta["strategy_name"] = pending.strategy_name
                            # margin_used stored so teardown can release it correctly
                            # (margin was already reserved when the LIMIT was placed)
                            meta["margin_used"] = rp.position_size_usdt / max(rp.leverage, 1)
                            self._position_meta[symbol] = meta
                            # Record ML feature vector on actual fill (not on signal)
                            df_cached = self._klines_cache.get(symbol)
                            if df_cached is not None and len(df_cached) >= 50:
                                ml_key = f"{symbol}_{trade_id}"
                                self._learner.record_prediction(ml_key, symbol, df_cached)
                                self._pending_labels[symbol] = ml_key
                            if pending.research_id:
                                await self._coin_db.mark_research_executed(
                                    pending.research_id, trade_id
                                )
                                self._pending_research_ids[symbol] = pending.research_id
                            self._risk.update_open_trades(
                                len(await self._portfolio.get_open_positions())
                            )
                        continue

                    # ── Timeout check ──────────────────────────────────────
                    if elapsed >= settings.limit_order_timeout_seconds:
                        logger.info(
                            "⏱ LIMIT order timed out ({:.0f}s): {} — cancelling",
                            elapsed, symbol,
                        )
                        try:
                            await executor.cancel_order(symbol, pending.order_id)
                        except Exception as exc:
                            logger.debug("Cancel timed-out order {}: {}", symbol, exc)

                        del self._pending_entry_orders[symbol]

                        if pending.retry_count < settings.limit_order_max_retries:
                            await self._retry_limit_entry(pending)
                            # margin stays reserved across retries (same pending order)
                        else:
                            # Final abandon — release the margin reserved at placement
                            self._risk.release_margin(
                                pending.risk_params.position_size_usdt / max(pending.risk_params.leverage, 1)
                            )
                            logger.info(
                                "🚫 LIMIT entry abandoned after {} retries: {}",
                                settings.limit_order_max_retries, symbol,
                            )
                            await self._alerts.send_text(
                                f"⚠️ *{symbol}* limit entry abandoned\n"
                                f"No fill after {settings.limit_order_max_retries + 1} "
                                f"attempt(s). Bot will re-scan for a new setup."
                            )

            except Exception as exc:
                logger.debug("Limit order monitor error: {}", exc)
            await asyncio.sleep(10)

    async def _retry_limit_entry(self, pending: PendingEntry) -> None:
        """Re-place a LIMIT entry with a refreshed market price after a timeout."""
        await self._sync_open_trades()
        if not self._risk.can_open_trade():
            logger.info("Limit retry skipped — trade limits reached: {}", pending.symbol)
            return

        live_ticker = self._cache.tickers.get(pending.symbol, {})
        live_price = float(live_ticker.get("c", 0))
        if live_price <= 0:
            logger.debug("Limit retry skipped — no live price for {}", pending.symbol)
            return

        # Allow 3× the normal deviation tolerance for retries (price moves are expected)
        if settings.entry_max_deviation_pct > 0:
            dev_pct = (
                abs(live_price - pending.risk_params.entry_price)
                / pending.risk_params.entry_price * 100
            )
            if dev_pct > settings.entry_max_deviation_pct * 3:
                logger.info(
                    "{} limit retry aborted — price drifted {:.2f}% from signal",
                    pending.symbol, dev_pct,
                )
                await self._alerts.send_text(
                    f"⚠️ *{pending.symbol}* limit retry cancelled\n"
                    f"Price drifted {dev_pct:.2f}% from signal price"
                )
                return

        offset = settings.limit_entry_offset_pct / 100
        rp = pending.risk_params
        if rp.direction == "LONG":
            limit_price = round(live_price * (1 - offset), 8)
        else:
            limit_price = round(live_price * (1 + offset), 8)

        executor = self._bybit_exec if self.exchange == "bybit" else self._binance_exec
        if self.dry_run:
            order_id = f"DRY-{int(time.time())}"
        else:
            try:
                if self.exchange == "bybit":
                    result = await executor.place_limit_order(
                        pending.symbol,
                        "Buy" if rp.direction == "LONG" else "Sell",
                        rp.quantity, limit_price,
                    )
                    order_id = result.get("result", {}).get("orderId")
                else:
                    result = await executor.place_limit_order(
                        pending.symbol,
                        "BUY" if rp.direction == "LONG" else "SELL",
                        rp.quantity, limit_price,
                    )
                    order_id = result.get("orderId")
            except Exception as exc:
                logger.error("Limit retry placement failed {}: {}", pending.symbol, exc)
                return

            if not order_id:
                logger.error("Limit retry rejected — no order_id for {}", pending.symbol)
                return

        attempt = pending.retry_count + 1
        self._pending_entry_orders[pending.symbol] = PendingEntry(
            symbol=pending.symbol,
            order_id=str(order_id),
            placed_at=time.time(),
            risk_params=rp,
            signal=pending.signal,
            retry_count=attempt,
            research_id=pending.research_id,
            research_score=pending.research_score,
            strategy_name=pending.strategy_name,
            limit_price=limit_price,
        )
        logger.info(
            "🔄 LIMIT retry {}/{}: {} @ {:.6f}",
            attempt, settings.limit_order_max_retries, pending.symbol, limit_price,
        )
        await self._alerts.send_text(
            f"🔄 *{pending.symbol}* limit order re-placed\n"
            f"Attempt {attempt}/{settings.limit_order_max_retries}\n"
            f"New price: `{limit_price:.6f}` ({rp.direction})\n"
            f"Cancels in {settings.limit_order_timeout_seconds}s if not filled"
        )

    # ── Smart position management ──────────────────────────────────────────

    def _make_position_meta(
        self,
        entry_price: float,
        sl_order_id=None,
        tp_order_id=None,
    ) -> dict:
        """Build the in-memory metadata dict for a freshly opened position."""
        return {
            "opened_at":      time.time(),
            "peak_price":     entry_price,
            "sl_order_id":    sl_order_id,
            "tp_order_id":    tp_order_id,
            "breakeven_done": False,
        }

    async def _teardown_closed_position(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        broadcast_extra: Optional[dict] = None,
    ) -> None:
        """
        Shared teardown for every position close (SL/TP fill, force-close, dry-run hit).
        Persists the closed trade, cleans up in-memory state, labels the ML trade,
        updates research outcome, syncs risk counter, and broadcasts the WS event.
        Callers are responsible for any Telegram alert and log message.
        """
        await self._portfolio.close_position(symbol, exit_price, pnl)
        # Release allocated margin BEFORE clearing meta so the risk manager's
        # free-balance is accurate for the next position sizing calculation.
        margin_used = self._position_meta.get(symbol, {}).get("margin_used", 0.0)
        if margin_used > 0:
            self._risk.release_margin(margin_used)
        strategy_name = self._position_meta.get(symbol, {}).get("strategy_name")
        self._position_meta.pop(symbol, None)
        self._risk.record_trade_closed(pnl)   # update daily PnL for loss-limit guard

        trade_id = self._pending_labels.pop(symbol, None)
        if trade_id:
            self._learner.label_trade(trade_id, pnl)

        research_id = self._pending_research_ids.pop(symbol, None)
        if research_id:
            await self._coin_db.update_research_outcome(research_id, pnl)

        # Persist strategy outcome for Auto Strategy Discovery learning
        if strategy_name and self._strategy_registry:
            await self._strategy_registry.record_outcome(strategy_name, pnl)

        self._risk.update_open_trades(len(await self._portfolio.get_open_positions()))

        payload = {"symbol": symbol, "pnl": pnl}
        if broadcast_extra:
            payload.update(broadcast_extra)
        await broadcast("position_closed", payload)

    async def _manage_open_position(
        self,
        symbol: str,
        pos_data: dict,
        executor,              # None in dry-run
        current_price: float,
    ) -> None:
        """
        Called every monitor cycle for each still-open position.
        Applies (in priority order):
          1. Time-based force-exit
          2. Reversal / stale-loss exit
          3. Breakeven stop (SL → entry after 50 % of TP reached)
          4. Trailing stop (slides SL behind the running peak)
        """
        meta = self._position_meta.get(symbol)
        if not meta:
            # Auto-recover after restart: rebuild meta from pos_data so smart-exit
            # rules (trailing, breakeven, time-limit, reversal) resume immediately.
            # opened_at from DB preserves the real hold-time; falls back to now.
            meta = self._make_position_meta(float(pos_data.get("entry_price", 0)))
            if pos_data.get("opened_at"):
                meta["opened_at"] = pos_data["opened_at"]
            self._position_meta[symbol] = meta
            logger.info("Recovered position meta for {} after restart (opened_at={})",
                        symbol, datetime.utcfromtimestamp(meta["opened_at"]).strftime("%H:%M"))

        direction  = pos_data.get("direction", "LONG")
        entry      = float(pos_data.get("entry_price", 0))
        sl         = float(pos_data.get("stop_loss", 0))
        tp         = float(pos_data.get("take_profit", 0))
        quantity   = float(pos_data.get("quantity", 0))

        if entry <= 0 or current_price <= 0:
            return

        # Track running peak (best price since entry in our favour)
        if direction == "LONG":
            meta["peak_price"] = max(meta.get("peak_price", entry), current_price)
            profit_pct       = (current_price - entry) / entry * 100
            counter_move_pct = max(0.0, (entry - current_price) / entry * 100)
        else:
            meta["peak_price"] = min(meta.get("peak_price", entry), current_price)
            profit_pct       = (entry - current_price) / entry * 100
            counter_move_pct = max(0.0, (current_price - entry) / entry * 100)

        hours_open = (time.time() - meta.get("opened_at", time.time())) / 3600

        # ── 1. Time-based force-exit ───────────────────────────────────────
        max_h = settings.effective_max_position_hold_hours
        if max_h > 0 and hours_open >= max_h:
            await self._force_close(
                symbol, pos_data, executor, current_price,
                f"⏱ Max hold time {max_h:.0f}h reached",
            )
            return

        # ── 2. Reversal / stale-loss exit ─────────────────────────────────
        if (
            settings.reversal_exit_enabled
            and hours_open >= settings.reversal_exit_min_hours
            and counter_move_pct >= settings.reversal_exit_pct
        ):
            await self._force_close(
                symbol, pos_data, executor, current_price,
                f"🔄 Reversal exit (−{counter_move_pct:.2f}% after {hours_open:.1f}h)",
            )
            return

        # ── 3. Breakeven stop ─────────────────────────────────────────────
        if settings.breakeven_stop_enabled and not meta.get("breakeven_done"):
            if profit_pct >= settings.breakeven_trigger_pct:
                buffer = entry * 0.001                              # +0.1 % protection
                new_sl = (entry + buffer) if direction == "LONG" else (entry - buffer)
                improves = (
                    (direction == "LONG" and new_sl > sl) or
                    (direction == "SHORT" and new_sl < sl)
                )
                if improves:
                    await self._update_sl_on_exchange(
                        symbol, direction, quantity, new_sl, meta, executor
                    )
                    pos_data["stop_loss"] = new_sl
                    sl = new_sl  # keep local var fresh for the trailing check below
                    await self._portfolio.update_position_field(symbol, "stop_loss", new_sl)
                    meta["breakeven_done"] = True
                    logger.info("{} breakeven SL set @ {:.6f}", symbol, new_sl)
                    await self._alerts.send_text(
                        f"🟡 *{symbol}* | SL → Breakeven\n"
                        f"SL moved to {new_sl:.4f} (profit: {profit_pct:.2f}%)"
                    )

        # ── 4. Trailing stop ──────────────────────────────────────────────
        if (
            settings.trailing_stop_enabled
            and profit_pct >= settings.trailing_stop_activation_pct
        ):
            peak       = meta.get("peak_price", entry)
            trail_dist = settings.trailing_stop_distance_pct / 100
            min_move   = settings.trailing_stop_min_move_pct / 100

            if direction == "LONG":
                trail_sl = round(peak * (1 - trail_dist), 8)
                if sl > 0 and trail_sl > sl * (1 + min_move):
                    await self._update_sl_on_exchange(
                        symbol, direction, quantity, trail_sl, meta, executor
                    )
                    pos_data["stop_loss"] = trail_sl
                    await self._portfolio.update_position_field(symbol, "stop_loss", trail_sl)
                    logger.info(
                        "{} trailing SL: {:.6f} → {:.6f} (peak {:.6f})",
                        symbol, sl, trail_sl, peak,
                    )
            else:
                trail_sl = round(peak * (1 + trail_dist), 8)
                if sl > 0 and trail_sl < sl * (1 - min_move):
                    await self._update_sl_on_exchange(
                        symbol, direction, quantity, trail_sl, meta, executor
                    )
                    pos_data["stop_loss"] = trail_sl
                    await self._portfolio.update_position_field(symbol, "stop_loss", trail_sl)
                    logger.info(
                        "{} trailing SL: {:.6f} → {:.6f} (peak {:.6f})",
                        symbol, sl, trail_sl, peak,
                    )

    async def _update_sl_on_exchange(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        new_sl: float,
        meta: dict,
        executor,
    ) -> None:
        """Place an updated SL order on the exchange (no-op in dry-run)."""
        if executor is None:
            return   # dry-run: only in-memory tracking needed
        try:
            result = await executor.update_stop_loss(
                symbol, direction, quantity, new_sl,
                old_order_id=meta.get("sl_order_id"),
            )
            new_id = result.get("orderId") or result.get("result", {}).get("orderId")
            if new_id:
                meta["sl_order_id"] = new_id
        except Exception as exc:
            logger.debug("_update_sl_on_exchange {}: {}", symbol, exc)

    async def _force_close(
        self,
        symbol: str,
        pos_data: dict,
        executor,
        current_price: float,
        reason: str,
    ) -> None:
        """
        Immediately close a position at market price.
        Works for both live (calls executor) and dry-run (calculates PnL locally).
        """
        direction = pos_data.get("direction", "LONG")
        quantity  = float(pos_data.get("quantity", 0))
        entry     = float(pos_data.get("entry_price", current_price))
        meta      = self._position_meta.get(symbol, {})
        pnl       = 0.0

        if executor is not None:
            try:
                await executor.close_position_market(symbol, direction, quantity)
            except Exception as exc:
                logger.error("Force-close market order failed {}: {}", symbol, exc)
                return

            for key in ("sl_order_id", "tp_order_id"):
                oid = meta.get(key)
                if oid:
                    try:
                        await executor.cancel_order(symbol, oid)
                    except Exception:
                        pass

            if hasattr(executor, "get_recent_pnl"):
                try:
                    pnl = await executor.get_recent_pnl(symbol)
                except Exception:
                    pass
        else:
            # Dry-run: calculate PnL at current price
            pnl = (
                (current_price - entry) * quantity
                if direction == "LONG"
                else (entry - current_price) * quantity
            )

        await self._teardown_closed_position(symbol, current_price, pnl, {"reason": reason})
        await self._alerts.send_text(
            f"🚫 *Force Close: {symbol}*\n"
            f"Reason: {reason}\n"
            f"Exit: {current_price:.4f} | PnL: {pnl:+.2f} USDT"
        )
        logger.info("Force-closed {} | {} | PnL={:.2f}", symbol, reason, pnl)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto AI Trading System")
    parser.add_argument(
        "--exchange",
        choices=["binance", "bybit"],
        default="binance",
        help="Primary exchange for trade execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Paper trading mode — no real orders",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Start only the web dashboard",
    )
    return parser.parse_args()


async def run_trading(args: argparse.Namespace) -> None:
    system = TradingSystem(exchange=args.exchange, dry_run=args.dry_run)

    loop = asyncio.get_running_loop()

    def _shutdown(*_):
        logger.info("Shutdown signal received …")
        loop.create_task(system.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass   # Windows doesn't support add_signal_handler for all signals

    await system.start()


async def run_all(args: argparse.Namespace) -> None:
    """Run trading system + uvicorn dashboard concurrently.
    Dashboard keeps running even if the trading system encounters an error.
    """
    config = uvicorn.Config(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    if args.dashboard_only:
        await server.serve()
    else:
        async def safe_trading():
            try:
                await run_trading(args)
            except Exception as exc:
                logger.error(
                    "Trading system exited with error: {} — dashboard remains running",
                    exc,
                )

        await asyncio.gather(safe_trading(), server.serve())


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
