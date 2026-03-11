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

from strategy.trend_strategy import TrendStrategy
from strategy.breakout_strategy import BreakoutStrategy
from strategy.liquidity_strategy import LiquidityStrategy

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
        self._updater = AutoUpdater()

        # Self-learning tracking
        self._pending_labels: Dict[str, str] = {}     # symbol -> trade_id
        self._prev_open_positions: Dict[str, dict] = {}  # symbol -> position snapshot
        self._klines_cache: Dict[str, object] = {}   # symbol -> DataFrame for ranking
        self._ml_predictions_cache: Dict[str, float] = {}  # symbol -> ml confidence
        self._signal_cooldowns: Dict[str, float] = {}      # symbol -> last decision timestamp

        # Watchlist + research phase
        self._watchlist: Dict[str, WatchlistEntry] = {}    # symbol -> entry awaiting research
        self._pending_research_ids: Dict[str, int] = {}    # symbol -> research_log id (for outcome)
        self._coin_db = CoinDatabase()
        self._research: Optional[ResearchEngine] = None    # initialised in start()

        # WS feed handle (populated in start())
        self._ws_feed = None

        # Bot on/off control (toggled via Telegram commands)
        self._bot_paused: bool = False

        # Strategy stack
        self._strategies = [
            TrendStrategy(),
            BreakoutStrategy(),
            LiquidityStrategy(),
        ]

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
        await self._alerts.start()
        self._alerts.set_command_handler(self._handle_telegram_command)
        await self._sentiment.start()
        await self._binance_exec.start()
        await self._bybit_exec.start()
        self._research = ResearchEngine(self._data_engine, self._coin_db)

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
        await self._alerts.send_text(
            f"🟢 Trading system ONLINE\n"
            f"Exchange: {self.exchange.upper()}\n"
            f"Dry run: {self.dry_run}\n"
            f"Scanning: {len(symbols)} coins"
        )

        # Run all async tasks
        await asyncio.gather(
            self._ws_feed.start(),
            self._scan_loop(),
            self._arbitrage_loop(symbols),
            self._portfolio_sync_loop(),
            self._position_monitor_loop(),
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
        """Handle /pause, /resume, /status, /help commands from Telegram."""
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

        elif cmd == "status":
            positions = await self._portfolio.get_open_positions()
            state_label = "⏸ PAUSED (no new trades)" if self._bot_paused else "▶️ RUNNING"
            await self._alerts.send_text(
                f"📊 *Bot Status*\n\n"
                f"State: `{state_label}`\n"
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
                "/help — show this message"
            )

        else:
            await self._alerts.send_text(
                f"❓ Unknown command: `/{cmd}`\n"
                "Send /help for available commands."
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

                # Re-sort: primary = signal score DESC, secondary = AI rank DESC.
                # When multiple signals tie on score, coins with higher AI grade
                # get processed first and take the limited daily trade slots.
                if state.coin_rankings:
                    _rank_map = {r["symbol"]: r["composite_score"] for r in state.coin_rankings}
                    signals.sort(
                        key=lambda s: (s.score, _rank_map.get(s.symbol, 0.0)),
                        reverse=True,
                    )

                # Track which symbols are currently qualifying (for watchlist expiry)
                live_qualifying: set = set()

                for signal in signals:
                    if not signal.is_high_probability:
                        continue

                    # Cooldown applies only when adding NEW entries to watchlist
                    in_watchlist = signal.symbol in self._watchlist
                    if not in_watchlist:
                        cooldown_secs = settings.signal_cooldown_minutes * 60
                        if now - self._signal_cooldowns.get(signal.symbol, 0) < cooldown_secs:
                            continue

                    # ML prediction boost
                    try:
                        df = await self._data_engine.get_klines_binance(
                            signal.symbol, "15m", 200
                        )
                        ml_score, confidence = self._ml.predict(df)
                        signal.score += ml_score
                        signal.ai_prediction = confidence
                        signal.confidence = confidence
                        self._ml_predictions_cache[signal.symbol] = confidence
                        trade_id = f"{signal.symbol}_{int(now)}"
                        self._learner.record_prediction(trade_id, signal.symbol, df)
                        self._pending_labels[signal.symbol] = trade_id
                        self._klines_cache[signal.symbol] = df
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
                    if ml_trained and signal.ai_prediction < settings.min_ml_confidence:
                        logger.debug(
                            "{} ML confidence {:.2f} < gate {:.2f} — skip",
                            signal.symbol, signal.ai_prediction, settings.min_ml_confidence,
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
                            signal, settings.watchlist_confirmations
                        )
                        await broadcast("signal", {
                            "symbol": signal.symbol, "score": signal.score,
                            "stage": "watchlist",
                        })
                        logger.info(
                            "📡 Watchlist add: {} | score={} | needs {} more scan(s)",
                            signal.symbol, signal.score, settings.watchlist_confirmations,
                        )
                    else:
                        # ── Stage 2: Increment confirmation count ──────────
                        entry = self._watchlist[signal.symbol]
                        entry.signal = signal        # refresh with latest scores
                        entry.confirmations += 1
                        logger.info(
                            "📡 Watchlist confirm: {} | {}/{} | score={}",
                            signal.symbol, entry.confirmations,
                            settings.watchlist_confirmations, signal.score,
                        )

                        if entry.confirmations >= settings.watchlist_confirmations:
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

        # Strategy evaluation to get precise SL/TP
        setup = None
        try:
            df = await self._data_engine.get_klines_binance(
                signal.symbol, "15m", 200
            )
            for strategy in self._strategies:
                setup = strategy.evaluate(signal.symbol, df)
                if setup:
                    break
        except Exception as exc:
            logger.debug("Strategy eval error {}: {}", signal.symbol, exc)
            return

        if not setup:
            logger.debug("{} — no strategy setup found", signal.symbol)
            return

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

        # Notify trade entry (before executing, so user sees intent)
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

        if trade_id and research_id:
            # Link research decision to the actual trade for outcome tracking
            await self._coin_db.mark_research_executed(research_id, trade_id)
            self._pending_research_ids[risk_params.symbol] = research_id

        if trade_id:
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
                        # Use PnL from the last known snapshot of the closed position
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
        """Poll exchange every 30s to detect SL/TP fills and close positions."""
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

                            await self._portfolio.close_position(symbol, 0.0, pnl)

                            trade_id = self._pending_labels.pop(symbol, None)
                            if trade_id:
                                self._learner.label_trade(trade_id, pnl)

                            # Update research outcome for self-learning
                            research_id = self._pending_research_ids.pop(symbol, None)
                            if research_id:
                                await self._coin_db.update_research_outcome(research_id, pnl)

                            self._risk.update_open_trades(
                                len(await self._portfolio.get_open_positions())
                            )
                            outcome = "✅ TP hit" if pnl > 0 else "🛑 SL hit"
                            await self._alerts.send_text(
                                f"📊 Position closed: {symbol}\n"
                                f"{outcome} | PnL: {pnl:+.2f} USDT"
                            )
                            await broadcast("position_closed", {"symbol": symbol, "pnl": pnl})
                            logger.info("Position monitor: closed {} PnL={:.2f}", symbol, pnl)
            except Exception as exc:
                logger.debug("Position monitor error: {}", exc)
            await asyncio.sleep(30)

    async def _check_dryrun_positions(self) -> None:
        """Dry-run: simulate position closing when current price hits SL or TP."""
        portfolio_positions = await self._portfolio.get_open_positions()
        for symbol, pos_data in list(portfolio_positions.items()):
            try:
                ticker = self._cache.tickers.get(symbol, {})
                current_price = float(ticker.get("c", pos_data.get("entry_price", 0)))
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
                    await self._portfolio.close_position(symbol, hit_price, pnl)
                    trade_id = self._pending_labels.pop(symbol, None)
                    if trade_id:
                        self._learner.label_trade(trade_id, pnl)
                    # Update research outcome for self-learning
                    research_id = self._pending_research_ids.pop(symbol, None)
                    if research_id:
                        await self._coin_db.update_research_outcome(research_id, pnl)
                    self._risk.update_open_trades(
                        len(await self._portfolio.get_open_positions())
                    )
                    outcome = "✅ TP hit" if pnl > 0 else "🛑 SL hit"
                    logger.info("[DRY RUN] {} {} PnL={:.2f} USDT", symbol, outcome, pnl)
                    await broadcast("position_closed", {"symbol": symbol, "pnl": pnl})
            except Exception as exc:
                logger.debug("Dry run position check error {}: {}", symbol, exc)


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
