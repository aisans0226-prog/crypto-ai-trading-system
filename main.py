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
from typing import Dict, Optional

import uvicorn
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from utils.logger import setup_logger

from data_engine.market_data import MarketDataEngine
from data_engine.websocket_feed import BinanceWebSocketFeed, MarketCache

from scanners.market_scanner import MarketScanner
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

        # WS feed handle (populated in start())
        self._ws_feed = None

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
        await self._alerts.start()
        await self._sentiment.start()
        await self._binance_exec.start()
        await self._bybit_exec.start()

        # Fetch symbol list for WS feeds
        symbols = await self._data_engine.get_binance_symbols()
        symbols = symbols[: settings.max_coins_to_scan]

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

    # ── Main scan loop ────────────────────────────────────────────────────
    async def _scan_loop(self) -> None:
        while self._running:
            try:
                logger.info("── Scan cycle starting ──")
                signals = await self._scanner.scan_all()

                for signal in signals:
                    if not signal.is_high_probability:
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
                        # Record feature vector for self-learning
                        trade_id = f"{signal.symbol}_{int(time.time())}"
                        self._learner.record_prediction(trade_id, signal.symbol, df)
                        self._pending_labels[signal.symbol] = trade_id
                        self._klines_cache[signal.symbol] = df
                    except Exception as exc:
                        logger.debug("ML predict error {}: {}", signal.symbol, exc)

                    # Sentiment
                    try:
                        sent_score, sent_signals = \
                            await self._sentiment.get_sentiment_score(signal.symbol)
                        if sent_score >= 0.3:
                            signal.score += 1
                            signal.signals.extend(sent_signals)
                    except Exception:
                        pass

                    # Skip if still below threshold after ML + sentiment
                    if not signal.is_high_probability:
                        continue

                    # Store in dashboard state
                    state.recent_signals.append({
                        "symbol": signal.symbol,
                        "score": signal.score,
                        "direction": signal.direction,
                        "signals": signal.signals,
                        "price": signal.price,
                        "price_change_pct": signal.price_change_pct,
                        "ai_prediction": signal.ai_prediction,
                    })

                    # Send alert
                    await self._alerts.send_trade_signal(signal)
                    await broadcast("signal", {"symbol": signal.symbol, "score": signal.score})

                    # Attempt trade execution
                    await self._try_execute_trade(signal)

                logger.info(
                    "── Scan complete. Next scan in {}s ──",
                    settings.scan_interval_seconds,
                )
                await asyncio.sleep(settings.scan_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Scan loop error: {}", exc)
                await asyncio.sleep(10)

    # ── Trade execution ────────────────────────────────────────────────────
    async def _try_execute_trade(self, signal) -> None:
        # Check open trade limit
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

        if self.dry_run:
            logger.info("DRY RUN — would open: {} {}", setup.symbol, setup.direction)
            await self._portfolio.open_position({
                "symbol": risk_params.symbol,
                "direction": risk_params.direction,
                "entry_price": risk_params.entry_price,
                "stop_loss": risk_params.stop_loss,
                "take_profit": risk_params.take_profit,
                "quantity": risk_params.quantity,
                "leverage": risk_params.leverage,
                "position_size_usdt": risk_params.position_size_usdt,
                "exchange": "dry_run",
                "signal_score": signal.score,
            })
            return

        # Live execution
        if self.exchange == "bybit":
            result = await self._bybit_exec.execute_trade(risk_params)
        else:
            result = await self._binance_exec.execute_trade(risk_params)

        if result.get("status") == "open":
            await self._portfolio.open_position({
                "symbol": risk_params.symbol,
                "direction": risk_params.direction,
                "entry_price": risk_params.entry_price,
                "stop_loss": risk_params.stop_loss,
                "take_profit": risk_params.take_profit,
                "quantity": risk_params.quantity,
                "leverage": risk_params.leverage,
                "position_size_usdt": risk_params.position_size_usdt,
                "exchange": self.exchange,
                "order_id": str(result.get("order_id", "")),
                "signal_score": signal.score,
            })
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

                    rankings = await self._ranker.rank(self._klines_cache)
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
                if metrics:
                    balance = metrics.get("balance", settings.account_balance_usdt)
                    self._risk.update_balance(balance)
                    await broadcast("metrics", metrics)
            except Exception as exc:
                logger.debug("Portfolio sync error: {}", exc)
            await asyncio.sleep(30)

    async def _sync_open_trades(self) -> None:
        positions = await self._portfolio.get_open_positions()
        self._risk.update_open_trades(len(positions))


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
    """Run trading system + uvicorn dashboard concurrently."""
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
        await asyncio.gather(
            run_trading(args),
            server.serve(),
        )


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
