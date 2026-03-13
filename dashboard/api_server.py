"""
dashboard/api_server.py - Professional responsive FastAPI dashboard v2.
No authentication required — open access on local/VPS network.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from config import settings
import ai_engine.llm_analyzer as llm_analyzer

@asynccontextmanager
async def lifespan(app_inst: "FastAPI"):
    llm_analyzer.update_cfg(_ai_cfg)
    _rt_task = asyncio.create_task(_realtime_loop())
    try:
        yield
    finally:
        _rt_task.cancel()
        try:
            await _rt_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Crypto AI Dashboard", version="2.0.0", docs_url="/docs", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class AppState:
    portfolio_manager = None
    market_scanner = None
    self_learner = None
    coin_ranker = None
    auto_updater = None
    coin_database = None          # CoinDatabase — per-coin learning stats
    strategy_registry = None      # StrategyRegistry — auto strategy discovery
    trading_system = None         # TradingSystem — set in main.py __init__
    pending_entry_orders: dict = {}   # symbol -> PendingEntry (LIMIT orders awaiting fill)
    recent_signals: List[dict] = []
    coin_rankings: List[dict] = []
    connected_ws: List[WebSocket] = []
    # Bot operational state — updated by main.py scan loop
    bot_stats: dict = {
        "last_scan_ts": None,
        "watchlist_count": 0,
        "daily_trades_today": 0,
        "scan_cycle": 0,
        "started_at": None,
    }


state = AppState()


def _make_bot_status_payload() -> dict:
    """Build bot operational status dict — used in WS broadcast and initial push."""
    import time as _t
    from config import settings as _cfg
    started = state.bot_stats.get("started_at")
    return {
        "last_scan_ts":       state.bot_stats.get("last_scan_ts"),
        "watchlist_count":    state.bot_stats.get("watchlist_count", 0),
        "daily_trades_today": state.bot_stats.get("daily_trades_today", 0),
        "scan_cycle":         state.bot_stats.get("scan_cycle", 0),
        "uptime_seconds":     round(_t.time() - started, 0) if started else 0,
        "signals_in_memory":  len(state.recent_signals),
        "training_mode":      _cfg.effective_training_mode,
        "dry_run":            getattr(_cfg, "dry_run", False),
        "score_threshold":    _cfg.effective_signal_score_threshold,
        "max_daily_trades":   _cfg.effective_max_daily_trades,
        "is_paused":          getattr(state.trading_system, "is_paused", False) if state.trading_system else False,
    }


async def broadcast(event: str, payload: dict) -> None:
    msg = json.dumps({"type": event, "data": payload, "ts": datetime.utcnow().isoformat()})
    dead: List[WebSocket] = []
    for ws in state.connected_ws:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.connected_ws:
            state.connected_ws.remove(ws)


@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


async def _fetch_all_prices() -> dict:
    """Fetch ALL futures prices — testnet-aware, proxy-aware, Bybit fallback."""
    import aiohttp
    # Respect BINANCE_TESTNET flag so geo-blocked VPS still gets prices
    base = (
        "https://testnet.binancefuture.com"
        if getattr(settings, "binance_testnet", False)
        else "https://fapi.binance.com"
    )
    proxy = getattr(settings, "http_proxy", None) or None
    try:
        async with aiohttp.ClientSession() as session:
            req_kw: dict = {"timeout": aiohttp.ClientTimeout(total=5)}
            if proxy:
                req_kw["proxy"] = proxy
            async with session.get(f"{base}/fapi/v1/ticker/price", **req_kw) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {item["symbol"]: float(item["price"]) for item in data}
    except Exception:
        pass

    # Bybit fallback — covers fully geo-blocked Binance or Bybit-exchange deployments
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.bybit.com/v5/market/tickers?category=linear",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    prices: dict = {}
                    for item in data.get("result", {}).get("list", []):
                        sym = item.get("symbol", "")
                        lp  = item.get("lastPrice")
                        if sym.endswith("USDT") and lp:
                            try:
                                prices[sym] = float(lp)
                            except Exception:
                                pass
                    return prices
    except Exception:
        pass

    return {}


async def _fetch_current_prices(symbols: list) -> dict:
    """Fetch current prices for a specific list of symbols (filters from full snapshot)."""
    all_p = await _fetch_all_prices()
    sym_set = set(symbols)
    return {k: v for k, v in all_p.items() if k in sym_set}


async def _enrich_positions(positions: Dict[str, dict], prices: Optional[dict] = None) -> None:
    """Add current_price / unrealized_pnl / liq_price / rr_live in-place.

    If `prices` is provided (pre-fetched snapshot) no extra API call is made.
    """
    if not positions:
        return
    if prices is None:
        prices = await _fetch_current_prices(list(positions.keys()))
    for sym, pos in positions.items():
        cp = prices.get(sym)
        if not cp:
            continue
        qty       = pos.get("quantity", 0)
        entry     = pos.get("entry_price", 0) or 0
        leverage  = pos.get("leverage", 1) or 1
        size      = pos.get("position_size_usdt") or (qty * entry) or 1
        direction = pos.get("direction", "LONG")
        # margin = collateral at risk; PnL % is ROI on margin (not notional)
        margin    = size / leverage

        if direction == "LONG":
            upnl      = (cp - entry) * qty
            # Simplified isolated-margin liquidation estimate
            liq_price = round(entry * max(0.0, 1.0 - 0.9 / leverage), 6) if leverage > 1 else 0.0
        else:
            upnl      = (entry - cp) * qty
            liq_price = round(entry * (1.0 + 0.9 / leverage), 6) if leverage > 1 else 0.0

        pos["current_price"]        = round(cp, 6)
        pos["unrealized_pnl_usdt"]  = round(upnl, 2)
        pos["unrealized_pnl_pct"]   = round(upnl / margin * 100, 2) if margin else 0.0
        pos["liq_price"]            = liq_price

        # Live Risk:Reward based on current price vs remaining SL/TP distance
        sl = pos.get("stop_loss", 0) or 0
        tp = pos.get("take_profit", 0) or 0
        if sl and tp:
            sl_dist = (cp - sl) if direction == "LONG" else (sl - cp)
            tp_dist = (tp - cp) if direction == "LONG" else (cp - tp)
            pos["rr_live"] = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0
        else:
            pos["rr_live"] = 0.0


@app.get("/api/portfolio")
async def get_portfolio():
    if not state.portfolio_manager:
        return {"balance": settings.account_balance_usdt, "positions": {}, "risk_exposure": {}}
    balance   = await state.portfolio_manager.get_balance()
    positions = await state.portfolio_manager.get_open_positions()
    exposure  = await state.portfolio_manager.get_risk_exposure()
    await _enrich_positions(positions)
    return {
        "balance": balance,
        "positions": positions,
        "risk_exposure": exposure,
        "manual_balance_active": state.portfolio_manager.is_manual_balance_active(),
    }


@app.post("/api/set-balance")
async def set_balance_override(request: Request):
    """Set a manual account balance. PnL from closed trades compounds on top in real-time."""
    if not state.portfolio_manager:
        return {"ok": False, "error": "Portfolio not initialised"}
    body = await request.json()

    if body.get("clear"):
        await state.portfolio_manager.clear_manual_balance()
        return {"ok": True, "message": "Manual balance cleared — using live exchange data"}

    balance = body.get("balance")
    if not balance or float(balance) <= 0:
        return {"ok": False, "error": "Balance must be > 0"}

    balance = float(balance)
    await state.portfolio_manager.set_manual_balance(balance)
    # Sync risk manager so position sizing uses the new balance immediately
    if state.trading_system and hasattr(state.trading_system, "_risk"):
        state.trading_system._risk.update_balance(balance)
    return {"ok": True, "message": f"Balance set to ${balance:.2f}"}


@app.get("/api/metrics")
async def get_metrics():
    if not state.portfolio_manager:
        return {
            "total_trades": 0, "open_trades": 0, "wins_count": 0, "losses_count": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "daily_pnl": 0.0,
            "total_pnl": 0.0, "balance": settings.account_balance_usdt, "pnl_trend": [],
            "gross_profit": 0.0, "gross_loss": 0.0, "total_funding_fees": 0.0,
        }
    return await state.portfolio_manager.calculate_metrics()


@app.get("/api/advanced-stats")
async def get_advanced_stats():
    """Extended analytics: avg duration, best/worst trade, monthly PnL, PnL by symbol/weekday."""
    empty = {
        "avg_duration_hours": 0.0, "best_trade_pnl": 0.0, "worst_trade_pnl": 0.0,
        "avg_pnl_per_trade": 0.0, "avg_signal_score": 0.0,
        "current_win_streak": 0, "current_loss_streak": 0, "max_loss_streak": 0,
        "monthly_pnl": [], "pnl_by_weekday": [], "pnl_by_symbol": [],
        # Enhanced analytics
        "avg_win_usdt": 0.0, "avg_loss_usdt": 0.0, "expected_value": 0.0,
        "sharpe_ratio": 0.0, "calmar_ratio": 0.0, "sortino_ratio": 0.0,
        "pnl_by_hour": [], "score_distribution": [],
        "total_wins": 0, "total_losses": 0, "gross_profit": 0.0, "gross_loss": 0.0,
        "win_rate_by_hour": [], "win_rate_by_weekday": [],
        "fee_impact_pct": 0.0, "total_funding_fees": 0.0,
    }
    if not state.portfolio_manager:
        return empty
    from sqlalchemy import select, func
    from portfolio.portfolio_manager import TradeRecord
    async with state.portfolio_manager._session_factory() as session:
        result = await session.execute(
            select(TradeRecord)
            .where(TradeRecord.status == "closed")
            .order_by(TradeRecord.closed_at.asc())
        )
        trades = result.scalars().all()

    if not trades:
        return empty

    # --- Avg duration ---
    durations = []
    for t in trades:
        if t.opened_at and t.closed_at:
            durations.append((t.closed_at - t.opened_at).total_seconds() / 3600)
    avg_dur = round(sum(durations) / len(durations), 2) if durations else 0.0

    # --- PnL stats ---
    pnls = [t.pnl_usdt or 0 for t in trades]
    best  = round(max(pnls), 2) if pnls else 0.0
    worst = round(min(pnls), 2) if pnls else 0.0
    avg_p = round(sum(pnls) / len(pnls), 2) if pnls else 0.0

    # --- Avg signal score ---
    scores = [t.signal_score for t in trades if t.signal_score is not None]
    avg_sc = round(sum(scores) / len(scores), 1) if scores else 0.0

    # --- Win/loss streak (from latest) ---
    win_streak = loss_streak = 0
    for t in reversed(trades):
        p = t.pnl_usdt or 0
        if win_streak == 0 and loss_streak == 0:
            if p > 0: win_streak = 1
            else:     loss_streak = 1
        elif win_streak > 0:
            if p > 0: win_streak += 1
            else:     break
        else:
            if p <= 0: loss_streak += 1
            else:      break

    # --- Monthly PnL (last 12 months) ---
    from collections import defaultdict
    monthly: dict = defaultdict(float)
    for t in trades:
        if t.closed_at:
            key = t.closed_at.strftime("%Y-%m")
            monthly[key] += t.pnl_usdt or 0
    monthly_list = [{"month": k, "pnl": round(v, 2)} for k, v in sorted(monthly.items())[-12:]]

    # --- PnL by weekday (0=Mon…6=Sun) ---
    wd_pnl: dict = defaultdict(float)
    wd_count: dict = defaultdict(int)
    for t in trades:
        if t.closed_at:
            wd = t.closed_at.weekday()
            wd_pnl[wd] += t.pnl_usdt or 0
            wd_count[wd] += 1
    wd_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pnl_by_wd = [{"day": wd_labels[i], "pnl": round(wd_pnl.get(i, 0), 2), "count": wd_count.get(i, 0)} for i in range(7)]

    # --- PnL by symbol (top 15) with per-symbol win rate ---
    sym_pnl: dict = defaultdict(float)
    sym_cnt: dict = defaultdict(int)
    sym_win: dict = defaultdict(int)
    for t in trades:
        sym_pnl[t.symbol] += t.pnl_usdt or 0
        sym_cnt[t.symbol] += 1
        if (t.pnl_usdt or 0) > 0:
            sym_win[t.symbol] += 1
    pnl_by_sym = sorted(
        [{"symbol": s, "pnl": round(p, 2), "trades": sym_cnt[s],
          "win_rate": round(sym_win[s] / sym_cnt[s] * 100, 1) if sym_cnt[s] else 0.0}
         for s, p in sym_pnl.items()],
        key=lambda x: x["pnl"], reverse=True
    )[:15]

    # --- Avg win / loss ---
    win_pnls  = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]
    avg_win   = round(sum(win_pnls) / len(win_pnls), 2)   if win_pnls  else 0.0
    avg_loss  = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0
    gross_profit = round(sum(win_pnls), 2)
    gross_loss   = round(sum(loss_pnls), 2)

    # --- Expected Value per trade ---
    total_c = len(pnls)
    wr_frac  = len(win_pnls) / total_c if total_c else 0.0
    lr_frac  = len(loss_pnls) / total_c if total_c else 0.0
    ev = round(wr_frac * avg_win + lr_frac * avg_loss, 2)

    # --- Sharpe Ratio (simplified: annualised per-trade std) ---
    import statistics as _stats
    sharpe = 0.0
    if len(pnls) >= 3:
        mu  = sum(pnls) / len(pnls)
        std = _stats.stdev(pnls) or 1e-9
        sharpe = round(mu / std * (len(pnls) ** 0.5), 3)

    # --- Calmar Ratio = total PnL / max drawdown (USDT) ---
    # Measures return quality relative to worst peak-to-trough loss
    calmar = 0.0
    initial_balance = settings.account_balance_usdt
    cumulative_curve = [initial_balance + sum(pnls[:i+1]) for i in range(len(pnls))]
    peak_eq = initial_balance
    max_dd_usdt = 0.0
    for val in cumulative_curve:
        if val > peak_eq:
            peak_eq = val
        dd_usdt = peak_eq - val
        max_dd_usdt = max(max_dd_usdt, dd_usdt)
    if max_dd_usdt > 0:
        calmar = round(sum(pnls) / max_dd_usdt, 3)

    # --- Sortino Ratio = mean / downside std (penalises losses only) ---
    sortino = 0.0
    if len(pnls) >= 3:
        mu = sum(pnls) / len(pnls)
        downside = [p for p in pnls if p < 0]
        if downside:
            import math as _math
            ds_std = (_math.sqrt(sum((p - 0) ** 2 for p in downside) / len(downside))) or 1e-9
            sortino = round(mu / ds_std, 3)

    # --- Max historical consecutive-loss streak ---
    max_loss_streak = 0
    cur_ls = 0
    for t in trades:
        if (t.pnl_usdt or 0) <= 0:
            cur_ls += 1
            max_loss_streak = max(max_loss_streak, cur_ls)
        else:
            cur_ls = 0

    # --- Fee impact: estimated fees as % of gross profit ---
    taker_pct = getattr(settings, "taker_fee_pct", 0.06)
    estimated_total_fees = sum(abs(t.pnl_usdt or 0) + (
        t.pnl_usdt or 0
    ) for t in trades)  # placeholder; use simpler approach:
    # round-trip taker fee × notional; notional ≈ entry × qty per trade
    # We compute it as: 2 × taker_pct% × position_size for each trade
    # position_size_usdt not stored, so approximate from entry × qty / leverage
    fee_approx_total = 0.0
    for t in trades:
        notional = (t.entry_price or 0) * (t.quantity or 0)
        fee_approx_total += notional * (taker_pct / 100.0) * 2
    fee_impact_pct = round(fee_approx_total / gross_profit * 100, 1) if gross_profit > 0 else 0.0

    # --- Win rate by hour of day (separate from PnL) ---
    hr_wins: dict = defaultdict(int)
    hr_tot: dict = defaultdict(int)
    for t in trades:
        if t.closed_at:
            h = t.closed_at.hour
            hr_tot[h] += 1
            if (t.pnl_usdt or 0) > 0:
                hr_wins[h] += 1
    win_rate_by_hour = [
        {
            "hour":     i,
            "win_rate": round(hr_wins.get(i, 0) / hr_tot[i] * 100, 1) if hr_tot.get(i) else 0.0,
            "count":    hr_tot.get(i, 0),
        }
        for i in range(24)
    ]

    # --- Win rate by weekday ---
    wd_wins: dict = defaultdict(int)
    wd_tot: dict = defaultdict(int)
    for t in trades:
        if t.closed_at:
            wd = t.closed_at.weekday()
            wd_tot[wd] += 1
            if (t.pnl_usdt or 0) > 0:
                wd_wins[wd] += 1
    wd_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    win_rate_by_wd = [
        {
            "day":      wd_labels[i],
            "win_rate": round(wd_wins.get(i, 0) / wd_tot[i] * 100, 1) if wd_tot.get(i) else 0.0,
            "count":    wd_tot.get(i, 0),
        }
        for i in range(7)
    ]

    # --- PnL by hour of day (0–23, UTC) ---
    hr_pnl: dict = defaultdict(float)
    hr_cnt: dict = defaultdict(int)
    for t in trades:
        if t.closed_at:
            hr_pnl[t.closed_at.hour] += t.pnl_usdt or 0
            hr_cnt[t.closed_at.hour] += 1
    pnl_by_hour = [
        {"hour": i, "pnl": round(hr_pnl.get(i, 0), 2), "count": hr_cnt.get(i, 0)}
        for i in range(24)
    ]

    # --- Score distribution: win rate per signal-score bucket ---
    sc_bkts: dict = {"<7": {"wins": 0, "total": 0}, "7–9": {"wins": 0, "total": 0},
                     "9–11": {"wins": 0, "total": 0}, "11–13": {"wins": 0, "total": 0},
                     "13+": {"wins": 0, "total": 0}}
    for t in trades:
        sc = t.signal_score or 0
        if   sc < 7:  k = "<7"
        elif sc < 9:  k = "7–9"
        elif sc < 11: k = "9–11"
        elif sc < 13: k = "11–13"
        else:         k = "13+"
        sc_bkts[k]["total"] += 1
        if (t.pnl_usdt or 0) > 0:
            sc_bkts[k]["wins"] += 1
    score_distribution = [
        {"range": k, "total": v["total"], "wins": v["wins"],
         "win_rate": round(v["wins"] / v["total"] * 100, 1) if v["total"] else 0.0}
        for k, v in sc_bkts.items()
    ]

    return {
        "avg_duration_hours":   avg_dur,
        "best_trade_pnl":       best,
        "worst_trade_pnl":      worst,
        "avg_pnl_per_trade":    avg_p,
        "avg_signal_score":     avg_sc,
        "current_win_streak":   win_streak,
        "current_loss_streak":  loss_streak,
        "max_loss_streak":      max_loss_streak,
        "monthly_pnl":          monthly_list,
        "pnl_by_weekday":       pnl_by_wd,
        "pnl_by_symbol":        pnl_by_sym,
        # Enhanced analytics
        "avg_win_usdt":         avg_win,
        "avg_loss_usdt":        avg_loss,
        "expected_value":       ev,
        "sharpe_ratio":         sharpe,
        "calmar_ratio":         calmar,
        "sortino_ratio":        sortino,
        "gross_profit":         gross_profit,
        "gross_loss":           gross_loss,
        "total_wins":           len(win_pnls),
        "total_losses":         len(loss_pnls),
        "fee_impact_pct":       fee_impact_pct,
        "total_funding_fees":   round(sum(t.funding_fee_usdt or 0.0 for t in trades), 4),
        "pnl_by_hour":          pnl_by_hour,
        "win_rate_by_hour":     win_rate_by_hour,
        "win_rate_by_weekday":  win_rate_by_wd,
        "score_distribution":   score_distribution,
    }


@app.get("/api/bot-status")
async def get_bot_status():
    """Real-time bot operational status and configuration snapshot."""
    from config import settings as _s
    import time as _t
    started = state.bot_stats.get("started_at")
    uptime_s = round(_t.time() - started, 0) if started else 0
    is_paused = getattr(state.trading_system, "is_paused", False) if state.trading_system else False
    return {
        "last_scan_ts":      state.bot_stats.get("last_scan_ts"),
        "watchlist_count":   state.bot_stats.get("watchlist_count", 0),
        "daily_trades_today": state.bot_stats.get("daily_trades_today", 0),
        "scan_cycle":        state.bot_stats.get("scan_cycle", 0),
        "uptime_seconds":    uptime_s,
        "signals_in_memory": len(state.recent_signals),
        "ws_clients":        len(state.connected_ws),
        "training_mode":     getattr(_s, "training_mode", False),
        "score_threshold":   getattr(_s, "effective_signal_score_threshold", _s.signal_score_threshold),
        "max_daily_trades":  getattr(_s, "effective_max_daily_trades", _s.max_daily_trades),
        "dry_run":           getattr(_s, "dry_run", True),
        "is_paused":         is_paused,
    }


@app.post("/api/bot/pause")
async def pause_bot():
    """Pause bot trading — scanning continues but no new trades are opened."""
    if not state.trading_system:
        return {"ok": False, "error": "Bot not running"}
    state.trading_system._bot_paused = True
    logger.info("Bot trading paused via dashboard")
    await broadcast("bot_state", {"paused": True})
    return {"ok": True, "paused": True}


@app.post("/api/bot/resume")
async def resume_bot():
    """Resume bot trading after a pause."""
    if not state.trading_system:
        return {"ok": False, "error": "Bot not running"}
    state.trading_system._bot_paused = False
    logger.info("Bot trading resumed via dashboard")
    await broadcast("bot_state", {"paused": False})
    return {"ok": True, "paused": False}


@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str):
    """Force-close a single open position at market price."""
    if not state.trading_system:
        return {"ok": False, "error": "Bot not running"}
    return await state.trading_system.close_position_from_dashboard(symbol.upper())


@app.post("/api/positions/close-all")
async def close_all_positions():
    """Force-close ALL open positions at market price."""
    if not state.trading_system:
        return {"ok": False, "error": "Bot not running"}
    return await state.trading_system.close_all_positions_from_dashboard()


@app.get("/api/signals")
async def get_signals():
    return {"signals": state.recent_signals[-100:]}


@app.get("/api/pending-orders")
async def get_pending_orders():
    """Return all pending LIMIT entry orders with elapsed/remaining time."""
    import time as _time
    now = _time.time()
    from config import settings as _s
    result = []
    for sym, p in state.pending_entry_orders.items():
        elapsed = now - p.placed_at
        remaining = max(0.0, _s.limit_order_timeout_seconds - elapsed)
        result.append({
            "symbol":        p.symbol,
            "order_id":      p.order_id,
            "direction":     p.risk_params.direction,
            "limit_price":   p.risk_params.entry_price,
            "stop_loss":     p.risk_params.stop_loss,
            "take_profit":   p.risk_params.take_profit,
            "quantity":      p.risk_params.quantity,
            "retry_count":   p.retry_count,
            "max_retries":   _s.limit_order_max_retries,
            "elapsed_s":     round(elapsed, 1),
            "remaining_s":   round(remaining, 1),
            "timeout_s":     _s.limit_order_timeout_seconds,
            "placed_at":     p.placed_at,
        })
    return {"pending_orders": result, "count": len(result)}


@app.get("/api/heatmap")
async def get_heatmap():
    coins = [
        {"symbol": s.get("symbol"), "score": s.get("score"),
         "direction": s.get("direction"), "price_change_pct": s.get("price_change_pct", 0),
         "volume_24h": s.get("volume_24h", 0), "ai_prediction": s.get("ai_prediction", 0.5)}
        for s in state.recent_signals[-150:]
    ]
    return {"coins": coins}


@app.get("/api/ai-rankings")
async def get_ai_rankings():
    return {"rankings": state.coin_rankings[:50]}


@app.get("/api/learning-stats")
async def get_learning_stats():
    if not state.self_learner:
        return {"stats": {}, "top_coins": []}
    stats = state.self_learner.get_stats()
    top_coins_raw = state.self_learner.get_top_coins(10)
    coin_stats_dict = state.self_learner._coin_stats
    top_coins = []
    for sym, wr in top_coins_raw:
        cs = coin_stats_dict.get(sym, {})
        top_coins.append({
            "symbol":   sym,
            "win_rate": round(wr * 100, 1),
            "wins":     cs.get("wins", 0),
            "losses":   cs.get("losses", 0),
            "total":    cs.get("total", 0),
        })
    return {"stats": stats, "top_coins": top_coins}


@app.get("/api/learning-insights")
async def get_learning_insights():
    """Deep learning insights: funnel, research quality, rolling win rate, feature importance."""
    result: dict = {
        "labels_timeline":    [],
        "rolling_wr_chart":   [],
        "funnel":             {"scanned": 0, "researched": 0, "executed": 0, "wins": 0, "losses": 0},
        "research_buckets":   [],
        "feature_importance": [],
    }

    if state.self_learner:
        result["labels_timeline"]    = state.self_learner.get_labels_timeline(50)
        result["feature_importance"] = state.self_learner.get_feature_importance(10)
        # Build rolling win-rate series for the learning curve chart
        tl = state.self_learner._labels_timeline
        chart_pts = []
        for i in range(len(tl)):
            window = [x["label"] for x in tl[max(0, i - 9): i + 1]]
            chart_pts.append({
                "idx": i + 1,
                "ts":  tl[i]["ts"],
                "wr":  round(sum(window) / len(window) * 100, 1),
                "sym": tl[i]["symbol"],
            })
        result["rolling_wr_chart"] = chart_pts[-50:]  # last 50 data points

    if state.coin_database:
        # Signal→trade funnel (aggregate over all coins)
        try:
            from sqlalchemy import func, select
            from data_engine.coin_database import CoinStats
            async with state.coin_database._sf() as session:
                agg = await session.execute(
                    select(
                        func.sum(CoinStats.total_signals).label("scanned"),
                        func.sum(CoinStats.times_researched).label("researched"),
                        func.sum(CoinStats.times_executed).label("executed"),
                        func.sum(CoinStats.wins).label("wins"),
                        func.sum(CoinStats.losses).label("losses"),
                    )
                )
                row = agg.fetchone()
                if row:
                    result["funnel"] = {
                        "scanned":    int(row.scanned   or 0),
                        "researched": int(row.researched or 0),
                        "executed":   int(row.executed   or 0),
                        "wins":       int(row.wins        or 0),
                        "losses":     int(row.losses      or 0),
                    }
        except Exception as exc:
            logger.warning("learning-insights funnel query failed: {}", exc)

        # Research score vs actual outcome — bucketed into 5 ranges
        try:
            from sqlalchemy import select
            from data_engine.coin_database import ResearchLog
            async with state.coin_database._sf() as session:
                res = await session.execute(
                    select(ResearchLog)
                    .where(ResearchLog.outcome_pnl.isnot(None))
                    .order_by(ResearchLog.id.desc())
                    .limit(200)
                )
                rl_rows = res.scalars().all()
            buckets = {k: {"wins": 0, "total": 0} for k in ("0–2", "2–4", "4–6", "6–8", "8–10")}
            bracket_map = [(0, 2, "0–2"), (2, 4, "2–4"), (4, 6, "4–6"), (6, 8, "6–8"), (8, 10.01, "8–10")]
            for r in rl_rows:
                for lo, hi, lbl in bracket_map:
                    if lo <= r.research_score < hi:
                        buckets[lbl]["total"] += 1
                        if r.outcome_pnl > 0:
                            buckets[lbl]["wins"] += 1
                        break
            result["research_buckets"] = [
                {
                    "range":    k,
                    "total":    v["total"],
                    "wins":     v["wins"],
                    "win_rate": round(v["wins"] / v["total"] * 100, 1) if v["total"] > 0 else 0.0,
                }
                for k, v in buckets.items()
            ]
        except Exception as exc:
            logger.warning("learning-insights research buckets failed: {}", exc)

    return result


@app.get("/api/update/status")
async def update_status():
    if not state.auto_updater:
        return {"update_available": False, "current_commit": "unknown"}
    return state.auto_updater.get_status()


@app.post("/api/update/check")
async def update_check():
    if not state.auto_updater:
        return {"update_available": False}
    return await state.auto_updater.check_for_update()


@app.post("/api/update/apply")
async def update_apply():
    if not state.auto_updater:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Updater not available")
    result = await state.auto_updater.apply_update()
    await broadcast("system_update", result)
    return result


@app.post("/api/update/rollback")
async def update_rollback():
    if not state.auto_updater:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Updater not available")
    result = await state.auto_updater.rollback()
    await broadcast("system_update", result)
    return result



@app.get("/api/trades")
async def get_trades(limit: int = 100, status: str = "all"):
    """Return trade history with win-rate stats."""
    if not state.portfolio_manager:
        return {"trades": [], "stats": {}}
    from sqlalchemy import select
    from portfolio.portfolio_manager import TradeRecord
    async with state.portfolio_manager._session_factory() as session:
        query = select(TradeRecord).order_by(TradeRecord.id.desc()).limit(limit)
        if status != "all":
            query = query.where(TradeRecord.status == status)
        result = await session.execute(query)
        trades = result.scalars().all()

    rows = [
        {
            "id":               t.id,
            "symbol":           t.symbol,
            "direction":        t.direction,
            "entry_price":      t.entry_price,
            "exit_price":       t.exit_price,
            "quantity":         t.quantity,
            "leverage":         t.leverage,
            "stop_loss":        t.stop_loss,
            "take_profit":      t.take_profit,
            "pnl_usdt":         t.pnl_usdt,
            "funding_fee_usdt": round(t.funding_fee_usdt or 0.0, 4),
            "status":           t.status,
            "exchange":         t.exchange,
            "signal_score":     t.signal_score,
            "strategy":         getattr(t, "strategy_name", None),
            "opened_at":        t.opened_at.isoformat() if t.opened_at else None,
            "closed_at":        t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in trades
    ]

    closed = [r for r in rows if r["status"] == "closed"]
    wins   = [r for r in closed if r["pnl_usdt"] and r["pnl_usdt"] > 0]
    stats  = {
        "total":              len(rows),
        "open":               len([r for r in rows if r["status"] == "open"]),
        "closed":             len(closed),
        "wins":               len(wins),
        "losses":             len(closed) - len(wins),
        "win_rate_pct":       round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl":          round(sum(r["pnl_usdt"] or 0 for r in closed), 2),
        "total_funding_fees": round(sum(r["funding_fee_usdt"] or 0 for r in closed), 4),
        "gross_profit":       round(sum(r["pnl_usdt"] for r in wins), 2) if wins else 0.0,
        "gross_loss":         round(sum(r["pnl_usdt"] for r in closed if (r["pnl_usdt"] or 0) <= 0), 2),
    }
    return {"trades": rows, "stats": stats}

@app.get("/api/coin-stats")
async def get_coin_stats(limit: int = 200):
    """All tracked coins from the self-learning database, sorted by executions then signals."""
    if not state.coin_database:
        return {"coins": [], "total": 0}
    coins = await state.coin_database.get_all_coin_stats(limit)
    return {"coins": coins, "total": len(coins)}


@app.get("/api/coin-stats/{symbol}")
async def get_single_coin_stats(symbol: str):
    """Full learning stats for one symbol."""
    if not state.coin_database:
        return {}
    stats = await state.coin_database.get_coin_stats(symbol.upper())
    return stats or {}


@app.get("/api/research-log")
async def get_research_log(limit: int = 50):
    """Recent deep-research decisions (passed/failed + outcome when available)."""
    if not state.coin_database:
        return {"log": []}
    log = await state.coin_database.get_recent_research(limit)
    return {"log": log}


@app.get("/api/strategy-stats")
async def get_strategy_stats():
    """Per-strategy performance: win rate, PnL, recent trend, best strategy badge."""
    if not state.coin_database:
        return {"strategies": [], "best": None}
    rows = await state.coin_database.get_strategy_stats()
    # Determine best strategy: highest win_rate among those with ≥ 3 trades
    qualified = [r for r in rows if r["total_trades"] >= 3]
    best = max(qualified, key=lambda r: r["win_rate"])["name"] if qualified else None
    # Also inject all registered strategy names with zero stats for new strategies
    all_names: list = []
    if state.strategy_registry:
        all_names = state.strategy_registry.strategy_names()
    existing = {r["name"] for r in rows}
    for name in all_names:
        if name not in existing:
            rows.append({
                "name": name, "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "recent_pnl": [], "updated_at": None,
            })
    return {"strategies": rows, "best": best}


@app.get("/api/performance-history")
async def get_performance_history(days: int = 90):
    """Daily performance snapshots for historical trend charts (last N days)."""
    if not state.portfolio_manager:
        return {"history": []}
    history = await state.portfolio_manager.get_performance_history(days)
    return {"history": history}


@app.get("/api/bot-config")
async def get_bot_config():
    """Dynamic bot configuration — all tunable parameters from settings.
    Replaces the hardcoded config table on the System page."""
    from config import settings as _s
    return {
        "risk": {
            "risk_per_trade_pct":       _s.risk_per_trade_pct,
            "max_leverage":             _s.max_leverage,
            "max_open_trades":          _s.effective_max_open_trades,
            "max_daily_trades":         _s.effective_max_daily_trades,
            "min_risk_reward_ratio":    _s.min_risk_reward_ratio,
            "max_stop_loss_pct":        _s.max_stop_loss_pct,
            "min_stop_loss_pct":        _s.min_stop_loss_pct,
            "max_position_size_pct":    _s.max_position_size_pct,
            "balance_reserve_pct":      _s.balance_reserve_pct,
            "max_daily_loss_pct":       _s.max_daily_loss_pct,
        },
        "fees": {
            "taker_fee_pct":            _s.taker_fee_pct,
            "max_funding_rate_pct":     getattr(_s, "max_funding_rate_pct", 0.10),
            "funding_periods_estimate": getattr(_s, "funding_periods_estimate", 3),
        },
        "position_mgmt": {
            "trailing_stop_activation_pct": getattr(_s, "trailing_stop_activation_pct", 3.0),
            "trailing_stop_distance_pct":   getattr(_s, "trailing_stop_distance_pct", 1.0),
            "trailing_stop_min_move_pct":   getattr(_s, "trailing_stop_min_move_pct", 0.3),
            "breakeven_trigger_pct":        getattr(_s, "breakeven_trigger_pct", 2.0),
            "max_position_hold_hours":      getattr(_s, "max_position_hold_hours", 18.0),
            "entry_max_deviation_pct":      getattr(_s, "entry_max_deviation_pct", 0.8),
            "reversal_exit_pct":            getattr(_s, "reversal_exit_pct", 2.0),
            "reversal_exit_min_hours":      getattr(_s, "reversal_exit_min_hours", 2.0),
        },
        "signals": {
            "signal_score_threshold":       _s.effective_signal_score_threshold,
            "watchlist_confirmations":      _s.effective_watchlist_confirmations,
            "signal_cooldown_minutes":      _s.effective_signal_cooldown_minutes,
            "research_min_score":           _s.effective_research_min_score,
            "research_min_mtf_alignment":   _s.effective_research_min_mtf_alignment,
            "min_ml_confidence":            _s.effective_min_ml_confidence,
            "min_volume_usdt":              _s.effective_min_volume_usdt,
            "max_signals_per_scan":         getattr(_s, "max_signals_per_scan", 150),
        },
        "mode": {
            "dry_run":                     getattr(_s, "dry_run", True),
            "training_mode":               getattr(_s, "training_mode", False),
            "exchange":                    getattr(_s, "exchange", "binance"),
            "binance_testnet":             getattr(_s, "binance_testnet", False),
            "trailing_stop_enabled":       getattr(_s, "trailing_stop_enabled", True),
            "breakeven_stop_enabled":      getattr(_s, "breakeven_stop_enabled", True),
            "reversal_exit_enabled":       getattr(_s, "reversal_exit_enabled", True),
            "strategy_discovery_enabled":  getattr(_s, "strategy_discovery_enabled", True),
            "ai_enabled":                  getattr(_s, "ai_analysis_enabled", False),
            "ai_provider":                 getattr(_s, "ai_provider", ""),
            "ai_model":                    getattr(_s, "ai_model", ""),
        },
    }


# Fields that require a restart to take effect (exchange plumbing, network)
_RESTART_REQUIRED = {"exchange", "binance_testnet"}

# payload_key → ENV_VAR_NAME  (only editable fields — no secrets here)
_CFG_KEY_MAP = {
    "risk_per_trade_pct":           "RISK_PER_TRADE_PCT",
    "max_leverage":                 "MAX_LEVERAGE",
    "max_open_trades":              "MAX_OPEN_TRADES",
    "max_daily_trades":             "MAX_DAILY_TRADES",
    "min_risk_reward_ratio":        "MIN_RISK_REWARD_RATIO",
    "max_stop_loss_pct":            "MAX_STOP_LOSS_PCT",
    "min_stop_loss_pct":            "MIN_STOP_LOSS_PCT",
    "max_position_size_pct":        "MAX_POSITION_SIZE_PCT",
    "balance_reserve_pct":          "BALANCE_RESERVE_PCT",
    "max_daily_loss_pct":           "MAX_DAILY_LOSS_PCT",
    "taker_fee_pct":                "TAKER_FEE_PCT",
    "max_funding_rate_pct":         "MAX_FUNDING_RATE_PCT",
    "funding_periods_estimate":     "FUNDING_PERIODS_ESTIMATE",
    "trailing_stop_enabled":        "TRAILING_STOP_ENABLED",
    "trailing_stop_activation_pct": "TRAILING_STOP_ACTIVATION_PCT",
    "trailing_stop_distance_pct":   "TRAILING_STOP_DISTANCE_PCT",
    "trailing_stop_min_move_pct":   "TRAILING_STOP_MIN_MOVE_PCT",
    "breakeven_stop_enabled":       "BREAKEVEN_STOP_ENABLED",
    "breakeven_trigger_pct":        "BREAKEVEN_TRIGGER_PCT",
    "max_position_hold_hours":      "MAX_POSITION_HOLD_HOURS",
    "entry_max_deviation_pct":      "ENTRY_MAX_DEVIATION_PCT",
    "reversal_exit_enabled":        "REVERSAL_EXIT_ENABLED",
    "reversal_exit_pct":            "REVERSAL_EXIT_PCT",
    "reversal_exit_min_hours":      "REVERSAL_EXIT_MIN_HOURS",
    "signal_score_threshold":       "SIGNAL_SCORE_THRESHOLD",
    "min_ml_confidence":            "MIN_ML_CONFIDENCE",
    "watchlist_confirmations":      "WATCHLIST_CONFIRMATIONS",
    "signal_cooldown_minutes":      "SIGNAL_COOLDOWN_MINUTES",
    "research_min_score":           "RESEARCH_MIN_SCORE",
    "research_min_mtf_alignment":   "RESEARCH_MIN_MTF_ALIGNMENT",
    "min_volume_usdt":              "MIN_VOLUME_USDT",
    "max_signals_per_scan":         "MAX_SIGNALS_PER_SCAN",
    "dry_run":                      "DRY_RUN",
    "training_mode":                "TRAINING_MODE",
    "exchange":                     "EXCHANGE",
    "binance_testnet":              "BINANCE_TESTNET",
    "strategy_discovery_enabled":   "STRATEGY_DISCOVERY_ENABLED",
}


@app.post("/api/bot-config/save")
async def save_bot_config(request: Request):
    """Persist bot configuration changes to .env and apply runtime where safe."""
    import re
    payload = await request.json()

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

    def _set(lines: list[str], key: str, value: str) -> list[str]:
        pat = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
        replaced = False
        result = []
        for ln in lines:
            if pat.match(ln):
                result.append(f"{key}={value}\n")
                replaced = True
            else:
                result.append(ln)
        if not replaced:
            result.append(f"{key}={value}\n")
        return result

    changed = 0
    restart_needed = False
    for key, env_var in _CFG_KEY_MAP.items():
        if key not in payload:
            continue
        val = payload[key]
        if isinstance(val, bool):
            val = str(val).lower()
        lines = _set(lines, env_var, str(val))
        changed += 1
        if key in _RESTART_REQUIRED:
            restart_needed = True

    try:
        with open(env_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    except Exception as exc:
        logger.warning("bot-config/save failed writing .env: {}", exc)
        return {"success": False, "message": f"Failed to write .env: {exc}"}

    # Apply runtime — safe numeric/bool fields (no exchange/network fields)
    _s = settings
    for key in payload:
        if key in _RESTART_REQUIRED or key not in _CFG_KEY_MAP:
            continue
        try:
            if hasattr(_s, key):
                field_type = type(getattr(_s, key))
                setattr(_s, key, field_type(payload[key]))
        except Exception:
            pass

    suffix = " — restart required for exchange/testnet changes" if restart_needed else " — applied immediately"
    return {
        "success": True,
        "message": f"Saved {changed} settings{suffix}",
        "restart_required": restart_needed,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# AI LLM Integration endpoints
# ---------------------------------------------------------------------------

# Runtime AI config — loaded from settings on startup, updatable via API
_ai_cfg: dict = {}


def _load_ai_cfg() -> None:
    """Populate runtime config from pydantic settings."""
    _ai_cfg.update({
        "provider":           settings.ai_provider,
        "model":              settings.ai_model,
        "enabled":            settings.ai_analysis_enabled,
        "openai_api_key":     settings.openai_api_key,
        "anthropic_api_key":  settings.anthropic_api_key,
        "gemini_api_key":     settings.gemini_api_key,
    })


_load_ai_cfg()


def _mask(key: str) -> str:
    if not key:
        return ""
    return key[:4] + "…" + key[-4:] if len(key) > 8 else "****"


def _active_key() -> str:
    """Return the API key for the currently configured provider."""
    prov = _ai_cfg.get("provider", "")
    return _ai_cfg.get(f"{prov}_api_key", "")


@app.get("/api/ai-config")
async def get_ai_config():
    """Return current AI provider configuration (API keys masked)."""
    return {
        "provider":            _ai_cfg.get("provider", ""),
        "model":               _ai_cfg.get("model", ""),
        "ai_analysis_enabled": _ai_cfg.get("enabled", False),
        "openai_key_set":      bool(_ai_cfg.get("openai_api_key")),
        "anthropic_key_set":   bool(_ai_cfg.get("anthropic_api_key")),
        "gemini_key_set":      bool(_ai_cfg.get("gemini_api_key")),
        "openai_key_preview":     _mask(_ai_cfg.get("openai_api_key", "")),
        "anthropic_key_preview":  _mask(_ai_cfg.get("anthropic_api_key", "")),
        "gemini_key_preview":     _mask(_ai_cfg.get("gemini_api_key", "")),
    }


@app.post("/api/ai-config/test")
async def test_ai_config(request: Request):
    """Send a minimal request to the AI provider to verify the API key."""
    import aiohttp
    payload  = await request.json()
    provider = payload.get("provider", "").lower().strip()
    api_key  = payload.get("api_key", "").strip()
    model    = payload.get("model", "").strip()

    if not provider:
        return {"success": False, "message": "Provider is required"}
    if not api_key:
        return {"success": False, "message": "API key is required"}

    try:
        async with aiohttp.ClientSession() as session:

            if provider == "openai":
                model = model or "gpt-4o-mini"
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json={"model": model,
                          "messages": [{"role": "user", "content": "Hi"}],
                          "max_tokens": 5},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as resp:
                    data = await resp.json()
                    if resp.status == 200:
                        return {"success": True,
                                "message": f"OpenAI OK — model: {data.get('model', model)}",
                                "model": data.get("model", model)}
                    return {"success": False,
                            "message": data.get("error", {}).get("message", "Authentication failed")}

            elif provider == "anthropic":
                model = model or "claude-3-5-haiku-20241022"
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key,
                             "anthropic-version": "2023-06-01",
                             "Content-Type": "application/json"},
                    json={"model": model, "max_tokens": 5,
                          "messages": [{"role": "user", "content": "Hi"}]},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as resp:
                    data = await resp.json()
                    if resp.status == 200:
                        return {"success": True,
                                "message": f"Anthropic OK — model: {data.get('model', model)}",
                                "model": data.get("model", model)}
                    return {"success": False,
                            "message": data.get("error", {}).get("message", "Authentication failed")}

            elif provider == "gemini":
                model = model or "gemini-1.5-flash"
                url = (f"https://generativelanguage.googleapis.com/v1beta"
                       f"/models/{model}:generateContent?key={api_key}")
                async with session.post(
                    url,
                    json={"contents": [{"parts": [{"text": "Hi"}]}]},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as resp:
                    data = await resp.json()
                    if resp.status == 200:
                        return {"success": True,
                                "message": f"Gemini OK — model: {model}",
                                "model": model}
                    return {"success": False,
                            "message": data.get("error", {}).get("message", "Authentication failed")}

            return {"success": False, "message": f"Unknown provider: {provider}"}

    except asyncio.TimeoutError:
        return {"success": False, "message": "Connection timed out (>12 s) — check your network"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": str(exc)}


@app.post("/api/ai-config/save")
async def save_ai_config(request: Request):
    """Persist AI config to .env and update the in-memory runtime config."""
    import re
    payload  = await request.json()
    provider = payload.get("provider", "").strip().lower()
    api_key  = payload.get("api_key", "").strip()
    model    = payload.get("model", "").strip()
    enabled  = bool(payload.get("enabled", False))

    if not provider:
        return {"success": False, "message": "Provider is required"}

    key_env_map = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini":    "GEMINI_API_KEY",
    }
    if provider not in key_env_map:
        return {"success": False, "message": f"Unknown provider: {provider}"}

    # Update runtime config immediately (no restart needed)
    _ai_cfg["provider"] = provider
    _ai_cfg["model"]    = model
    _ai_cfg["enabled"]  = enabled
    env_key_name = key_env_map[provider]
    # Only update stored key if a real (non-masked) value was provided
    if api_key and "…" not in api_key and len(api_key) > 8:
        _ai_cfg[f"{provider}_api_key"] = api_key

    # Push updated config into the LLM analyzer (live, no restart)
    llm_analyzer.update_cfg(_ai_cfg)

    # Persist to .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

    def _set(lines: list[str], key: str, value: str) -> list[str]:
        pat = re.compile(rf"^\s*{re.escape(key)}\s*=", re.IGNORECASE)
        replaced = False
        result = []
        for ln in lines:
            if pat.match(ln):
                result.append(f"{key}={value}\n")
                replaced = True
            else:
                result.append(ln)
        if not replaced:
            result.append(f"{key}={value}\n")
        return result

    lines = _set(lines, "AI_PROVIDER", provider)
    lines = _set(lines, "AI_MODEL", model)
    lines = _set(lines, "AI_ANALYSIS_ENABLED", str(enabled).lower())
    if api_key and "…" not in api_key and len(api_key) > 8:
        lines = _set(lines, env_key_name, api_key)

    try:
        with open(env_path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write .env: {}", exc)
        return {"success": False, "message": f"Failed to write .env: {exc}"}

    label = {"openai": "OpenAI", "anthropic": "Anthropic (Claude)", "gemini": "Google Gemini"}
    return {
        "success": True,
        "message": (f"{label[provider]} saved"
                    + (f" — model: {model}" if model else "")
                    + (" — AI analysis ENABLED" if enabled else " — AI analysis disabled")),
    }


async def _realtime_loop() -> None:
    """Push metrics + enriched positions + tickers to all connected WS clients every 5 seconds."""
    import time as _time
    while True:
        await asyncio.sleep(5)
        if not state.connected_ws or not state.portfolio_manager:
            continue
        try:
            # Single price snapshot used for positions enrichment + signal tickers
            all_prices = await _fetch_all_prices()

            # Enrich open positions with live prices
            positions = dict(await state.portfolio_manager.get_open_positions())
            await _enrich_positions(positions, all_prices)
            total_upnl = round(sum(p.get("unrealized_pnl_usdt", 0) for p in positions.values()), 2)

            # Metrics — inject total unrealized PnL and risk exposure
            metrics  = await state.portfolio_manager.calculate_metrics()
            exposure = await state.portfolio_manager.get_risk_exposure()
            metrics["risk_exposure"]        = exposure
            metrics["total_unrealized_pnl"] = total_upnl
            await broadcast("metrics", metrics)

            await broadcast("positions", {"positions": positions, "exposure": exposure})

            # Broadcast live prices for recent signal symbols + open position symbols
            sig_syms = list({s.get("symbol") for s in state.recent_signals[-100:] if s.get("symbol")})
            pos_syms = list(positions.keys())
            all_relevant = list(set(sig_syms) | set(pos_syms))
            if all_relevant and all_prices:
                rel_prices = {sym: all_prices[sym] for sym in all_relevant if sym in all_prices}
                if rel_prices:
                    await broadcast("tickers", {"prices": rel_prices})

            # Broadcast pending LIMIT entry orders so dashboard can show countdown
            if state.pending_entry_orders:
                from config import settings as _s
                now = _time.time()
                pending_list = []
                for sym, p in state.pending_entry_orders.items():
                    elapsed   = now - p.placed_at
                    remaining = max(0.0, _s.limit_order_timeout_seconds - elapsed)
                    pending_list.append({
                        "symbol":      p.symbol,
                        "order_id":    p.order_id,
                        "direction":   p.risk_params.direction,
                        "limit_price": p.risk_params.entry_price,
                        "stop_loss":   p.risk_params.stop_loss,
                        "take_profit": p.risk_params.take_profit,
                        "retry_count": p.retry_count,
                        "max_retries": _s.limit_order_max_retries,
                        "elapsed_s":   round(elapsed, 1),
                        "remaining_s": round(remaining, 1),
                        "timeout_s":   _s.limit_order_timeout_seconds,
                    })
                await broadcast("pending_orders", {"orders": pending_list})
            else:
                await broadcast("pending_orders", {"orders": []})

            # Broadcast bot operational status so dashboard stats stay live
            await broadcast("bot_status", _make_bot_status_payload())

        except Exception as exc:
            logger.warning("Realtime broadcast error: {}", exc)

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_ws.append(websocket)
    logger.info("WS connected ({})", len(state.connected_ws))

    # Push current state immediately so client doesn't wait up to 5s for first update
    if state.portfolio_manager:
        try:
            all_prices_init = await _fetch_all_prices()
            metrics = await state.portfolio_manager.calculate_metrics()
            exposure_init = await state.portfolio_manager.get_risk_exposure()
            metrics["risk_exposure"] = exposure_init
            positions = dict(await state.portfolio_manager.get_open_positions())
            await _enrich_positions(positions, all_prices_init)
            metrics["total_unrealized_pnl"] = round(
                sum(p.get("unrealized_pnl_usdt", 0) for p in positions.values()), 2
            )
            await websocket.send_text(json.dumps({
                "type": "metrics", "data": metrics, "ts": datetime.utcnow().isoformat()
            }))
            exposure  = await state.portfolio_manager.get_risk_exposure()
            await websocket.send_text(json.dumps({
                "type": "positions",
                "data": {"positions": positions, "exposure": exposure},
                "ts": datetime.utcnow().isoformat(),
            }))
            # Push bot status immediately so panel shows live data on connect
            await websocket.send_text(json.dumps({
                "type": "bot_status",
                "data": _make_bot_status_payload(),
                "ts": datetime.utcnow().isoformat(),
            }))
        except Exception:
            pass

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=25)
                await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in state.connected_ws:
            state.connected_ws.remove(websocket)


@app.post("/api/reset-session")
async def reset_session():
    """Backup + wipe trade history, performance data, and in-memory runtime state.

    PRESERVED (not touched): ml_training_samples, coin_stats, research_log.
    CLEARED:
      - DB: closed/cancelled trades, performance_snapshots
      - Memory: watchlist, signal cooldowns, position_meta, pending LIMIT orders,
                signal feed, bot_stats counters, RiskManager daily/margin state
    """
    if not state.portfolio_manager:
        return {"ok": False, "error": "Bot portfolio not initialised"}

    backup_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backups")
    try:
        result = await state.portfolio_manager.reset_session(backup_dir)
        if state.trading_system:
            state.trading_system.reset_runtime_state()
        await broadcast("session_reset", {"backup": result["backup_file"]})
        logger.info("Dashboard: session reset → {}", result["backup_file"])
        return {"ok": True, **result}
    except Exception as exc:
        logger.exception("Session reset failed")
        return {"ok": False, "error": str(exc)}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    html_file = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_file, encoding="utf-8") as f:
        return f.read()
