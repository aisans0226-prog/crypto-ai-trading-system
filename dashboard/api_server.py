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
    asyncio.create_task(_realtime_loop())
    yield


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
    return {"balance": balance, "positions": positions, "risk_exposure": exposure}


@app.get("/api/metrics")
async def get_metrics():
    if not state.portfolio_manager:
        return {
            "total_trades": 0, "open_trades": 0, "wins_count": 0, "losses_count": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "daily_pnl": 0.0,
            "total_pnl": 0.0, "balance": settings.account_balance_usdt, "pnl_trend": [],
        }
    return await state.portfolio_manager.calculate_metrics()


@app.get("/api/advanced-stats")
async def get_advanced_stats():
    """Extended analytics: avg duration, best/worst trade, monthly PnL, PnL by symbol/weekday."""
    empty = {
        "avg_duration_hours": 0.0, "best_trade_pnl": 0.0, "worst_trade_pnl": 0.0,
        "avg_pnl_per_trade": 0.0, "avg_signal_score": 0.0,
        "current_win_streak": 0, "current_loss_streak": 0,
        "monthly_pnl": [], "pnl_by_weekday": [], "pnl_by_symbol": [],
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

    # --- PnL by symbol (top 15) ---
    sym_pnl: dict = defaultdict(float)
    sym_cnt: dict = defaultdict(int)
    for t in trades:
        sym_pnl[t.symbol] += t.pnl_usdt or 0
        sym_cnt[t.symbol] += 1
    pnl_by_sym = sorted(
        [{"symbol": s, "pnl": round(p, 2), "trades": sym_cnt[s]} for s, p in sym_pnl.items()],
        key=lambda x: x["pnl"], reverse=True
    )[:15]

    return {
        "avg_duration_hours":  avg_dur,
        "best_trade_pnl":      best,
        "worst_trade_pnl":     worst,
        "avg_pnl_per_trade":   avg_p,
        "avg_signal_score":    avg_sc,
        "current_win_streak":  win_streak,
        "current_loss_streak": loss_streak,
        "monthly_pnl":         monthly_list,
        "pnl_by_weekday":      pnl_by_wd,
        "pnl_by_symbol":       pnl_by_sym,
    }


@app.get("/api/bot-status")
async def get_bot_status():
    """Real-time bot operational status and configuration snapshot."""
    from config import settings as _s
    import time as _t
    started = state.bot_stats.get("started_at")
    uptime_s = round(_t.time() - started, 0) if started else 0
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
    }


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
            "id":           t.id,
            "symbol":       t.symbol,
            "direction":    t.direction,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "quantity":     t.quantity,
            "leverage":     t.leverage,
            "stop_loss":    t.stop_loss,
            "take_profit":  t.take_profit,
            "pnl_usdt":     t.pnl_usdt,
            "status":       t.status,
            "exchange":     t.exchange,
            "signal_score": t.signal_score,
            "strategy":     getattr(t, "strategy_name", None),
            "opened_at":    t.opened_at.isoformat() if t.opened_at else None,
            "closed_at":    t.closed_at.isoformat() if t.closed_at else None,
        }
        for t in trades
    ]

    closed = [r for r in rows if r["status"] == "closed"]
    wins   = [r for r in closed if r["pnl_usdt"] and r["pnl_usdt"] > 0]
    stats  = {
        "total":        len(rows),
        "open":         len([r for r in rows if r["status"] == "open"]),
        "closed":       len(closed),
        "wins":         len(wins),
        "losses":       len(closed) - len(wins),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl":    round(sum(r["pnl_usdt"] or 0 for r in closed), 2),
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    html_file = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_file, encoding="utf-8") as f:
        return f.read()
