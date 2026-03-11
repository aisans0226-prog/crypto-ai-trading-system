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
    recent_signals: List[dict] = []
    coin_rankings: List[dict] = []
    connected_ws: List[WebSocket] = []


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


async def _fetch_current_prices(symbols: list) -> dict:
    """Fetch current mark prices for symbols from Binance Futures (no auth required)."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    sym_set = set(symbols)
                    return {item["symbol"]: float(item["price"])
                            for item in data if item["symbol"] in sym_set}
    except Exception:
        pass
    return {}


async def _enrich_positions(positions: Dict[str, dict]) -> None:
    """Add current_price / unrealized_pnl_usdt / unrealized_pnl_pct in-place."""
    if not positions:
        return
    prices = await _fetch_current_prices(list(positions.keys()))
    for sym, pos in positions.items():
        cp = prices.get(sym)
        if not cp:
            continue
        qty   = pos.get("quantity", 0)
        entry = pos.get("entry_price", 0)
        size  = pos.get("position_size_usdt") or (qty * entry) or 1
        upnl  = (cp - entry) * qty if pos.get("direction", "LONG") == "LONG" else (entry - cp) * qty
        pos["current_price"]        = round(cp, 6)
        pos["unrealized_pnl_usdt"]  = round(upnl, 2)
        pos["unrealized_pnl_pct"]   = round(upnl / size * 100, 2) if size else 0


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
            "total_trades": 0, "open_trades": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0, "daily_pnl": 0.0,
            "total_pnl": 0.0, "balance": settings.account_balance_usdt, "pnl_trend": [],
        }
    return await state.portfolio_manager.calculate_metrics()


@app.get("/api/signals")
async def get_signals():
    return {"signals": state.recent_signals[-100:]}


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
    top_coins = state.self_learner.get_top_coins(10)
    return {
        "stats": stats,
        "top_coins": [{"symbol": s, "win_rate": round(w * 100, 1)} for s, w in top_coins],
    }


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
async def get_coin_stats(limit: int = 50):
    """Top coins ranked by win rate from the self-learning database."""
    if not state.coin_database:
        return {"coins": [], "total": 0}
    coins = await state.coin_database.get_top_coins(limit)
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
    """Push metrics + enriched positions to all connected WS clients every 5 seconds."""
    while True:
        await asyncio.sleep(5)
        if not state.connected_ws or not state.portfolio_manager:
            continue
        try:
            metrics = await state.portfolio_manager.calculate_metrics()
            await broadcast("metrics", metrics)

            positions = dict(await state.portfolio_manager.get_open_positions())
            await _enrich_positions(positions)
            exposure  = await state.portfolio_manager.get_risk_exposure()
            await broadcast("positions", {"positions": positions, "exposure": exposure})
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
            metrics = await state.portfolio_manager.calculate_metrics()
            await websocket.send_text(json.dumps({
                "type": "metrics", "data": metrics, "ts": datetime.utcnow().isoformat()
            }))
            positions = dict(await state.portfolio_manager.get_open_positions())
            await _enrich_positions(positions)
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
