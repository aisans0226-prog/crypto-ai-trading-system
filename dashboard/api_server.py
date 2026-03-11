import os
"""
dashboard/api_server.py - Professional responsive FastAPI dashboard v2.
No authentication required — open access on local/VPS network.
"""
import asyncio
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from config import settings

app = FastAPI(title="Crypto AI Dashboard", version="2.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class AppState:
    portfolio_manager = None
    market_scanner = None
    self_learner = None
    coin_ranker = None
    auto_updater = None
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


@app.get("/api/portfolio")
async def get_portfolio():
    if not state.portfolio_manager:
        return {"balance": settings.account_balance_usdt, "positions": {}, "risk_exposure": {}}
    balance = await state.portfolio_manager.get_balance()
    positions = await state.portfolio_manager.get_open_positions()
    exposure = await state.portfolio_manager.get_risk_exposure()
    return {"balance": balance, "positions": positions, "risk_exposure": exposure}


@app.get("/api/metrics")
async def get_metrics():
    if not state.portfolio_manager:
        return {}
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


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_ws.append(websocket)
    logger.info("WS connected ({})", len(state.connected_ws))
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
