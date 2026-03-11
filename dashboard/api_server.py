import os
"""
dashboard/api_server.py - Professional responsive FastAPI dashboard v2.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from jose import jwt
from passlib.context import CryptContext
from loguru import logger

from config import settings

app = FastAPI(title="Crypto AI Dashboard", version="2.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
ALGORITHM = "HS256"
DEMO_USERS = {"admin": pwd_context.hash("admin123")}


def create_token(username: str) -> str:
    exp = datetime.utcnow() + timedelta(minutes=settings.dashboard_jwt_expire_minutes)
    return jwt.encode({"sub": username, "exp": exp},
                      settings.dashboard_secret_key, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, settings.dashboard_secret_key, algorithms=[ALGORITHM])
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    hashed = DEMO_USERS.get(form.username)
    if not hashed or not pwd_context.verify(form.password, hashed):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": create_token(form.username), "token_type": "bearer"}


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
async def get_portfolio(user: str = Depends(get_current_user)):
    if not state.portfolio_manager:
        return {"balance": settings.account_balance_usdt, "positions": {}, "risk_exposure": {}}
    balance = await state.portfolio_manager.get_balance()
    positions = await state.portfolio_manager.get_open_positions()
    exposure = await state.portfolio_manager.get_risk_exposure()
    return {"balance": balance, "positions": positions, "risk_exposure": exposure}


@app.get("/api/metrics")
async def get_metrics(user: str = Depends(get_current_user)):
    if not state.portfolio_manager:
        return {}
    return await state.portfolio_manager.calculate_metrics()


@app.get("/api/signals")
async def get_signals(user: str = Depends(get_current_user)):
    return {"signals": state.recent_signals[-100:]}


@app.get("/api/heatmap")
async def get_heatmap(user: str = Depends(get_current_user)):
    coins = [
        {"symbol": s.get("symbol"), "score": s.get("score"),
         "direction": s.get("direction"), "price_change_pct": s.get("price_change_pct", 0),
         "volume_24h": s.get("volume_24h", 0), "ai_prediction": s.get("ai_prediction", 0.5)}
        for s in state.recent_signals[-150:]
    ]
    return {"coins": coins}


@app.get("/api/ai-rankings")
async def get_ai_rankings(user: str = Depends(get_current_user)):
    return {"rankings": state.coin_rankings[:50]}


@app.get("/api/learning-stats")
async def get_learning_stats(user: str = Depends(get_current_user)):
    if not state.self_learner:
        return {"stats": {}, "top_coins": []}
    stats = state.self_learner.get_stats()
    top_coins = state.self_learner.get_top_coins(10)
    return {
        "stats": stats,
        "top_coins": [{"symbol": s, "win_rate": round(w * 100, 1)} for s, w in top_coins],
    }


@app.get("/api/update/status")
async def update_status(user: str = Depends(get_current_user)):
    if not state.auto_updater:
        return {"update_available": False, "current_commit": "unknown"}
    return state.auto_updater.get_status()


@app.post("/api/update/check")
async def update_check(user: str = Depends(get_current_user)):
    if not state.auto_updater:
        return {"update_available": False}
    return await state.auto_updater.check_for_update()


@app.post("/api/update/apply")
async def update_apply(user: str = Depends(get_current_user)):
    if not state.auto_updater:
        raise HTTPException(status_code=503, detail="Updater not available")
    result = await state.auto_updater.apply_update()
    await broadcast("system_update", result)
    return result


@app.post("/api/update/rollback")
async def update_rollback(user: str = Depends(get_current_user)):
    if not state.auto_updater:
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
