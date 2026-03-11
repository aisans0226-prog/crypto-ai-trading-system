"""
dashboard/api_server.py — FastAPI web dashboard with real-time WebSocket push.

Endpoints:
  GET  /health
  GET  /api/portfolio
  GET  /api/signals
  GET  /api/trades
  GET  /api/metrics
  GET  /api/heatmap
  WS   /ws  — live updates stream
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from loguru import logger

from config import settings

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crypto AI Trading Dashboard",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
ALGORITHM = "HS256"

DEMO_USERS = {
    "admin": pwd_context.hash("admin123")   # change in production!
}


def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(
        minutes=settings.dashboard_jwt_expire_minutes
    )
    return jwt.encode(
        {"sub": username, "exp": expire},
        settings.dashboard_secret_key,
        algorithm=ALGORITHM,
    )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(
            token, settings.dashboard_secret_key, algorithms=[ALGORITHM]
        )
        return payload["sub"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


@app.post("/auth/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    hashed = DEMO_USERS.get(form.username)
    if not hashed or not pwd_context.verify(form.password, hashed):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": create_token(form.username), "token_type": "bearer"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared state (injected at startup from main.py)
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    portfolio_manager = None
    market_scanner = None
    recent_signals: List[dict] = []
    connected_ws: List[WebSocket] = []


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/portfolio")
async def get_portfolio(user: str = Depends(get_current_user)):
    if not state.portfolio_manager:
        return {"balance": settings.account_balance_usdt, "positions": {}}
    balance = await state.portfolio_manager.get_balance()
    positions = await state.portfolio_manager.get_open_positions()
    exposure = await state.portfolio_manager.get_risk_exposure()
    return {
        "balance": balance,
        "positions": positions,
        "risk_exposure": exposure,
    }


@app.get("/api/metrics")
async def get_metrics(user: str = Depends(get_current_user)):
    if not state.portfolio_manager:
        return {}
    return await state.portfolio_manager.calculate_metrics()


@app.get("/api/signals")
async def get_signals(user: str = Depends(get_current_user)):
    return {"signals": state.recent_signals[-50:]}


@app.get("/api/heatmap")
async def get_heatmap(user: str = Depends(get_current_user)):
    """Return 24h performance data for market heatmap."""
    # In production, populate from DB; here we return recent signals as proxy
    return {
        "coins": [
            {
                "symbol": s.get("symbol"),
                "score": s.get("score"),
                "direction": s.get("direction"),
                "price_change_pct": s.get("price_change_pct"),
            }
            for s in state.recent_signals[-100:]
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket live updates
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_ws.append(websocket)
    logger.info("WS client connected (total: {})", len(state.connected_ws))
    try:
        while True:
            # Keep alive + echo ping
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            await websocket.send_text(json.dumps({"type": "pong"}))
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        state.connected_ws.remove(websocket)
        logger.info("WS client disconnected")


async def broadcast(event: str, payload: dict) -> None:
    """Push event to all connected WebSocket clients."""
    message = json.dumps({"type": event, "data": payload})
    dead: List[WebSocket] = []
    for ws in state.connected_ws:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.connected_ws.remove(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML (embedded single-page)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Crypto AI Trading Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0e1a; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
  header { background: #141928; padding: 16px 24px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #1e2a40; }
  header h1 { font-size: 1.2rem; color: #00d4aa; }
  .badge { background: #00d4aa22; color: #00d4aa; border: 1px solid #00d4aa55; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; padding: 20px; }
  .card { background: #141928; border: 1px solid #1e2a40; border-radius: 10px; padding: 16px; }
  .card h2 { font-size: 0.85rem; color: #8899aa; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
  .value { font-size: 1.8rem; font-weight: 700; color: #e0e0e0; }
  .positive { color: #00d4aa; }
  .negative { color: #ff4466; }
  .signal-item { background: #0d1220; border-left: 3px solid #00d4aa; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.82rem; }
  .signal-item .score { float: right; font-weight: 700; color: #ffd700; }
  #log { background: #080c14; padding: 8px; border-radius: 6px; font-family: monospace; font-size: 0.75rem; height: 120px; overflow-y: auto; color: #66cc88; }
</style>
</head>
<body>
<header>
  <h1>⚡ Crypto AI Trading System</h1>
  <span class="badge" id="status">Connecting…</span>
</header>
<div class="grid">
  <div class="card">
    <h2>Balance</h2>
    <div class="value positive" id="balance">$—</div>
  </div>
  <div class="card">
    <h2>Daily PnL</h2>
    <div class="value" id="daily-pnl">$—</div>
  </div>
  <div class="card">
    <h2>Open Trades</h2>
    <div class="value" id="open-trades">—</div>
  </div>
  <div class="card">
    <h2>Win Rate</h2>
    <div class="value" id="win-rate">—</div>
  </div>
  <div class="card" style="grid-column: 1/-1;">
    <h2>Active Signals</h2>
    <div id="signals-list"><em style="color:#556">No signals yet</em></div>
  </div>
  <div class="card" style="grid-column: 1/-1;">
    <h2>Live Log</h2>
    <div id="log"></div>
  </div>
</div>
<script>
const token = localStorage.getItem('jwt') || prompt('Enter admin password (demo: admin123)');
async function login() {
  const fd = new FormData();
  fd.append('username', 'admin');
  fd.append('password', token || 'admin123');
  const r = await fetch('/auth/token', { method: 'POST', body: fd });
  if (r.ok) {
    const d = await r.json();
    localStorage.setItem('jwt', d.access_token);
    return d.access_token;
  }
  return null;
}
async function loadData(jwt) {
  const h = { Authorization: 'Bearer ' + jwt };
  const [port, met, sig] = await Promise.all([
    fetch('/api/portfolio', {headers: h}).then(r => r.json()),
    fetch('/api/metrics', {headers: h}).then(r => r.json()),
    fetch('/api/signals', {headers: h}).then(r => r.json()),
  ]);
  document.getElementById('balance').textContent = '$' + (port.balance || 0).toFixed(2);
  const dpnl = document.getElementById('daily-pnl');
  dpnl.textContent = '$' + (met.daily_pnl || 0).toFixed(2);
  dpnl.className = 'value ' + ((met.daily_pnl || 0) >= 0 ? 'positive' : 'negative');
  document.getElementById('open-trades').textContent = (port.risk_exposure || {}).open_trades || 0;
  document.getElementById('win-rate').textContent = (met.win_rate || 0).toFixed(1) + '%';
  const sl = document.getElementById('signals-list');
  const signals = (sig.signals || []).slice(-10).reverse();
  sl.innerHTML = signals.length ? signals.map(s =>
    `<div class="signal-item">${s.symbol} — ${s.direction} — ${(s.signals||[]).join(', ')} <span class="score">${s.score}</span></div>`
  ).join('') : '<em style="color:#556">No signals yet</em>';
}
function log(msg) {
  const el = document.getElementById('log');
  el.innerHTML += new Date().toLocaleTimeString() + ' ' + msg + '<br>';
  el.scrollTop = el.scrollHeight;
}
(async () => {
  const jwt = await login() || localStorage.getItem('jwt');
  if (!jwt) return;
  await loadData(jwt);
  setInterval(() => loadData(jwt), 10000);
  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => { document.getElementById('status').textContent = 'LIVE'; document.getElementById('status').style.background = '#00d4aa22'; };
  ws.onmessage = e => { const d = JSON.parse(e.data); log(JSON.stringify(d)); if (d.type === 'signal') loadData(jwt); };
  ws.onclose = () => document.getElementById('status').textContent = 'Disconnected';
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return DASHBOARD_HTML
