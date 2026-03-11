# Crypto AI Trading System

Production-ready crypto futures trading bot with AI-powered signal detection,
automatic order execution, risk management, and a real-time web dashboard.

---

## Feature Overview

| Module | Description |
|---|---|
| Market Scanner | Scans 500+ coins every 60 s for pump/whale/breakout signals |
| AI Prediction | XGBoost + LSTM ensemble, +3 score on bullish prediction ≥ 0.7 |
| Risk Manager | 1 % risk/trade, max 3× leverage, max 5 open trades, 1:2 min RR |
| Trade Executor | Binance Futures + Bybit Futures auto-execution with SL/TP |
| Portfolio Manager | PostgreSQL persistence, Redis caching, PnL tracking |
| Alert System | Telegram + Discord notifications |
| Dashboard | FastAPI + embedded React-style SPA at `http://host:8000` |
| Arbitrage Engine | Cross-exchange funding-rate and price-spread detection |

---

## Signal Scoring System

| Source | Max Points |
|---|---|
| Pump Detector (volume, RSI, EMA, OI, funding) | 6 |
| Whale Tracker (order-book + accumulation) | 4 |
| Memecoin Scanner (24h move, volume surge, breakout) | 3 |
| AI Prediction (XGBoost + LSTM ensemble) | +3 |
| Social Sentiment | +1 |

**Threshold: score ≥ 8 → HIGH PROBABILITY signal**

---

## Project Structure

```
crypto_ai_trading_system/
├── main.py                    # Entry point — starts all subsystems
├── config.py                  # Pydantic settings (loaded from .env)
├── requirements.txt
├── .env.example               # Copy to .env and fill in API keys
│
├── data_engine/
│   ├── market_data.py         # Async REST: OHLCV, OI, funding, order-book
│   └── websocket_feed.py      # Binance + Bybit WS feeds with auto-reconnect
│
├── scanners/
│   ├── market_scanner.py      # Orchestrates all sub-scanners
│   ├── pump_detector.py       # Volume spike / RSI / EMA / OI scoring
│   ├── whale_tracker.py       # Order-book imbalance + accumulation
│   └── memecoin_scanner.py    # Meme/narrative coin detection
│
├── social_ai/
│   └── sentiment_analyzer.py  # Twitter + Reddit + VADER sentiment
│
├── ml_models/
│   ├── prediction_model.py    # XGBoost + LSTM ensemble predictor
│   └── saved_models/          # Auto-created on first training run
│
├── strategy/
│   ├── trend_strategy.py      # EMA stack trend-following
│   ├── breakout_strategy.py   # Resistance/support breakout
│   └── liquidity_strategy.py  # Liquidity sweep / stop-hunt reversal
│
├── trading/
│   ├── risk_manager.py        # Position sizing + rule enforcement
│   ├── trade_executor.py      # Binance + Bybit order placement
│   └── arbitrage_engine.py    # Cross-exchange spread detection
│
├── portfolio/
│   └── portfolio_manager.py   # DB persistence, metrics, risk exposure
│
├── alerts/
│   └── telegram_bot.py        # Telegram + Discord alert sender
│
├── dashboard/
│   └── api_server.py          # FastAPI REST + WebSocket dashboard
│
└── utils/
    └── logger.py              # Loguru structured logging
```

---

## Requirements

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- (Optional) TA-Lib C library for full indicator support

---

## Installation — Local / VPS

### 1. Clone or copy the project

```bash
cd /opt
git clone <your-repo-url> crypto_bot
cd crypto_bot/crypto_ai_trading_system
```

### 2. Create Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate            # Linux / Mac
# venv\Scripts\activate             # Windows
```

### 3. Install TA-Lib C library (required before pip install)

**Ubuntu / Debian:**
```bash
sudo apt-get install build-essential wget
wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib && ./configure --prefix=/usr && make && sudo make install && cd ..
```

**Windows:**
Download the pre-built wheel from https://github.com/cgohlke/talib-build/releases

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Set up PostgreSQL

```bash
sudo -u postgres psql
CREATE USER trader WITH PASSWORD 'password';
CREATE DATABASE crypto_trading OWNER trader;
\q
```

### 6. Set up Redis

```bash
sudo apt-get install redis-server
sudo systemctl enable redis-server --now
```

### 7. Configure environment

```bash
cp .env.example .env
nano .env          # fill in your API keys
```

Required `.env` values:

```
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DATABASE_URL=postgresql+asyncpg://trader:password@localhost:5432/crypto_trading
REDIS_URL=redis://localhost:6379/0
```

### 8. Run in dry-run mode first (recommended)

```bash
python main.py --dry-run
```

Open dashboard: `http://localhost:8000`
Login: `admin` / `admin123` (change in `api_server.py` for production)

### 9. Run live trading

```bash
# Binance Futures
python main.py --exchange binance

# Bybit Futures
python main.py --exchange bybit
```

---

## Deployment — VPS / Cloud Server

### Systemd service (recommended for 24/7 operation)

```bash
sudo nano /etc/systemd/system/crypto-bot.service
```

```ini
[Unit]
Description=Crypto AI Trading System
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/crypto_bot/crypto_ai_trading_system
ExecStart=/opt/crypto_bot/crypto_ai_trading_system/venv/bin/python main.py --exchange binance
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable crypto-bot --now
sudo systemctl status crypto-bot
sudo journalctl -u crypto-bot -f    # tail logs
```

### Docker Compose (alternative)

```yaml
# docker-compose.yml
version: "3.9"
services:
  bot:
    build: .
    restart: always
    env_file: .env
    depends_on:
      - db
      - redis
    ports:
      - "8000:8000"

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: password
      POSTGRES_DB: crypto_trading
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

**Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential wget && \
    wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && cd .. && \
    rm -rf ta-lib*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py", "--exchange", "binance"]
```

```bash
docker-compose up -d
docker-compose logs -f bot
```

---

## Web Dashboard

| URL | Description |
|---|---|
| `http://host:8000/` | Live trading dashboard |
| `http://host:8000/docs` | FastAPI Swagger UI |
| `http://host:8000/api/portfolio` | Portfolio JSON |
| `http://host:8000/api/metrics` | Performance metrics |
| `http://host:8000/api/signals` | Recent signal list |
| `ws://host:8000/ws` | Live WebSocket stream |

---

## Risk Warning

This software is for **educational and research purposes**.
Futures trading involves substantial risk of loss.
Always test thoroughly in dry-run mode before using real capital.
Never risk capital you cannot afford to lose.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `RISK_PER_TRADE_PCT` | `1.0` | Percent of balance risked per trade |
| `MAX_LEVERAGE` | `3` | Maximum futures leverage |
| `MAX_OPEN_TRADES` | `5` | Maximum simultaneous open positions |
| `MIN_RISK_REWARD_RATIO` | `2.0` | Minimum RR before accepting trade |
| `SIGNAL_SCORE_THRESHOLD` | `8` | Minimum score to trigger signal |
| `SCAN_INTERVAL_SECONDS` | `60` | Seconds between full market scans |
| `MAX_COINS_TO_SCAN` | `500` | Number of coins to include in scan |
| `MIN_VOLUME_USDT` | `500000` | Minimum 24h volume filter |
