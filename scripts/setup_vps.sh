#!/usr/bin/env bash
# =============================================================================
# setup_vps.sh — Full VPS setup for Crypto AI Trading Bot
# Run this ONCE on a fresh Ubuntu/Debian VPS as root.
# =============================================================================
set -euo pipefail

DEPLOY_DIR="/root/Super Trade Coin/crypto_ai_trading_system"
REPO_URL="https://github.com/aisans0226-prog/crypto-ai-trading-system.git"
BRANCH="master"
SERVICE_NAME="crypto-ai-bot"
PYTHON_MIN="3.11"

echo "============================================================"
echo "  Crypto AI Trading Bot — VPS Setup"
echo "  Target: $DEPLOY_DIR"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    git build-essential libssl-dev \
    postgresql postgresql-contrib \
    redis-server \
    curl wget nano screen

# ── 2. Create deployment directory (SAFE: won't touch other bots) ────────────
echo "[2/8] Creating deployment directory..."
mkdir -p "/root/Super Trade Coin"
cd "/root/Super Trade Coin"

if [ -d "crypto_ai_trading_system" ]; then
    echo "  [!] Directory exists — pulling latest code..."
    cd crypto_ai_trading_system
    git fetch origin $BRANCH
    git reset --hard origin/$BRANCH
else
    echo "  Cloning repository..."
    git clone -b $BRANCH "$REPO_URL" crypto_ai_trading_system
    cd crypto_ai_trading_system
fi

# ── 3. Python virtual environment ────────────────────────────────────────────
echo "[3/8] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# ── 4. Install Python dependencies ────────────────────────────────────────────
echo "[4/8] Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q

# ── 5. PostgreSQL setup ───────────────────────────────────────────────────────
echo "[5/8] Configuring PostgreSQL..."
systemctl enable postgresql
systemctl start postgresql
# Create DB and user (idempotent)
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='trader'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER trader WITH PASSWORD 'trader_pass';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='trading_db'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE trading_db OWNER trader;"
echo "  PostgreSQL: trading_db / trader"

# ── 6. Redis ──────────────────────────────────────────────────────────────────
echo "[6/8] Starting Redis..."
systemctl enable redis-server
systemctl start redis-server

# ── 7. Environment file ───────────────────────────────────────────────────────
echo "[7/8] Setting up .env configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    echo "  │  IMPORTANT: Edit /root/Super Trade Coin/crypto_ai_trading_  │"
    echo "  │  system/.env with your API keys before starting the bot!    │"
    echo "  └─────────────────────────────────────────────────────────────┘"
    echo ""
else
    echo "  .env already exists — skipping."
fi

# ── 8. Systemd service ────────────────────────────────────────────────────────
echo "[8/8] Installing systemd service..."
VENV_PYTHON="$DEPLOY_DIR/venv/bin/python"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Crypto AI Trading Bot — Super Trade Coin
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=root
WorkingDirectory=$DEPLOY_DIR
ExecStart=$VENV_PYTHON $DEPLOY_DIR/main.py --exchange binance
Restart=on-failure
RestartSec=15
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable $SERVICE_NAME
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1) Edit /root/Super Trade Coin/crypto_ai_trading_system/.env"
echo "     with your Binance/Bybit API keys, Telegram token, etc."
echo "  2) Start the bot:"
echo "     systemctl start $SERVICE_NAME"
echo "  3) Check logs:"
echo "     journalctl -u $SERVICE_NAME -f"
echo "  4) Dashboard: http://<VPS_IP>:8080"
echo "============================================================"
