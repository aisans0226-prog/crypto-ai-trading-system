#!/usr/bin/env bash
# =============================================================================
# manage.sh — Bot management commands on VPS
# Usage: ./manage.sh [start|stop|restart|status|logs|update|dry-run]
# =============================================================================
SERVICE="crypto-ai-bot"
DEPLOY_DIR="/root/Super Trade Coin/crypto_ai_trading_system"
VENV="$DEPLOY_DIR/venv/bin/python"

case "${1:-help}" in
    start)
        systemctl start $SERVICE
        echo "✅ Bot started"
        systemctl status $SERVICE --no-pager
        ;;
    stop)
        systemctl stop $SERVICE
        echo "⏹  Bot stopped"
        ;;
    restart)
        systemctl restart $SERVICE
        echo "🔄 Bot restarted"
        ;;
    status)
        systemctl status $SERVICE --no-pager
        ;;
    logs)
        journalctl -u $SERVICE -n 100 -f
        ;;
    update)
        echo "📦 Pulling latest code..."
        cd "$DEPLOY_DIR"
        git pull origin master
        source venv/bin/activate
        pip install -r requirements.txt -q
        systemctl restart $SERVICE
        echo "✅ Updated and restarted"
        ;;
    dry-run)
        echo "🧪 Starting in dry-run mode (paper trading)..."
        cd "$DEPLOY_DIR"
        $VENV main.py --dry-run
        ;;
    dashboard)
        echo "📊 Starting dashboard only..."
        cd "$DEPLOY_DIR"
        $VENV main.py --dashboard-only
        ;;
    help|*)
        echo "Usage: $0 [start|stop|restart|status|logs|update|dry-run|dashboard]"
        ;;
esac
