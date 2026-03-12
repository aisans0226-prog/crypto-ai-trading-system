"""
config.py - Centralised configuration loaded from .env
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Exchange
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = False

    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = False

    # Proxy (optional — use if exchange API is geo-blocked)
    http_proxy: str = ""   # e.g. http://user:pass@proxy-host:port

    # Database
    database_url: str = "postgresql+asyncpg://trader:password@localhost:5432/crypto_trading"
    database_sync_url: str = "postgresql://trader:password@localhost:5432/crypto_trading"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_password: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_webhook_url: str = ""

    # Account / Risk
    account_balance_usdt: float = 1000.0
    risk_per_trade_pct: float = 1.0        # 1% risk per trade
    max_leverage: int = 3
    max_open_trades: int = 5               # max 5 simultaneous positions
    min_risk_reward_ratio: float = 2.0
    max_daily_trades: int = 3              # max trades bot can open per day
    max_daily_loss_pct: float = 5.0        # halt trading if daily realized loss >= 5% of balance

    # ── Smart position exit controls ──────────────────────────────────────────
    # Entry order type: MARKET fills instantly; LIMIT waits for a better price
    entry_order_type: str = "MARKET"          # MARKET | LIMIT
    limit_entry_offset_pct: float = 0.05      # place LIMIT this % inside market (LONG=below, SHORT=above)
    limit_order_timeout_seconds: int = 60     # cancel LIMIT entry if not filled within N seconds
    limit_order_max_retries: int = 3          # max re-place attempts before giving up on entry

    # Entry guard: abort if live price has drifted this far from signal price
    entry_max_deviation_pct: float = 0.8

    # Stop-loss cap: strategy SL is capped at this % distance from entry
    max_stop_loss_pct: float = 2.5         # SL never further than 2.5% from entry

    # Trailing stop: slides SL upward (LONG) / downward (SHORT) as price moves in favour
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 3.0   # % profit required to activate
    trailing_stop_distance_pct: float = 1.0      # keep SL this % behind the peak price
    trailing_stop_min_move_pct: float = 0.3      # only update exchange SL if new SL improves by ≥ this %

    # Breakeven stop: move SL to entry + 0.1% buffer once profit reaches breakeven_trigger_pct
    breakeven_stop_enabled: bool = True
    breakeven_trigger_pct: float = 2.0     # activate after +2% unrealised profit

    # Time-based exit: force-close position after N hours regardless of SL/TP (0 = disabled)
    max_position_hold_hours: float = 18.0

    # Reversal exit: exit early if position is losing AND has been open long enough
    reversal_exit_enabled: bool = True
    reversal_exit_pct: float = 2.0         # % loss from entry that triggers early exit
    reversal_exit_min_hours: float = 2.0   # minimum hours open before reversal exit fires

    # AI Signal — stricter quality gates
    signal_score_threshold: int = 11       # raised from 8 → requires strong scanner score
    min_ml_confidence: float = 0.60        # min ML confidence to confirm entry (when trained)
    signal_cooldown_minutes: int = 30      # cool-off after watchlist decision (trade or skip)

    # Watchlist / Research phase (confirm signal before researching)
    watchlist_confirmations: int = 2       # scan cycles to monitor before deep research
    research_min_score: float = 6.0        # deep research score gate (out of 10)
    research_min_mtf_alignment: float = 0.67  # 2/3 timeframes must agree

    # Scanner
    scan_interval_seconds: int = 60
    max_coins_to_scan: int = 500
    min_volume_usdt: float = 1_000_000.0   # raised from 500k → higher quality coins

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    dashboard_secret_key: str = "change_this_to_a_random_secret"
    dashboard_jwt_expire_minutes: int = 1440

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading.log"

    # Social
    twitter_bearer_token: str = ""
    huggingface_token: str = ""

    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "crypto_scanner/1.0"

    # AI LLM Integration (OpenAI / Anthropic / Gemini)
    ai_provider: str = ""              # openai | anthropic | gemini
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    ai_model: str = ""                 # e.g. gpt-4o, claude-3-5-sonnet-20241022, gemini-1.5-pro
    ai_analysis_enabled: bool = False  # enable LLM-enhanced trade research

    # Auto Strategy Discovery
    strategy_discovery_enabled: bool = True  # if False, falls back to first-match (legacy)


settings = Settings()
