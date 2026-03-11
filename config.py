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
    risk_per_trade_pct: float = 1.0
    max_leverage: int = 3
    max_open_trades: int = 5
    min_risk_reward_ratio: float = 2.0

    # AI Signal
    signal_score_threshold: int = 8

    # Scanner
    scan_interval_seconds: int = 60
    max_coins_to_scan: int = 500
    min_volume_usdt: float = 500_000.0

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


settings = Settings()
