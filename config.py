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

    # ── Capital allocation safety ──────────────────────────────────────────────
    # Fraction of total balance kept as a permanent anti-liquidation reserve.
    # This is the "50% always in the bank" concept:
    #   tradeable_balance = account_balance × (1 - balance_reserve_pct/100)
    # The risk manager only draws from tradeable_balance when sizing positions.
    # Example: balance=1000, reserve=50% → max deployable = 500 USDT;
    #   with 5 trades × 10% cap each = 500 USDT max margin, 0 USDT wasted.
    # Remaining 500 USDT covers: unrealized loss, funding fees, maintenance margin.
    balance_reserve_pct: float = 50.0

    # Position-size safety guards
    # Limits margin per single trade to this % of FREE tradeable balance.
    # Prevents over-allocation when SL is tight (tight SL → large position → huge margin).
    # Example: reserve=50%, balance=1000 → tradeable=500; cap=10% → max margin/trade=$50,
    #   max notional=150 USDT at 3×.  5 trades × $50 = $250 max total margin (25% of balance).
    max_position_size_pct: float = 10.0

    # Minimum SL distance from entry (%). Prevents ultra-tight SL that would create a
    # position too large to be safe.  0 = disabled (not recommended).
    min_stop_loss_pct: float = 0.5

    # Estimated taker fee per side (%).
    # Binance Futures = 0.04 %, Bybit Linear = 0.055 %; use 0.06 % as a conservative estimate
    # (adds ~0.02% for spread/slippage on altcoins).
    # Used to warn when round-trip fees eat a large fraction of the risk budget.
    taker_fee_pct: float = 0.06

    # Funding fee safety estimation.
    # Futures positions pay funding every 8 h.  These fields let the risk manager
    # budget for expected funding costs before sizing the position.
    #   max_funding_rate_pct = worst-case rate per 8 h period (0.10% = extreme bull market)
    #   funding_periods_estimate = number of 8 h periods to budget for
    #     (max_position_hold_hours=18 → ceil(18/8) = 3 periods)
    # The estimated funding cost is added to taker fees in the safety cap check.
    max_funding_rate_pct: float = 0.10    # % per 8-h period (pessimistic; normal ≈ 0.01%)
    funding_periods_estimate: int = 3     # periods to budget for (matches 18 h hold limit)

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
    max_signals_per_scan: int = 150        # cap on new signals processed per cycle (prevents 7-8 min loops)

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

    # Paper trading flag — also settable via --dry-run CLI arg (CLI takes precedence)
    dry_run: bool = False

    # ── Training Mode ─────────────────────────────────────────────────────────
    # Set TRAINING_MODE=true in .env to maximise trade frequency and collect ML data.
    # ALL thresholds are relaxed — intended for use with --dry-run.
    # Production rules are restored automatically when training_mode=False.
    training_mode: bool = False

    # ── Effective thresholds (auto-relaxed in training mode) ──────────────────
    @property
    def effective_signal_score_threshold(self) -> int:
        """Gate score after ML+sentiment boost. In training mode: accept score ≥ 5 (quality filter)."""
        return 5 if self.training_mode else self.signal_score_threshold

    @property
    def effective_watchlist_confirmations(self) -> int:
        """Scan cycles before deep research. In training mode: 1 cycle only."""
        return 1 if self.training_mode else self.watchlist_confirmations

    @property
    def effective_signal_cooldown_minutes(self) -> int:
        """Cooldown after watchlist decision. In training mode: 5 min."""
        return 5 if self.training_mode else self.signal_cooldown_minutes

    @property
    def effective_research_min_score(self) -> float:
        """Research score gate (0–10). In training mode: 4.0 (meaningful quality gate)."""
        return 4.0 if self.training_mode else self.research_min_score

    @property
    def effective_research_min_mtf_alignment(self) -> float:
        """Fraction of TFs that must agree. In training mode: 0.33 (at least 1 of 3 TFs)."""
        return 0.33 if self.training_mode else self.research_min_mtf_alignment

    @property
    def effective_min_ml_confidence(self) -> float:
        """Min ML confidence gate. In training mode: 0.0 (bypass — model not trained yet)."""
        return 0.0 if self.training_mode else self.min_ml_confidence

    @property
    def effective_max_daily_trades(self) -> int:
        """Max trades per day. In training mode: 20 (balanced data collection)."""
        return 20 if self.training_mode else self.max_daily_trades

    @property
    def effective_max_open_trades(self) -> int:
        """Max simultaneous open positions. In training mode: 8 (reduce concurrent risk)."""
        return 8 if self.training_mode else self.max_open_trades

    @property
    def effective_max_position_hold_hours(self) -> float:
        """Force-close after N hours. In training mode: 4h to cycle data faster."""
        return 4.0 if self.training_mode else self.max_position_hold_hours

    @property
    def effective_min_volume_usdt(self) -> float:
        """Minimum 24h volume filter. In training mode: 200k (more coins eligible)."""
        return 200_000.0 if self.training_mode else self.min_volume_usdt

    @property
    def effective_max_signals_per_scan(self) -> int:
        """Max NEW signals processed per scan cycle (watchlist coins always included).
        Training mode uses 50 to keep each scan cycle under ~60 seconds."""
        return 50 if self.training_mode else self.max_signals_per_scan


settings = Settings()
