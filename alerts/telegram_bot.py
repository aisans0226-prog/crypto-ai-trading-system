"""
alerts/telegram_bot.py — Real-time trade alerts via Telegram and Discord.
"""
import asyncio
import html
import time
from collections import deque
from typing import Optional, Callable, Awaitable
import aiohttp
from loguru import logger

from config import settings
from scanners.market_scanner import SignalResult

# Hard cap: never send more than this many messages per 60-second window.
# Telegram ban threshold is ~30/min; we stay well below it.
_MAX_MSGS_PER_MINUTE = 20


TELEGRAM_API = "https://api.telegram.org"

# Type alias for async command handler: receives command string, returns None
CommandHandler = Callable[[str], Awaitable[None]]


class AlertSystem:
    """Sends HIGH PROBABILITY TRADE alerts to Telegram and/or Discord."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._command_handler: Optional[CommandHandler] = None
        self._poll_offset: int = 0          # Telegram getUpdates offset
        self._poll_task: Optional[asyncio.Task] = None
        self._muted: bool = False           # suppress all outgoing messages when True
        self._tg_rate_limit_until: float = 0.0  # Unix timestamp when rate limit expires
        # Sliding-window rate governor: tracks timestamps of recent sends
        self._msg_times: deque = deque()

    @property
    def is_muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        """Silence all Telegram/Discord outgoing messages. Trading continues normally."""
        self._muted = True
        logger.info("AlertSystem muted — alerts suppressed, trading active")

    def unmute(self) -> None:
        """Re-enable Telegram/Discord outgoing messages."""
        self._muted = False
        logger.info("AlertSystem unmuted — alerts restored")

    def set_command_handler(self, handler: CommandHandler) -> None:
        """Register async callback for incoming Telegram bot commands."""
        self._command_handler = handler

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        # Start Telegram command polling if bot token is configured
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self._poll_task = asyncio.create_task(self._poll_telegram_commands())
        logger.info("AlertSystem started")

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
        if self._session:
            await self._session.close()

    # ── Telegram command polling ───────────────────────────────────────────
    async def _poll_telegram_commands(self) -> None:
        """Long-poll Telegram getUpdates to receive bot commands."""
        logger.info("Telegram command listener started")
        while True:
            try:
                url = f"{TELEGRAM_API}/bot{settings.telegram_bot_token}/getUpdates"
                params = {
                    "offset": self._poll_offset,
                    "timeout": 30,
                    "allowed_updates": ["message"],
                }
                async with self._session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=40),
                ) as r:
                    if r.status != 200:
                        await asyncio.sleep(5)
                        continue
                    data = await r.json()

                for update in data.get("result", []):
                    self._poll_offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = (msg.get("text") or "").strip()
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Security: only accept commands from the configured chat
                    if chat_id != str(settings.telegram_chat_id):
                        continue

                    if text.startswith("/") and self._command_handler:
                        cmd = text.split()[0].lstrip("/").lower()
                        try:
                            await self._command_handler(cmd)
                        except Exception as exc:
                            logger.error("Command handler error: {}", exc)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Telegram poll error: {}", exc)
                await asyncio.sleep(5)

    # ── Public entrypoint ─────────────────────────────────────────────────
    async def send_trade_signal(self, signal: SignalResult) -> None:
        message = self._format_signal(signal)
        await asyncio.gather(
            self._send_telegram(message),
            self._send_discord(message),
            return_exceptions=True,
        )

    async def send_watchlist_alert(self, signal: SignalResult, confirmations_needed: int) -> None:
        """Stage 1: signal detected, added to watchlist for monitoring."""
        # Training mode floods hundreds of watchlist adds per cycle — skip to avoid Telegram 429 ban.
        if settings.training_mode:
            return
        msg = (
            f"🔍 <b>SIGNAL DETECTED — Monitoring</b>\n\n"
            f"Coin: <code>{self._esc(signal.symbol)}</code>\n"
            f"Direction: <b>{self._esc(signal.direction)}</b>\n"
            f"Score: <b>{signal.score}</b>\n"
            f"Price: <code>{signal.price}</code>\n"
            f"24h Change: <code>{signal.price_change_pct:+.2f}%</code>\n\n"
            f"⏳ Watching for <code>{confirmations_needed}</code> more scan cycle(s) "
            f"before deep research..."
        )
        await asyncio.gather(
            self._send_telegram(msg),
            self._send_discord(msg),
            return_exceptions=True,
        )

    async def send_research_result(self, signal: SignalResult, result) -> None:
        """Stage 2: deep research completed — passed or failed."""
        # In training mode suppress failed-research noise; only passed (entry about to happen) matters.
        if settings.training_mode and not result.passed:
            return
        if result.passed:
            ok_lines = "\n".join(f"  ✅ {self._esc(r)}" for r in result.reasons[:6])
            header = "📊 <b>RESEARCH PASSED</b> ✅ — Entering trade"
        else:
            ok_lines = "\n".join(f"  ❌ {self._esc(r)}" for r in result.failed_reasons[:6])
            header = "❌ <b>RESEARCH FAILED</b> — Skipping trade"

        msg = (
            f"{header}\n\n"
            f"Coin: <code>{self._esc(signal.symbol)}</code> | <b>{self._esc(signal.direction)}</b>\n"
            f"Signal Score: <b>{signal.score}</b>\n"
            f"Research Score: <b>{result.score:.1f}/10</b> | "
            f"Confidence: <b>{result.confidence*100:.0f}%</b>\n"
            f"MTF Alignment: <b>{result.mtf_alignment*100:.0f}%</b>\n\n"
            f"{'Reasons:' if result.passed else 'Issues:'}\n{ok_lines}"
        )
        await asyncio.gather(
            self._send_telegram(msg),
            self._send_discord(msg),
            return_exceptions=True,
        )

    async def send_trade_entry(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        rr: float,
        signal_score: int,
        research_score: float,
        daily_remaining: int,
    ) -> None:
        """Stage 3: trade is being executed — full entry details."""
        sl_pct = abs(entry - sl) / entry * 100
        tp_pct = abs(tp - entry) / entry * 100
        msg = (
            f"⚡ <b>ENTERING TRADE</b>\n\n"
            f"Coin: <code>{self._esc(symbol)}</code> | <b>{self._esc(direction)}</b>\n"
            f"Entry: <code>${entry:,.4f}</code>\n"
            f"Stop Loss: <code>${sl:,.4f}</code> ({sl_pct:.2f}%)\n"
            f"Take Profit: <code>${tp:,.4f}</code> ({tp_pct:.2f}%)\n"
            f"Risk/Reward: <b>1:{rr:.1f}</b>\n\n"
            f"Signal Score: <b>{signal_score}</b> | Research: <b>{research_score:.1f}/10</b>\n"
            f"Trades left today: <b>{daily_remaining}</b>"
        )
        await asyncio.gather(
            self._send_telegram(msg),
            self._send_discord(msg),
            return_exceptions=True,
        )

    async def send_text(self, text: str) -> None:
        await asyncio.gather(
            self._send_telegram(text),
            self._send_discord(text),
            return_exceptions=True,
        )

    async def send_critical_error(self, title: str, detail: str) -> None:
        """Send a critical error alert to Telegram/Discord with a distinctive format."""
        msg = (
            f"🔴 <b>CRITICAL ERROR — {self._esc(title)}</b>\n\n"
            f"<code>{self._esc(detail[:400])}</code>\n\n"
            f"⚠️ Bot may require manual intervention."
        )
        await self.send_text(msg)

    @staticmethod
    def _esc(s) -> str:
        """HTML-escape a value so it's safe inside Telegram HTML messages."""
        return html.escape(str(s))

    @staticmethod
    def _format_signal(s: SignalResult) -> str:
        esc = html.escape
        signals_list = "\n".join(f"  ✅ {esc(sig.replace('_', ' '))}" for sig in s.signals)
        ai_line = "🤖 AI Prediction: BULLISH +3\n" if s.ai_prediction >= 0.7 else ""
        return (
            f"🚨 <b>HIGH PROBABILITY TRADE</b>\n\n"
            f"Coin: <code>{esc(s.symbol)}</code>\n"
            f"Score: <b>{s.score}/13</b>\n\n"
            f"📊 <b>Signals:</b>\n{signals_list}\n\n"
            f"{ai_line}"
            f"📈 Direction: <b>{esc(s.direction)}</b>\n"
            f"💰 Price: <code>{s.price}</code>\n"
            f"📉 24h Change: <code>{s.price_change_pct:+.2f}%</code>\n"
            f"💧 24h Volume: <code>${s.volume_24h:,.0f}</code>\n"
        )

    # ── Telegram ──────────────────────────────────────────────────────────
    async def _send_telegram(self, text: str) -> None:
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return
        if self._muted:
            return  # notifications silenced — trading continues normally
        # Back-off when rate-limited (429)
        if time.time() < self._tg_rate_limit_until:
            remaining = int(self._tg_rate_limit_until - time.time())
            if remaining % 300 == 0:   # log reminder every 5 min
                logger.warning("Telegram rate-limited — {} s remaining", remaining)
            return
        url = f"{TELEGRAM_API}/bot{settings.telegram_bot_token}/sendMessage"
        # Sliding-window rate governor: drop message if we're already at the cap.
        now = time.time()
        while self._msg_times and now - self._msg_times[0] > 60:
            self._msg_times.popleft()
        if len(self._msg_times) >= _MAX_MSGS_PER_MINUTE:
            logger.debug("Telegram rate governor: {} msgs/60s cap reached — message dropped", _MAX_MSGS_PER_MINUTE)
            return
        self._msg_times.append(now)
        payload = {
            "chat_id": settings.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            async with self._session.post(url, json=payload) as r:
                if r.status == 429:
                    body = await r.json()
                    retry_after = body.get("parameters", {}).get("retry_after", 60)
                    self._tg_rate_limit_until = time.time() + retry_after
                    logger.warning(
                        "Telegram 429 rate-limit hit — pausing alerts for {:.0f} min",
                        retry_after / 60,
                    )
                elif r.status != 200:
                    body = await r.text()
                    logger.warning("Telegram send failed {}: {}", r.status, body)
        except Exception as exc:
            logger.error("Telegram error: {}", exc)

    # ── Discord ───────────────────────────────────────────────────────────
    async def _send_discord(self, text: str) -> None:
        if not settings.discord_webhook_url:
            return
        if self._muted:
            return  # notifications silenced — trading continues normally
        # Convert HTML tags to Discord Markdown equivalents
        clean = (
            text.replace("<b>", "**").replace("</b>", "**")
                .replace("<code>", "`").replace("</code>", "`")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        )
        payload = {"content": clean}
        try:
            async with self._session.post(
                settings.discord_webhook_url, json=payload
            ) as r:
                if r.status not in (200, 204):
                    body = await r.text()
                    logger.warning("Discord send failed {}: {}", r.status, body)
        except Exception as exc:
            logger.error("Discord error: {}", exc)
