"""
alerts/telegram_bot.py — Real-time trade alerts via Telegram and Discord.
"""
import asyncio
from typing import Optional
import aiohttp
from loguru import logger

from config import settings
from scanners.market_scanner import SignalResult


TELEGRAM_API = "https://api.telegram.org"


class AlertSystem:
    """Sends HIGH PROBABILITY TRADE alerts to Telegram and/or Discord."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        logger.info("AlertSystem started")

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ── Public entrypoint ─────────────────────────────────────────────────
    async def send_trade_signal(self, signal: SignalResult) -> None:
        message = self._format_signal(signal)
        await asyncio.gather(
            self._send_telegram(message),
            self._send_discord(message),
            return_exceptions=True,
        )

    async def send_text(self, text: str) -> None:
        await asyncio.gather(
            self._send_telegram(text),
            self._send_discord(text),
            return_exceptions=True,
        )

    # ── Message formatting ────────────────────────────────────────────────
    @staticmethod
    def _format_signal(s: SignalResult) -> str:
        signals_list = "\n".join(f"  ✅ {sig.replace('_', ' ')}" for sig in s.signals)
        ai_line = f"🤖 AI Prediction: BULLISH +3\n" if s.ai_prediction >= 0.7 else ""
        return (
            f"🚨 *HIGH PROBABILITY TRADE*\n\n"
            f"Coin: `{s.symbol}`\n"
            f"Score: *{s.score}/13*\n\n"
            f"📊 *Signals:*\n{signals_list}\n\n"
            f"{ai_line}"
            f"📈 Direction: *{s.direction}*\n"
            f"💰 Price: `{s.price}`\n"
            f"📉 24h Change: `{s.price_change_pct:+.2f}%`\n"
            f"💧 24h Volume: `${s.volume_24h:,.0f}`\n"
        )

    # ── Telegram ──────────────────────────────────────────────────────────
    async def _send_telegram(self, text: str) -> None:
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return
        url = f"{TELEGRAM_API}/bot{settings.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": settings.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            async with self._session.post(url, json=payload) as r:
                if r.status != 200:
                    body = await r.text()
                    logger.warning("Telegram send failed {}: {}", r.status, body)
        except Exception as exc:
            logger.error("Telegram error: {}", exc)

    # ── Discord ───────────────────────────────────────────────────────────
    async def _send_discord(self, text: str) -> None:
        if not settings.discord_webhook_url:
            return
        # Convert Markdown bold to Discord-compatible
        clean = text.replace("*", "**").replace("`", "`")
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
