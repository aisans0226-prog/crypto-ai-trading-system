"""
ai_engine/llm_analyzer.py — LLM-powered trade signal analysis.

Supports OpenAI (GPT), Anthropic (Claude), and Google Gemini.
Called from research_engine.py ONLY when ai_analysis_enabled=True.

Config is injected via update_cfg() from api_server.py whenever the user
saves settings through the dashboard — no bot restart required.

DISABLED BY DEFAULT. Enable only after reviewing bot performance statistics
on the Self Learning / Performance pages.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from loguru import logger


# ---------------------------------------------------------------------------
# Shared runtime config — updated by api_server.save_ai_config() at runtime
# ---------------------------------------------------------------------------
_active_cfg: dict = {
    "provider":           "",
    "model":              "",
    "enabled":            False,
    "openai_api_key":     "",
    "anthropic_api_key":  "",
    "gemini_api_key":     "",
}


def update_cfg(cfg: dict) -> None:
    """
    Push a new AI config dict into the module-level store.
    Called from api_server whenever the user saves via the dashboard.
    Takes effect immediately — no restart needed.
    """
    _active_cfg.update(cfg)


def is_enabled() -> bool:
    """Quick helper used by research_engine to skip the import entirely."""
    return bool(_active_cfg.get("enabled") and _active_cfg.get("provider"))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class LLMAnalysis:
    enabled:        bool          # False = LLM was skipped
    score_delta:    float         # −1.0 … +1.0 applied to research score
    risk_level:     str           # low | medium | high
    recommendation: str           # strong_entry | entry | wait | avoid
    reasoning:      str           # 1-2 sentence explanation
    provider:       str = ""      # which provider answered
    raw_response:   str = ""      # raw text for debugging


# Returned whenever LLM is disabled or an error occurs — zero net effect
_FALLBACK = LLMAnalysis(
    enabled=False,
    score_delta=0.0,
    risk_level="medium",
    recommendation="entry",
    reasoning="AI analysis unavailable",
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _build_prompt(
    symbol: str,
    direction: str,
    signal_score: int,
    research_score: float,
    mtf_alignment: float,
    tf_tags: Dict[str, List[str]],
) -> str:
    tf_lines = "\n".join(
        f"  {tf}: {', '.join(tags) if tags else 'no data'}"
        for tf, tags in tf_tags.items()
    )
    return (
        "You are a crypto futures trading analyst. "
        "Evaluate the signal below and reply with a single JSON object — no markdown, no extra text.\n\n"
        f"Symbol:          {symbol}\n"
        f"Direction:       {direction}\n"
        f"Signal Score:    {signal_score}/18\n"
        f"Research Score:  {research_score:.1f}/10\n"
        f"MTF Alignment:   {mtf_alignment*100:.0f}% ({int(round(mtf_alignment*3))}/3 timeframes agree)\n"
        f"Timeframe tags:\n{tf_lines}\n\n"
        'Reply ONLY with this JSON:\n'
        '{"score_delta": <float -1.0 to 1.0>, '
        '"risk_level": "low|medium|high", '
        '"recommendation": "strong_entry|entry|wait|avoid", '
        '"reasoning": "<1-2 sentences>"}'
    )


def _parse_llm_json(text: str) -> dict:
    """Extract JSON from response even when wrapped in markdown code fences."""
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Provider-specific callers
# ---------------------------------------------------------------------------
async def _call_openai(session, api_key: str, model: str, prompt: str) -> str:
    import aiohttp
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 220,
              "temperature": 0.2},
        timeout=aiohttp.ClientTimeout(total=18),
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(data.get("error", {}).get("message", "OpenAI error"))
        return data["choices"][0]["message"]["content"]


async def _call_anthropic(session, api_key: str, model: str, prompt: str) -> str:
    import aiohttp
    async with session.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key,
                 "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json={"model": model,
              "max_tokens": 220,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=aiohttp.ClientTimeout(total=18),
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(data.get("error", {}).get("message", "Anthropic error"))
        return data["content"][0]["text"]


async def _call_gemini(session, api_key: str, model: str, prompt: str) -> str:
    import aiohttp
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:generateContent?key={api_key}"
    )
    async with session.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"maxOutputTokens": 220, "temperature": 0.2}},
        timeout=aiohttp.ClientTimeout(total=18),
    ) as resp:
        data = await resp.json()
        if resp.status != 200:
            raise RuntimeError(data.get("error", {}).get("message", "Gemini error"))
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------
class LLMAnalyzer:
    """
    Call an external LLM to enhance trade research scoring.

    Usage (inside research_engine.py):

        if llm_analyzer.is_enabled():
            analysis = await LLMAnalyzer().analyze(...)
            score = max(0, min(10, score + analysis.score_delta))
    """

    async def analyze(
        self,
        symbol: str,
        direction: str,
        signal_score: int,
        research_score: float,
        mtf_alignment: float,
        tf_tags: Dict[str, List[str]],
    ) -> LLMAnalysis:
        if not _active_cfg.get("enabled"):
            return _FALLBACK

        provider = _active_cfg.get("provider", "").lower()
        model    = _active_cfg.get("model", "")
        api_key  = _active_cfg.get(f"{provider}_api_key", "")

        if not provider or not api_key:
            logger.debug("LLM skipped — provider/key not set")
            return _FALLBACK

        prompt = _build_prompt(
            symbol, direction, signal_score, research_score, mtf_alignment, tf_tags
        )

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                if provider == "openai":
                    raw = await _call_openai(session, api_key, model or "gpt-4o-mini", prompt)
                elif provider == "anthropic":
                    raw = await _call_anthropic(
                        session, api_key, model or "claude-3-5-haiku-20241022", prompt
                    )
                elif provider == "gemini":
                    raw = await _call_gemini(
                        session, api_key, model or "gemini-1.5-flash", prompt
                    )
                else:
                    logger.warning("LLMAnalyzer: unknown provider '{}'", provider)
                    return _FALLBACK

            parsed = _parse_llm_json(raw)
            delta  = float(parsed.get("score_delta", 0.0))
            delta  = max(-1.0, min(1.0, delta))           # hard clamp

            result = LLMAnalysis(
                enabled=True,
                score_delta=delta,
                risk_level=parsed.get("risk_level", "medium"),
                recommendation=parsed.get("recommendation", "entry"),
                reasoning=str(parsed.get("reasoning", raw[:200])),
                provider=provider,
                raw_response=raw,
            )
            logger.info(
                "LLM {} | {} {} | score_delta={:+.2f} risk={} rec={}",
                provider.upper(), symbol, direction,
                result.score_delta, result.risk_level, result.recommendation,
            )
            return result

        except asyncio.TimeoutError:
            logger.warning("LLM timeout for {} (provider={})", symbol, provider)
            return _FALLBACK
        except Exception as exc:
            logger.warning("LLM error for {}: {}", symbol, exc)
            return _FALLBACK
