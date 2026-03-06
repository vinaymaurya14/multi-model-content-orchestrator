"""Anthropic Claude provider using httpx for async HTTP calls."""

from __future__ import annotations

import logging
from typing import List

import httpx

from app.config import settings
from app.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
}

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_API_BASE = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Real Anthropic Claude API integration via httpx."""

    name = "anthropic"

    def __init__(self) -> None:
        self.api_key = settings.anthropic_api_key
        self.model = _DEFAULT_MODEL

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def health_check(self) -> bool:
        if not self.is_available():
            return False
        try:
            # A lightweight ping: send a tiny request
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{_API_BASE}/messages",
                    headers=self._headers(),
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                return resp.status_code in (200, 201)
        except Exception:
            logger.warning("Anthropic health-check failed", exc_info=True)
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic provider is not available (no API key).")

        model = kwargs.get("model", self.model)
        start = self._timer()

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_API_BASE}/messages",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = self._timer() - start

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", self._estimate_token_count(prompt))
        output_tokens = usage.get("output_tokens", 0)

        # Anthropic returns content as a list of blocks
        text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        text = "\n".join(text_blocks) if text_blocks else ""
        finish_reason = data.get("stop_reason", "end_turn")

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(elapsed, 2),
            cost_estimate=self.estimate_cost(input_tokens, output_tokens, model=model),
            finish_reason=finish_reason,
            raw=data,
        )

    def get_models(self) -> List[dict]:
        return [
            {
                "provider": self.name,
                "model_id": mid,
                "display_name": mid.replace("-", " ").title(),
                "capabilities": ["text_generation", "summarization", "analysis", "code"],
                "cost_per_1k_input_tokens": p["input"],
                "cost_per_1k_output_tokens": p["output"],
                "max_tokens": 200000 if "opus" in mid else 200000,
                "quality_tier": "premium" if "opus" in mid or "sonnet" in mid else "standard",
                "is_available": self.is_available(),
            }
            for mid, p in _PRICING.items()
        ]

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, model: str | None = None) -> float:
        model = model or self.model
        pricing = _PRICING.get(model, _PRICING[_DEFAULT_MODEL])
        return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }
