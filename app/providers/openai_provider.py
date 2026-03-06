"""OpenAI GPT provider using httpx for async HTTP calls."""

from __future__ import annotations

import logging
from typing import List

import httpx

from app.config import settings
from app.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Cost per 1K tokens (approximate, GPT-4o pricing as of 2024)
_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

_DEFAULT_MODEL = "gpt-4o-mini"
_API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider):
    """Real OpenAI API integration via httpx."""

    name = "openai"

    def __init__(self) -> None:
        self.api_key = settings.openai_api_key
        self.org_id = settings.openai_org_id
        self.model = _DEFAULT_MODEL

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def health_check(self) -> bool:
        if not self.is_available():
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{_API_BASE}/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            logger.warning("OpenAI health-check failed", exc_info=True)
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI provider is not available (no API key).")

        model = kwargs.get("model", self.model)
        start = self._timer()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_API_BASE}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = self._timer() - start

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", self._estimate_token_count(prompt))
        output_tokens = usage.get("completion_tokens", 0)
        text = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "stop")

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
                "display_name": mid.upper().replace("-", " "),
                "capabilities": ["text_generation", "summarization", "translation", "code"],
                "cost_per_1k_input_tokens": p["input"],
                "cost_per_1k_output_tokens": p["output"],
                "max_tokens": 128000 if "4o" in mid else 16384,
                "quality_tier": "premium" if mid == "gpt-4o" else "standard",
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
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.org_id:
            h["OpenAI-Organization"] = self.org_id
        return h
