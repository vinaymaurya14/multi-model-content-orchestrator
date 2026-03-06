"""Ollama local model provider using httpx."""

from __future__ import annotations

import logging
from typing import List

import httpx

from app.config import settings
from app.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3"

# Ollama is free (local), but we assign a nominal compute cost for comparison.
_NOMINAL_COST_PER_1K = 0.00001


class OllamaProvider(LLMProvider):
    """Ollama local API integration (requires Ollama running on the host)."""

    name = "ollama"

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = _DEFAULT_MODEL
        self._detected_models: list[str] = []

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        # We cannot do async I/O here, so just return True; the real check
        # happens in health_check().
        return True

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    self._detected_models = [
                        m["name"] for m in data.get("models", [])
                    ]
                    return True
                return False
        except Exception:
            logger.debug("Ollama is not reachable at %s", self.base_url)
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        model = kwargs.get("model", self.model)
        start = self._timer()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = self._timer() - start

        text = data.get("response", "")
        input_tokens = data.get("prompt_eval_count", self._estimate_token_count(prompt))
        output_tokens = data.get("eval_count", self._estimate_token_count(text))

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(elapsed, 2),
            cost_estimate=self.estimate_cost(input_tokens, output_tokens),
            finish_reason="stop",
            raw=data,
        )

    def get_models(self) -> List[dict]:
        models = self._detected_models or [_DEFAULT_MODEL]
        return [
            {
                "provider": self.name,
                "model_id": m,
                "display_name": f"Ollama {m}",
                "capabilities": ["text_generation"],
                "cost_per_1k_input_tokens": _NOMINAL_COST_PER_1K,
                "cost_per_1k_output_tokens": _NOMINAL_COST_PER_1K,
                "max_tokens": 8192,
                "quality_tier": "standard",
                "is_available": True,
            }
            for m in models
        ]

    def estimate_cost(self, input_tokens: int, output_tokens: int, **_) -> float:
        return ((input_tokens + output_tokens) / 1000) * _NOMINAL_COST_PER_1K
