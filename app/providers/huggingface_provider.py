"""HuggingFace Inference API provider using httpx."""

from __future__ import annotations

import logging
from typing import List

import httpx

from app.config import settings
from app.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_API_BASE = "https://api-inference.huggingface.co/models"

_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "display": "Mistral 7B Instruct v0.2",
        "cost_input": 0.0001,
        "cost_output": 0.0001,
        "max_tokens": 8192,
        "quality_tier": "standard",
    },
    "google/gemma-2b-it": {
        "display": "Gemma 2B IT",
        "cost_input": 0.00005,
        "cost_output": 0.00005,
        "max_tokens": 4096,
        "quality_tier": "draft",
    },
    "HuggingFaceH4/zephyr-7b-beta": {
        "display": "Zephyr 7B Beta",
        "cost_input": 0.0001,
        "cost_output": 0.0001,
        "max_tokens": 8192,
        "quality_tier": "standard",
    },
}

_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API integration via httpx."""

    name = "huggingface"

    def __init__(self) -> None:
        self.api_token = settings.hf_api_token
        self.model = _DEFAULT_MODEL

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self.api_token)

    async def health_check(self) -> bool:
        if not self.is_available():
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{_API_BASE}/{self.model}",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            logger.warning("HuggingFace health-check failed", exc_info=True)
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("HuggingFace provider is not available (no API token).")

        model = kwargs.get("model", self.model)
        start = self._timer()

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(max_tokens, 2048),
                "temperature": max(temperature, 0.01),
                "return_full_text": False,
            },
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_API_BASE}/{model}",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = self._timer() - start

        # HF returns a list of generated texts
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            text = data.get("generated_text", str(data))
        else:
            text = str(data)

        input_tokens = self._estimate_token_count(prompt)
        output_tokens = self._estimate_token_count(text)

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(elapsed, 2),
            cost_estimate=self.estimate_cost(input_tokens, output_tokens, model=model),
            finish_reason="stop",
            raw=data if isinstance(data, dict) else {"results": data},
        )

    def get_models(self) -> List[dict]:
        return [
            {
                "provider": self.name,
                "model_id": mid,
                "display_name": info["display"],
                "capabilities": ["text_generation"],
                "cost_per_1k_input_tokens": info["cost_input"],
                "cost_per_1k_output_tokens": info["cost_output"],
                "max_tokens": info["max_tokens"],
                "quality_tier": info["quality_tier"],
                "is_available": self.is_available(),
            }
            for mid, info in _MODELS.items()
        ]

    def estimate_cost(self, input_tokens: int, output_tokens: int, *, model: str | None = None) -> float:
        model = model or self.model
        info = _MODELS.get(model, _MODELS[_DEFAULT_MODEL])
        return (input_tokens / 1000) * info["cost_input"] + (output_tokens / 1000) * info["cost_output"]

    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
