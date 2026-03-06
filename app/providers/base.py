"""Abstract base class for all LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMResponse:
    """Standardised response returned by every provider."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_estimate: float
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface that every LLM provider must implement."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from the model."""
        ...

    @abstractmethod
    def get_models(self) -> List[dict]:
        """Return list of models offered by this provider."""
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for the given token counts."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True when the provider is reachable and operational."""
        ...

    def is_available(self) -> bool:
        """Quick synchronous availability check (e.g. API key present)."""
        return True

    # Utility ------------------------------------------------------------------

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return max(1, len(text) // 4)

    @staticmethod
    def _timer() -> float:
        """Return current monotonic time in milliseconds."""
        return time.monotonic() * 1000
