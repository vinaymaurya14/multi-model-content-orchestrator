"""Model capabilities registry.

Maintains a catalogue of all known models across providers with their
capabilities, costs, token limits, and quality tiers.  Used by the router
and benchmarker to make informed decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A single model's metadata inside the registry."""
    provider: str
    model_id: str
    display_name: str
    capabilities: List[str]
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_tokens: int
    quality_tier: str  # "draft", "standard", "premium"
    is_available: bool = True

    @property
    def avg_cost_per_1k(self) -> float:
        return (self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens) / 2

    @property
    def quality_score(self) -> float:
        """Numeric quality score derived from the tier."""
        return {"draft": 0.5, "standard": 0.75, "premium": 1.0}.get(
            self.quality_tier, 0.6
        )


class ModelRegistry:
    """Central registry of all models from all providers."""

    def __init__(self) -> None:
        self._entries: Dict[str, ModelEntry] = {}  # key = "provider:model_id"
        self._providers: Dict[str, LLMProvider] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_provider(self, provider: LLMProvider) -> int:
        """Import all models from a provider.  Returns count of models added."""
        self._providers[provider.name] = provider
        count = 0
        for m in provider.get_models():
            key = f"{m['provider']}:{m['model_id']}"
            self._entries[key] = ModelEntry(
                provider=m["provider"],
                model_id=m["model_id"],
                display_name=m.get("display_name", m["model_id"]),
                capabilities=m.get("capabilities", []),
                cost_per_1k_input_tokens=m.get("cost_per_1k_input_tokens", 0),
                cost_per_1k_output_tokens=m.get("cost_per_1k_output_tokens", 0),
                max_tokens=m.get("max_tokens", 4096),
                quality_tier=m.get("quality_tier", "standard"),
                is_available=m.get("is_available", True),
            )
            count += 1
        logger.info("Registered %d models from provider '%s'", count, provider.name)
        return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all(self) -> List[ModelEntry]:
        return list(self._entries.values())

    def get_available(self) -> List[ModelEntry]:
        return [e for e in self._entries.values() if e.is_available]

    def get_by_provider(self, provider: str) -> List[ModelEntry]:
        return [e for e in self._entries.values() if e.provider == provider]

    def get_by_capability(self, capability: str) -> List[ModelEntry]:
        return [
            e for e in self._entries.values()
            if capability in e.capabilities and e.is_available
        ]

    def get_by_quality_tier(self, tier: str) -> List[ModelEntry]:
        return [
            e for e in self._entries.values()
            if e.quality_tier == tier and e.is_available
        ]

    def get_entry(self, provider: str, model_id: str) -> Optional[ModelEntry]:
        return self._entries.get(f"{provider}:{model_id}")

    def get_provider_instance(self, name: str) -> Optional[LLMProvider]:
        return self._providers.get(name)

    def available_provider_names(self) -> List[str]:
        """Return provider names that have at least one available model."""
        return list({e.provider for e in self._entries.values() if e.is_available})

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        available = self.get_available()
        return {
            "total_models": len(self._entries),
            "available_models": len(available),
            "providers": list({e.provider for e in available}),
            "quality_tiers": {
                tier: len([e for e in available if e.quality_tier == tier])
                for tier in ("draft", "standard", "premium")
            },
        }
