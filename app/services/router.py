"""Intelligent model routing engine.

Takes content type, quality requirements, cost constraints, and latency
preferences.  Scores every available provider across these dimensions and
returns the optimal one.  Falls back to the mock provider if no real
providers are available.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.metrics import MetricsCollector
from app.core.model_registry import ModelEntry, ModelRegistry
from app.core.routing_strategy import ScoredCandidate, rank_candidates
from app.models.schemas import ContentType, QualityLevel, RoutingStrategy
from app.providers.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Map QualityLevel to minimum quality tier needed
_QUALITY_TIER_MIN = {
    QualityLevel.draft: "draft",
    QualityLevel.standard: "standard",
    QualityLevel.premium: "premium",
}

_TIER_ORDER = {"draft": 0, "standard": 1, "premium": 2}


class RoutingEngine:
    """Selects the best provider + model for a given request."""

    def __init__(self, registry: ModelRegistry, metrics: MetricsCollector) -> None:
        self.registry = registry
        self.metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_and_generate(
        self,
        prompt: str,
        content_type: ContentType = ContentType.general,
        quality_level: QualityLevel = QualityLevel.standard,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        strategy: RoutingStrategy = RoutingStrategy.balanced,
        preferred_provider: Optional[str] = None,
    ) -> LLMResponse:
        """Select the optimal provider, generate content, and record metrics."""

        # 1. Pick provider + model
        provider, entry = self._select(
            content_type=content_type,
            quality_level=quality_level,
            strategy=strategy,
            preferred_provider=preferred_provider,
        )

        logger.info(
            "Routing to %s / %s  (strategy=%s, quality=%s)",
            provider.name, entry.model_id, strategy.value, quality_level.value,
        )

        # 2. Generate
        try:
            response = await provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                content_type=content_type.value,
                model=entry.model_id,
            )
        except Exception as exc:
            self.metrics.record_failure(provider.name, entry.model_id, 0)
            raise RuntimeError(f"Generation failed on {provider.name}: {exc}") from exc

        # 3. Record metrics
        self.metrics.record_success(
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost_estimate,
            latency_ms=response.latency_ms,
            content_type=content_type.value,
        )

        return response

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def _select(
        self,
        content_type: ContentType,
        quality_level: QualityLevel,
        strategy: RoutingStrategy,
        preferred_provider: Optional[str] = None,
    ) -> tuple[LLMProvider, ModelEntry]:
        """Return (provider_instance, model_entry) for the best candidate."""

        candidates = self.registry.get_available()

        # If a specific provider was requested, filter to it
        if preferred_provider:
            filtered = [c for c in candidates if c.provider == preferred_provider]
            if filtered:
                candidates = filtered
            else:
                logger.warning(
                    "Preferred provider '%s' has no available models; using all.",
                    preferred_provider,
                )

        # Filter by minimum quality tier
        min_tier = _QUALITY_TIER_MIN.get(quality_level, "standard")
        tier_filtered = [
            c for c in candidates
            if _TIER_ORDER.get(c.quality_tier, 0) >= _TIER_ORDER.get(min_tier, 0)
        ]
        if tier_filtered:
            candidates = tier_filtered

        # If still nothing, fallback to everything (will hit mock)
        if not candidates:
            candidates = self.registry.get_available()

        # Rank using the chosen strategy
        latency_estimates = self.metrics.avg_latency_by_model() or None
        ranked = rank_candidates(candidates, strategy.value, latency_estimates)

        if not ranked:
            # Ultimate fallback: grab mock
            mock = self.registry.get_provider_instance("mock")
            if mock is None:
                raise RuntimeError("No providers available and mock provider not registered.")
            mock_models = self.registry.get_by_provider("mock")
            return mock, mock_models[0]

        best: ScoredCandidate = ranked[0]
        provider = self.registry.get_provider_instance(best.entry.provider)
        if provider is None:
            raise RuntimeError(f"Provider instance '{best.entry.provider}' not found in registry.")

        return provider, best.entry
