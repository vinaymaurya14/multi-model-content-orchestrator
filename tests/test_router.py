"""Tests for the routing engine and routing strategies."""

from __future__ import annotations

import pytest
import asyncio

from app.core.metrics import MetricsCollector
from app.core.model_registry import ModelRegistry
from app.core.routing_strategy import rank_candidates, available_strategies
from app.models.schemas import ContentType, QualityLevel, RoutingStrategy
from app.providers.mock_provider import MockProvider
from app.services.router import RoutingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    reg = ModelRegistry()
    mock = MockProvider()
    reg.register_provider(mock)
    return reg


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def engine(registry, metrics):
    return RoutingEngine(registry, metrics)


# ---------------------------------------------------------------------------
# Routing strategy tests
# ---------------------------------------------------------------------------

class TestRoutingStrategy:
    def test_available_strategies(self):
        strategies = available_strategies()
        assert "balanced" in strategies
        assert "cost_optimized" in strategies
        assert "quality_optimized" in strategies
        assert "latency_optimized" in strategies
        assert len(strategies) == 4

    def test_rank_candidates_returns_sorted(self, registry):
        candidates = registry.get_available()
        ranked = rank_candidates(candidates, "balanced")
        assert len(ranked) > 0
        # Scores should be in descending order
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_quality_optimized_prefers_premium(self, registry):
        candidates = registry.get_available()
        ranked = rank_candidates(candidates, "quality_optimized")
        # The top candidate should have a premium quality tier
        top = ranked[0]
        assert top.entry.quality_tier == "premium"

    def test_cost_optimized_prefers_cheap(self, registry):
        candidates = registry.get_available()
        ranked = rank_candidates(candidates, "cost_optimized")
        top = ranked[0]
        # The cheapest mock model should rank first
        cheapest = min(candidates, key=lambda c: c.avg_cost_per_1k)
        assert top.entry.avg_cost_per_1k <= cheapest.avg_cost_per_1k + 0.001

    def test_rank_empty_candidates(self):
        ranked = rank_candidates([], "balanced")
        assert ranked == []

    def test_scored_candidate_has_breakdown(self, registry):
        candidates = registry.get_available()
        ranked = rank_candidates(candidates, "balanced")
        for sc in ranked:
            assert "quality" in sc.breakdown
            assert "cost" in sc.breakdown
            assert "latency" in sc.breakdown
            assert "capacity" in sc.breakdown


# ---------------------------------------------------------------------------
# Routing engine tests
# ---------------------------------------------------------------------------

class TestRoutingEngine:
    @pytest.mark.asyncio
    async def test_route_and_generate_returns_response(self, engine):
        resp = await engine.route_and_generate(
            prompt="Write a blog post about AI",
            content_type=ContentType.blog_post,
            quality_level=QualityLevel.standard,
            max_tokens=256,
        )
        assert resp.text
        assert resp.provider == "mock"
        assert resp.total_tokens > 0
        assert resp.latency_ms > 0

    @pytest.mark.asyncio
    async def test_route_respects_preferred_provider(self, engine):
        resp = await engine.route_and_generate(
            prompt="Hello world",
            preferred_provider="mock",
        )
        assert resp.provider == "mock"

    @pytest.mark.asyncio
    async def test_metrics_recorded_after_generate(self, engine, metrics):
        await engine.route_and_generate(prompt="Test prompt")
        assert metrics.total_requests() == 1
        assert metrics.total_cost() > 0

    @pytest.mark.asyncio
    async def test_different_strategies_produce_results(self, engine):
        for strat in RoutingStrategy:
            resp = await engine.route_and_generate(
                prompt="Explain quantum computing",
                strategy=strat,
            )
            assert resp.text

    @pytest.mark.asyncio
    async def test_content_types_produce_varied_content(self, engine):
        outputs = {}
        for ct in [ContentType.blog_post, ContentType.marketing_copy, ContentType.technical_doc]:
            resp = await engine.route_and_generate(
                prompt="AI content orchestration",
                content_type=ct,
            )
            outputs[ct] = resp.text

        # Different content types should produce different outputs
        texts = list(outputs.values())
        assert len(set(texts)) > 1, "Expected different outputs for different content types"
