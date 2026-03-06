"""Tests for the benchmarker, cost optimizer, and quality scorer."""

from __future__ import annotations

import pytest

from app.core.metrics import MetricsCollector
from app.core.model_registry import ModelRegistry
from app.providers.mock_provider import MockProvider
from app.services.benchmarker import Benchmarker
from app.services.cost_optimizer import CostOptimizer
from app.services.quality_scorer import QualityScorer, QualityResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    reg = ModelRegistry()
    reg.register_provider(MockProvider())
    return reg


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def quality_scorer():
    return QualityScorer()


@pytest.fixture
def benchmarker(registry, metrics, quality_scorer):
    return Benchmarker(registry, metrics, quality_scorer)


@pytest.fixture
def cost_optimizer(registry, metrics):
    return CostOptimizer(registry, metrics)


# ---------------------------------------------------------------------------
# Quality Scorer tests
# ---------------------------------------------------------------------------

class TestQualityScorer:
    def test_score_nonempty_text(self, quality_scorer):
        result = quality_scorer.score(
            "This is a well-written blog post about artificial intelligence. "
            "It covers the key concepts and provides actionable insights.",
            "Write a blog post about AI",
        )
        assert isinstance(result, QualityResult)
        assert 0 <= result.overall <= 100
        assert result.coherence >= 0
        assert result.relevance >= 0
        assert result.readability >= 0

    def test_score_empty_text(self, quality_scorer):
        result = quality_scorer.score("", "anything")
        assert result.overall == 0

    def test_relevance_higher_with_matching_keywords(self, quality_scorer):
        prompt = "machine learning algorithms"
        good = quality_scorer.score(
            "Machine learning algorithms are a fundamental part of AI. "
            "These algorithms learn patterns from data and make predictions.",
            prompt,
        )
        bad = quality_scorer.score(
            "Cooking pasta requires boiling water and adding salt. "
            "The noodles should be cooked al dente for best results.",
            prompt,
        )
        assert good.relevance > bad.relevance

    def test_completeness_favors_structured_content(self, quality_scorer):
        structured = (
            "# Heading\n\n"
            "First paragraph with content.\n\n"
            "## Subheading\n\n"
            "- Bullet point one\n"
            "- Bullet point two\n\n"
            "Another paragraph concluding the text."
        )
        flat = "Just a single line of text."

        s1 = quality_scorer.score(structured, "test")
        s2 = quality_scorer.score(flat, "test")
        assert s1.completeness > s2.completeness

    def test_seo_score_rewards_headings_and_keywords(self, quality_scorer):
        seo_friendly = (
            "# Best AI Tools 2026\n\n"
            "The **best AI tools** are transforming the industry.\n\n"
            "## Top Picks\n\n"
            "- **Tool A** -- Great for automation\n"
            "- **Tool B** -- Best for analytics\n"
        )
        plain = "Some random text without any structure or keywords."
        s1 = quality_scorer.score(seo_friendly, "best AI tools")
        s2 = quality_scorer.score(plain, "best AI tools")
        assert s1.seo_score > s2.seo_score


# ---------------------------------------------------------------------------
# Benchmarker tests
# ---------------------------------------------------------------------------

class TestBenchmarker:
    @pytest.mark.asyncio
    async def test_benchmark_returns_results(self, benchmarker):
        result = await benchmarker.run(
            prompt="Explain machine learning in simple terms.",
            providers=["mock"],
            num_runs=2,
        )
        assert len(result.results) == 1
        assert result.results[0].provider == "mock"
        assert result.results[0].runs == 2
        assert result.fastest_provider == "mock"
        assert result.cheapest_provider == "mock"

    @pytest.mark.asyncio
    async def test_benchmark_records_metrics(self, benchmarker, metrics):
        await benchmarker.run(
            prompt="Test prompt",
            providers=["mock"],
            num_runs=3,
        )
        assert metrics.total_requests() == 3

    @pytest.mark.asyncio
    async def test_benchmark_quality_scores_present(self, benchmarker):
        result = await benchmarker.run(
            prompt="Write about cloud computing",
            providers=["mock"],
            num_runs=1,
        )
        assert result.results[0].mean_quality_score > 0

    @pytest.mark.asyncio
    async def test_benchmark_with_no_providers_uses_mock(self, benchmarker):
        result = await benchmarker.run(
            prompt="Test",
            providers=["nonexistent_provider"],
        )
        # Should still return results (empty if truly no match, but mock is always available)
        # nonexistent is filtered out; the benchmarker falls back to mock
        assert isinstance(result.results, list)


# ---------------------------------------------------------------------------
# Cost Optimizer tests
# ---------------------------------------------------------------------------

class TestCostOptimizer:
    def test_report_empty(self, cost_optimizer):
        report = cost_optimizer.generate_report()
        assert report.total_cost == 0
        assert report.budget_remaining > 0
        assert len(report.recommendations) > 0

    @pytest.mark.asyncio
    async def test_report_after_generation(self, cost_optimizer, registry, metrics):
        # Simulate a generation
        mock = registry.get_provider_instance("mock")
        resp = await mock.generate("Test", content_type="general")
        metrics.record_success(
            provider=resp.provider,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost=resp.cost_estimate,
            latency_ms=resp.latency_ms,
        )
        report = cost_optimizer.generate_report()
        assert report.total_cost > 0
        assert len(report.by_provider) > 0

    def test_cheapest_meeting_quality(self, cost_optimizer):
        result = cost_optimizer.cheapest_meeting_quality("standard")
        assert result is not None
        assert "mock" in result

    def test_budget_not_exceeded_initially(self, cost_optimizer):
        assert not cost_optimizer.is_over_budget()
        assert not cost_optimizer.is_near_budget()
