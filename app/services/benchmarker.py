"""Model comparison and benchmarking service.

Runs the same prompt across multiple providers, collects timing, token
usage, and quality scores, then returns structured benchmark results with
statistical summaries.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from app.core.metrics import MetricsCollector
from app.core.model_registry import ModelRegistry
from app.models.schemas import BenchmarkModelResult, BenchmarkResponse
from app.providers.base import LLMProvider, LLMResponse
from app.services.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


class Benchmarker:
    """Run benchmarks across available providers."""

    def __init__(
        self,
        registry: ModelRegistry,
        metrics: MetricsCollector,
        quality_scorer: QualityScorer,
    ) -> None:
        self.registry = registry
        self.metrics = metrics
        self.quality_scorer = quality_scorer

    async def run(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        num_runs: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> BenchmarkResponse:
        """Execute the benchmark and return results."""

        # Determine which providers to test
        available = self.registry.available_provider_names()
        if providers:
            target_providers = [p for p in providers if p in available]
        else:
            target_providers = available

        if not target_providers:
            target_providers = ["mock"]  # always have the mock

        results: List[BenchmarkModelResult] = []

        for pname in target_providers:
            provider = self.registry.get_provider_instance(pname)
            if provider is None:
                continue
            models = self.registry.get_by_provider(pname)
            if not models:
                continue
            # Use the first (best) model from each provider
            model_entry = models[0]

            latencies: List[float] = []
            token_counts: List[int] = []
            costs: List[float] = []
            quality_scores: List[float] = []
            sample_output = ""

            for run_idx in range(num_runs):
                try:
                    resp: LLMResponse = await provider.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model=model_entry.model_id,
                        content_type="general",
                    )
                    latencies.append(resp.latency_ms)
                    token_counts.append(resp.total_tokens)
                    costs.append(resp.cost_estimate)

                    qs = self.quality_scorer.score(resp.text, prompt)
                    quality_scores.append(qs.overall)

                    if run_idx == 0:
                        sample_output = resp.text[:500]

                    # Record to metrics
                    self.metrics.record_success(
                        provider=resp.provider,
                        model=resp.model,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cost=resp.cost_estimate,
                        latency_ms=resp.latency_ms,
                    )
                except Exception as exc:
                    logger.warning("Benchmark run %d failed for %s: %s", run_idx, pname, exc)
                    self.metrics.record_failure(pname, model_entry.model_id, 0)

            if not latencies:
                continue

            lat_arr = np.array(latencies)
            results.append(BenchmarkModelResult(
                provider=pname,
                model=model_entry.model_id,
                mean_latency_ms=round(float(np.mean(lat_arr)), 2),
                std_latency_ms=round(float(np.std(lat_arr)), 2),
                p50_latency_ms=round(float(np.percentile(lat_arr, 50)), 2),
                p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 2),
                mean_tokens=round(float(np.mean(token_counts)), 1),
                mean_cost=round(float(np.mean(costs)), 6),
                total_cost=round(float(np.sum(costs)), 6),
                mean_quality_score=round(float(np.mean(quality_scores)), 2) if quality_scores else 0.0,
                sample_output=sample_output,
                runs=len(latencies),
            ))

        # Determine winners
        fastest = min(results, key=lambda r: r.mean_latency_ms).provider if results else "N/A"
        cheapest = min(results, key=lambda r: r.mean_cost).provider if results else "N/A"
        highest_q = max(results, key=lambda r: r.mean_quality_score).provider if results else "N/A"

        return BenchmarkResponse(
            prompt=prompt,
            results=results,
            fastest_provider=fastest,
            cheapest_provider=cheapest,
            highest_quality_provider=highest_q,
            timestamp=datetime.now(timezone.utc),
        )
