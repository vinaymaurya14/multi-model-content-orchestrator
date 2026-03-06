"""FastAPI application entry point.

Initialises providers, registry, services, and exposes all API endpoints.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.metrics import MetricsCollector
from app.core.model_registry import ModelRegistry
from app.models.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    CompareRequest,
    CompareResponse,
    ComparisonItem,
    CostReport,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    QualityLevel,
    TaskStatus,
)
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.huggingface_provider import HuggingFaceProvider
from app.providers.mock_provider import MockProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider
from app.services.benchmarker import Benchmarker
from app.services.cost_optimizer import CostOptimizer
from app.services.quality_scorer import QualityScorer
from app.services.router import RoutingEngine
from app.services.task_queue import TaskQueue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (initialised during lifespan)
# ---------------------------------------------------------------------------
registry = ModelRegistry()
metrics = MetricsCollector()
quality_scorer = QualityScorer()
task_queue = TaskQueue()

# Services (set during startup)
routing_engine: RoutingEngine | None = None
benchmarker: Benchmarker | None = None
cost_optimizer: CostOptimizer | None = None

_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global routing_engine, benchmarker, cost_optimizer, _start_time
    _start_time = time.time()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Register providers ---
    # Always register mock
    mock = MockProvider()
    registry.register_provider(mock)

    # Register real providers if keys are present
    openai_prov = OpenAIProvider()
    if openai_prov.is_available():
        registry.register_provider(openai_prov)
        logger.info("OpenAI provider registered.")

    anthropic_prov = AnthropicProvider()
    if anthropic_prov.is_available():
        registry.register_provider(anthropic_prov)
        logger.info("Anthropic provider registered.")

    hf_prov = HuggingFaceProvider()
    if hf_prov.is_available():
        registry.register_provider(hf_prov)
        logger.info("HuggingFace provider registered.")

    ollama_prov = OllamaProvider()
    if await ollama_prov.health_check():
        registry.register_provider(ollama_prov)
        logger.info("Ollama provider registered (local).")

    logger.info("Registry summary: %s", registry.summary())

    # --- Initialise services ---
    routing_engine = RoutingEngine(registry, metrics)
    benchmarker = Benchmarker(registry, metrics, quality_scorer)
    cost_optimizer = CostOptimizer(registry, metrics)

    logger.info(
        "%s v%s started (demo_mode=%s, default_provider=%s)",
        settings.app_name, settings.app_version,
        settings.demo_mode, settings.default_provider,
    )

    yield  # Application is running

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Intelligent multi-model content orchestrator that routes requests "
        "to the optimal LLM provider based on content type, quality "
        "requirements, cost constraints, and latency targets."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        demo_mode=settings.demo_mode,
        available_providers=registry.available_provider_names(),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/models", response_model=List[ModelInfo], tags=["models"])
async def list_models():
    """List all known models with their capabilities and availability."""
    entries = registry.get_all()
    return [
        ModelInfo(
            provider=e.provider,
            model_id=e.model_id,
            display_name=e.display_name,
            capabilities=e.capabilities,
            cost_per_1k_input_tokens=e.cost_per_1k_input_tokens,
            cost_per_1k_output_tokens=e.cost_per_1k_output_tokens,
            max_tokens=e.max_tokens,
            quality_tier=QualityLevel(e.quality_tier) if e.quality_tier in ("draft", "standard", "premium") else QualityLevel.standard,
            is_available=e.is_available,
        )
        for e in entries
    ]


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate(req: GenerateRequest):
    """Generate content via the optimally routed model.

    The router evaluates available providers against the requested content
    type, quality level, and strategy to pick the best model.  If no real
    providers are configured the mock provider handles the request.
    """
    if routing_engine is None:
        raise HTTPException(503, "Service not yet initialised.")

    if cost_optimizer and cost_optimizer.is_over_budget():
        raise HTTPException(
            429,
            "Monthly cost budget exhausted. Increase MONTHLY_COST_LIMIT or switch to demo mode.",
        )

    try:
        resp = await routing_engine.route_and_generate(
            prompt=req.prompt,
            content_type=req.content_type,
            quality_level=req.quality_level,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            strategy=req.strategy,
            preferred_provider=req.preferred_provider,
        )
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}")

    # Score quality
    qs = quality_scorer.score(resp.text, req.prompt)

    return GenerateResponse(
        content=resp.text,
        provider_used=resp.provider,
        model_used=resp.model,
        tokens_used=resp.total_tokens,
        cost_estimate=resp.cost_estimate,
        quality_score=qs.overall,
        latency_ms=resp.latency_ms,
        content_type=req.content_type,
    )


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["benchmarking"])
async def benchmark(req: BenchmarkRequest):
    """Benchmark a prompt across multiple providers.

    Returns statistical summaries (mean, std, percentiles) of latency,
    cost, and quality for each provider.
    """
    if benchmarker is None:
        raise HTTPException(503, "Service not yet initialised.")

    try:
        result = await benchmarker.run(
            prompt=req.prompt,
            providers=req.providers,
            num_runs=req.num_runs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except Exception as exc:
        raise HTTPException(500, f"Benchmark failed: {exc}")

    return result


@app.post("/compare", response_model=CompareResponse, tags=["comparison"])
async def compare(req: CompareRequest):
    """Side-by-side comparison of model outputs for the same prompt."""
    available = registry.available_provider_names()
    target = [p for p in (req.providers or available) if p in available] or ["mock"]

    comparisons: List[ComparisonItem] = []

    for pname in target:
        provider = registry.get_provider_instance(pname)
        if provider is None:
            continue
        models = registry.get_by_provider(pname)
        if not models:
            continue
        model_entry = models[0]

        try:
            resp = await provider.generate(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                model=model_entry.model_id,
                content_type="general",
            )
            qs = quality_scorer.score(resp.text, req.prompt)

            metrics.record_success(
                provider=resp.provider,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost=resp.cost_estimate,
                latency_ms=resp.latency_ms,
            )

            comparisons.append(ComparisonItem(
                provider=resp.provider,
                model=resp.model,
                content=resp.text,
                tokens_used=resp.total_tokens,
                cost_estimate=resp.cost_estimate,
                quality_score=qs.overall,
                latency_ms=resp.latency_ms,
                quality_breakdown=qs.to_dict(),
            ))
        except Exception as exc:
            logger.warning("Compare failed for %s: %s", pname, exc)

    if not comparisons:
        raise HTTPException(500, "All providers failed during comparison.")

    # Recommend the best overall (quality / cost ratio)
    best = max(comparisons, key=lambda c: c.quality_score / max(c.cost_estimate, 0.0001))
    reason = (
        f"{best.provider} achieved the highest quality-to-cost ratio "
        f"(quality={best.quality_score:.1f}, cost=${best.cost_estimate:.6f})."
    )

    return CompareResponse(
        prompt=req.prompt,
        comparisons=comparisons,
        recommended_provider=best.provider,
        recommendation_reason=reason,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/costs/report", response_model=CostReport, tags=["costs"])
async def cost_report():
    """Return a detailed cost analysis report."""
    if cost_optimizer is None:
        raise HTTPException(503, "Service not yet initialised.")
    return cost_optimizer.generate_report()


# ---------------------------------------------------------------------------
# Task queue endpoints (async tasks)
# ---------------------------------------------------------------------------

@app.post("/tasks/generate", response_model=TaskStatus, tags=["tasks"])
async def submit_generate_task(req: GenerateRequest):
    """Submit a generation request for asynchronous processing."""
    if routing_engine is None:
        raise HTTPException(503, "Service not yet initialised.")

    async def _work():
        resp = await routing_engine.route_and_generate(
            prompt=req.prompt,
            content_type=req.content_type,
            quality_level=req.quality_level,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            strategy=req.strategy,
            preferred_provider=req.preferred_provider,
        )
        qs = quality_scorer.score(resp.text, req.prompt)
        return GenerateResponse(
            content=resp.text,
            provider_used=resp.provider,
            model_used=resp.model,
            tokens_used=resp.total_tokens,
            cost_estimate=resp.cost_estimate,
            quality_score=qs.overall,
            latency_ms=resp.latency_ms,
            content_type=req.content_type,
        )

    status = task_queue.submit(_work)
    return status


@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["tasks"])
async def get_task(task_id: str):
    """Poll a task's status and retrieve its result."""
    status = task_queue.get_status(task_id)
    if status is None:
        raise HTTPException(404, f"Task {task_id} not found.")
    return status


# ---------------------------------------------------------------------------
# Metrics (internal)
# ---------------------------------------------------------------------------

@app.get("/metrics/summary", tags=["system"])
async def metrics_summary():
    """Internal metrics summary."""
    return metrics.summary()
