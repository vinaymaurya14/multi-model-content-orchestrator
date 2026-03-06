"""Comprehensive Pydantic schemas for request / response objects."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    blog_post = "blog_post"
    product_description = "product_description"
    technical_doc = "technical_doc"
    marketing_copy = "marketing_copy"
    social_media = "social_media"
    email = "email"
    general = "general"


class QualityLevel(str, Enum):
    draft = "draft"
    standard = "standard"
    premium = "premium"


class RoutingStrategy(str, Enum):
    cost_optimized = "cost_optimized"
    quality_optimized = "quality_optimized"
    balanced = "balanced"
    latency_optimized = "latency_optimized"


class TaskState(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt to send to the model.")
    content_type: ContentType = Field(ContentType.general, description="Type of content to generate.")
    quality_level: QualityLevel = Field(QualityLevel.standard, description="Desired quality tier.")
    max_tokens: int = Field(1024, ge=1, le=32768, description="Maximum tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    preferred_provider: Optional[str] = Field(None, description="Force a specific provider (optional).")
    strategy: RoutingStrategy = Field(RoutingStrategy.balanced, description="Routing strategy to use.")


class GenerateResponse(BaseModel):
    content: str
    provider_used: str
    model_used: str
    tokens_used: int
    cost_estimate: float
    quality_score: float
    latency_ms: float
    content_type: ContentType
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    providers: Optional[List[str]] = Field(None, description="Providers to benchmark. None = all available.")
    num_runs: int = Field(1, ge=1, le=10, description="Number of runs per provider.")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class BenchmarkModelResult(BaseModel):
    provider: str
    model: str
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_tokens: float
    mean_cost: float
    total_cost: float
    mean_quality_score: float
    sample_output: str
    runs: int


class BenchmarkResponse(BaseModel):
    prompt: str
    results: List[BenchmarkModelResult]
    fastest_provider: str
    cheapest_provider: str
    highest_quality_provider: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    providers: Optional[List[str]] = Field(None, description="Providers to compare. None = all available.")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ComparisonItem(BaseModel):
    provider: str
    model: str
    content: str
    tokens_used: int
    cost_estimate: float
    quality_score: float
    latency_ms: float
    quality_breakdown: Dict[str, float] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    prompt: str
    comparisons: List[ComparisonItem]
    recommended_provider: str
    recommendation_reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    provider: str
    model_id: str
    display_name: str
    capabilities: List[str]
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_tokens: int
    quality_tier: QualityLevel
    is_available: bool = True


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------

class ProviderCost(BaseModel):
    provider: str
    total_cost: float
    request_count: int
    total_tokens: int


class ModelCost(BaseModel):
    model: str
    provider: str
    total_cost: float
    request_count: int
    total_tokens: int


class CostReport(BaseModel):
    total_cost: float
    monthly_limit: float
    budget_remaining: float
    budget_usage_pct: float
    by_provider: List[ProviderCost]
    by_model: List[ModelCost]
    recommendations: List[str]
    period_start: datetime
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Task queue
# ---------------------------------------------------------------------------

class TaskStatus(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = TaskState.pending
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    demo_mode: bool
    available_providers: List[str]
    uptime_seconds: float
