"""In-memory performance metrics collection and aggregation.

Tracks request counts, latencies, error rates, token usage, and costs.
Provides summary statistics for monitoring and the cost report endpoint.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RequestRecord:
    """A single recorded API interaction."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    content_type: str = "general"


class MetricsCollector:
    """Thread-safe in-memory metrics store."""

    def __init__(self) -> None:
        self._records: List[RequestRecord] = []
        self._lock = threading.Lock()
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def record_success(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
        content_type: str = "general",
    ) -> None:
        self.record(RequestRecord(
            provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost=cost, latency_ms=latency_ms, success=True,
            content_type=content_type,
        ))

    def record_failure(self, provider: str, model: str, latency_ms: float) -> None:
        self.record(RequestRecord(
            provider=provider, model=model,
            input_tokens=0, output_tokens=0,
            cost=0.0, latency_ms=latency_ms, success=False,
        ))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def total_requests(self) -> int:
        return len(self._records)

    def total_cost(self) -> float:
        return sum(r.cost for r in self._records)

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self._records)

    def error_rate(self) -> float:
        if not self._records:
            return 0.0
        failures = sum(1 for r in self._records if not r.success)
        return failures / len(self._records)

    def cost_by_provider(self) -> Dict[str, dict]:
        buckets: Dict[str, dict] = defaultdict(lambda: {"cost": 0.0, "requests": 0, "tokens": 0})
        for r in self._records:
            b = buckets[r.provider]
            b["cost"] += r.cost
            b["requests"] += 1
            b["tokens"] += r.input_tokens + r.output_tokens
        return dict(buckets)

    def cost_by_model(self) -> Dict[str, dict]:
        buckets: Dict[str, dict] = defaultdict(lambda: {"cost": 0.0, "requests": 0, "tokens": 0, "provider": ""})
        for r in self._records:
            key = f"{r.provider}:{r.model}"
            b = buckets[key]
            b["cost"] += r.cost
            b["requests"] += 1
            b["tokens"] += r.input_tokens + r.output_tokens
            b["provider"] = r.provider
        return dict(buckets)

    def latency_stats(self, provider: Optional[str] = None) -> dict:
        """Return latency statistics (mean, std, p50, p95) optionally filtered by provider."""
        latencies = [
            r.latency_ms for r in self._records
            if r.success and (provider is None or r.provider == provider)
        ]
        if not latencies:
            return {"mean": 0, "std": 0, "p50": 0, "p95": 0, "count": 0}
        arr = np.array(latencies)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "count": len(latencies),
        }

    def avg_latency_by_model(self) -> Dict[str, float]:
        """Return average latency per provider:model key."""
        buckets: Dict[str, List[float]] = defaultdict(list)
        for r in self._records:
            if r.success:
                buckets[f"{r.provider}:{r.model}"].append(r.latency_ms)
        return {k: round(float(np.mean(v)), 2) for k, v in buckets.items()}

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests(),
            "total_cost_usd": round(self.total_cost(), 6),
            "total_tokens": self.total_tokens(),
            "error_rate": round(self.error_rate(), 4),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "latency": self.latency_stats(),
        }

    def period_start(self) -> datetime:
        return datetime.fromtimestamp(self._start_time, tz=timezone.utc)
