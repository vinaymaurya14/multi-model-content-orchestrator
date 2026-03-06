"""Cost tracking and optimization service.

Provides cost breakdowns by provider and model, budget alerts, and
recommendations for cheaper alternatives that meet quality thresholds.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from app.config import settings
from app.core.metrics import MetricsCollector
from app.core.model_registry import ModelRegistry
from app.models.schemas import CostReport, ModelCost, ProviderCost


class CostOptimizer:
    """Analyse spending and suggest cost-saving strategies."""

    def __init__(self, registry: ModelRegistry, metrics: MetricsCollector) -> None:
        self.registry = registry
        self.metrics = metrics

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> CostReport:
        total = self.metrics.total_cost()
        limit = settings.monthly_cost_limit
        remaining = max(0.0, limit - total)
        usage_pct = (total / limit * 100) if limit > 0 else 0.0

        # By provider
        by_provider_raw = self.metrics.cost_by_provider()
        by_provider = [
            ProviderCost(
                provider=pname,
                total_cost=round(info["cost"], 6),
                request_count=info["requests"],
                total_tokens=info["tokens"],
            )
            for pname, info in by_provider_raw.items()
        ]

        # By model
        by_model_raw = self.metrics.cost_by_model()
        by_model = [
            ModelCost(
                model=key.split(":", 1)[1] if ":" in key else key,
                provider=info["provider"],
                total_cost=round(info["cost"], 6),
                request_count=info["requests"],
                total_tokens=info["tokens"],
            )
            for key, info in by_model_raw.items()
        ]

        recommendations = self._generate_recommendations(
            total, limit, by_provider, by_model
        )

        return CostReport(
            total_cost=round(total, 6),
            monthly_limit=limit,
            budget_remaining=round(remaining, 6),
            budget_usage_pct=round(usage_pct, 2),
            by_provider=sorted(by_provider, key=lambda p: p.total_cost, reverse=True),
            by_model=sorted(by_model, key=lambda m: m.total_cost, reverse=True),
            recommendations=recommendations,
            period_start=self.metrics.period_start(),
            generated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Budget check
    # ------------------------------------------------------------------

    def is_over_budget(self) -> bool:
        if settings.monthly_cost_limit <= 0:
            return False
        return self.metrics.total_cost() >= settings.monthly_cost_limit

    def is_near_budget(self) -> bool:
        if settings.monthly_cost_limit <= 0:
            return False
        return (
            self.metrics.total_cost()
            >= settings.monthly_cost_limit * settings.cost_alert_threshold
        )

    # ------------------------------------------------------------------
    # Cheapest model recommendation
    # ------------------------------------------------------------------

    def cheapest_meeting_quality(self, min_quality_tier: str = "standard") -> str | None:
        """Return the cheapest available model id that meets the quality threshold."""
        tier_order = {"draft": 0, "standard": 1, "premium": 2}
        min_level = tier_order.get(min_quality_tier, 1)

        eligible = [
            e for e in self.registry.get_available()
            if tier_order.get(e.quality_tier, 0) >= min_level
        ]
        if not eligible:
            return None
        cheapest = min(eligible, key=lambda e: e.avg_cost_per_1k)
        return f"{cheapest.provider}:{cheapest.model_id}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendations(
        total: float,
        limit: float,
        by_provider: List[ProviderCost],
        by_model: List[ModelCost],
    ) -> List[str]:
        recs: List[str] = []

        if limit > 0:
            pct = total / limit * 100
            if pct >= 100:
                recs.append(
                    "ALERT: Monthly budget exhausted. Switch to the mock provider "
                    "or increase MONTHLY_COST_LIMIT."
                )
            elif pct >= 80:
                recs.append(
                    f"WARNING: {pct:.0f}% of monthly budget used. Consider "
                    "switching to a cheaper model or reducing request volume."
                )

        if by_provider:
            most_expensive = max(by_provider, key=lambda p: p.total_cost)
            if most_expensive.total_cost > 0:
                recs.append(
                    f"Most expensive provider: {most_expensive.provider} "
                    f"(${most_expensive.total_cost:.4f}). Evaluate if a cheaper "
                    "alternative meets your quality needs."
                )

        if by_model and len(by_model) > 1:
            cheapest = min(by_model, key=lambda m: m.total_cost / max(m.request_count, 1))
            recs.append(
                f"Most cost-effective model: {cheapest.model} "
                f"(${cheapest.total_cost / max(cheapest.request_count, 1):.6f}/request). "
                "Consider routing more traffic here."
            )

        if not recs:
            recs.append("Spending is within budget. No action needed.")

        return recs
