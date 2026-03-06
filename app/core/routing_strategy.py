"""Routing algorithms and strategies.

Each strategy implements a scoring function that evaluates candidate models
across multiple dimensions (cost, quality, latency) and returns a ranked
list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.model_registry import ModelEntry


@dataclass
class ScoredCandidate:
    """A model entry with a composite routing score."""
    entry: ModelEntry
    score: float
    breakdown: Dict[str, float]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize(values: List[float], invert: bool = False) -> List[float]:
    """Min-max normalise a list of floats to [0, 1].  If *invert* is True
    lower raw values get higher normalised scores (useful for cost / latency)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [1.0] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    if invert:
        normed = [1.0 - n for n in normed]
    return normed


def _score_candidates(
    candidates: List[ModelEntry],
    weights: Dict[str, float],
    latency_estimates: Optional[Dict[str, float]] = None,
) -> List[ScoredCandidate]:
    """Score a list of model entries using weighted dimensions.

    Dimensions:
      - quality  : higher is better
      - cost     : lower is better (inverted)
      - latency  : lower is better (inverted)
      - capacity : higher max_tokens is better
    """
    if not candidates:
        return []

    # Raw values
    quality_raw = [c.quality_score for c in candidates]
    cost_raw = [c.avg_cost_per_1k for c in candidates]
    capacity_raw = [float(c.max_tokens) for c in candidates]

    if latency_estimates:
        latency_raw = [latency_estimates.get(f"{c.provider}:{c.model_id}", 500.0) for c in candidates]
    else:
        # Approximate: cheaper models tend to be faster
        latency_raw = [200 + c.avg_cost_per_1k * 50000 for c in candidates]

    # Normalise
    quality_norm = _normalize(quality_raw)
    cost_norm = _normalize(cost_raw, invert=True)
    latency_norm = _normalize(latency_raw, invert=True)
    capacity_norm = _normalize(capacity_raw)

    w_q = weights.get("quality", 0.4)
    w_c = weights.get("cost", 0.3)
    w_l = weights.get("latency", 0.2)
    w_cap = weights.get("capacity", 0.1)

    scored: List[ScoredCandidate] = []
    for i, c in enumerate(candidates):
        breakdown = {
            "quality": round(quality_norm[i], 4),
            "cost": round(cost_norm[i], 4),
            "latency": round(latency_norm[i], 4),
            "capacity": round(capacity_norm[i], 4),
        }
        composite = (
            w_q * quality_norm[i]
            + w_c * cost_norm[i]
            + w_l * latency_norm[i]
            + w_cap * capacity_norm[i]
        )
        scored.append(ScoredCandidate(entry=c, score=round(composite, 4), breakdown=breakdown))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

_STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "cost_optimized": {"quality": 0.15, "cost": 0.60, "latency": 0.15, "capacity": 0.10},
    "quality_optimized": {"quality": 0.60, "cost": 0.10, "latency": 0.15, "capacity": 0.15},
    "balanced": {"quality": 0.35, "cost": 0.30, "latency": 0.20, "capacity": 0.15},
    "latency_optimized": {"quality": 0.15, "cost": 0.15, "latency": 0.60, "capacity": 0.10},
}


def rank_candidates(
    candidates: List[ModelEntry],
    strategy: str = "balanced",
    latency_estimates: Optional[Dict[str, float]] = None,
) -> List[ScoredCandidate]:
    """Rank model entries using the specified strategy.

    Parameters
    ----------
    candidates : available model entries
    strategy   : one of cost_optimized, quality_optimized, balanced, latency_optimized
    latency_estimates : optional dict mapping "provider:model_id" -> avg latency ms

    Returns
    -------
    List of ScoredCandidate, best first.
    """
    weights = _STRATEGY_WEIGHTS.get(strategy, _STRATEGY_WEIGHTS["balanced"])
    return _score_candidates(candidates, weights, latency_estimates)


def available_strategies() -> List[str]:
    return list(_STRATEGY_WEIGHTS.keys())
