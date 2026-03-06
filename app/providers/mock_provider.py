"""Mock LLM provider for demo mode.

Generates realistic, varied responses for different content types without
requiring any API keys.  Designed to make the demo experience convincing and
useful for integration testing.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from typing import List

from app.providers.base import LLMProvider, LLMResponse

# ---------------------------------------------------------------------------
# Response templates keyed by content type keyword
# ---------------------------------------------------------------------------

_BLOG_POSTS = [
    (
        "# {topic}\n\n"
        "In today's rapidly evolving digital landscape, understanding {topic} "
        "has become essential for businesses and individuals alike. This "
        "comprehensive guide explores the key concepts, best practices, and "
        "emerging trends you need to know.\n\n"
        "## Why {topic} Matters\n\n"
        "The importance of {topic} cannot be overstated. Recent studies show "
        "that organizations embracing {topic} see a 40% improvement in "
        "operational efficiency and a 25% increase in customer satisfaction.\n\n"
        "## Key Strategies\n\n"
        "1. **Start with a clear vision** -- Define your goals before diving "
        "into implementation. A well-articulated vision ensures alignment "
        "across your team.\n\n"
        "2. **Iterate and measure** -- Adopt an agile approach. Launch small "
        "experiments, gather data, and refine your strategy based on real "
        "results.\n\n"
        "3. **Invest in your people** -- Technology is only as effective as "
        "the team behind it. Training and upskilling are non-negotiable.\n\n"
        "## Looking Ahead\n\n"
        "As we move further into 2026, {topic} will continue to shape the "
        "competitive landscape. Early adopters who build strong foundations "
        "today will be best positioned to capitalize on tomorrow's "
        "opportunities.\n\n"
        "---\n*Published by the Multi-Model Content Orchestrator*"
    ),
    (
        "# The Complete Guide to {topic}\n\n"
        "Whether you are a seasoned professional or just getting started, "
        "mastering {topic} is a game-changer. In this article we break down "
        "the essentials into actionable insights.\n\n"
        "## The Fundamentals\n\n"
        "At its core, {topic} revolves around three pillars:\n"
        "- **Clarity** -- Knowing exactly what you want to achieve.\n"
        "- **Consistency** -- Showing up every day with disciplined execution.\n"
        "- **Creativity** -- Finding novel angles that set you apart.\n\n"
        "## Common Mistakes to Avoid\n\n"
        "Many newcomers fall into the trap of over-engineering their approach. "
        "Instead, focus on delivering value quickly and iterating from there. "
        "Perfectionism is the enemy of progress.\n\n"
        "## Expert Tips\n\n"
        "We interviewed 50 industry leaders and distilled their advice into "
        "five universal principles:\n\n"
        "1. Prioritize user experience above all else.\n"
        "2. Leverage automation wherever possible.\n"
        "3. Build in public -- transparency builds trust.\n"
        "4. Use data, but don't ignore intuition.\n"
        "5. Stay curious and never stop learning.\n\n"
        "## Conclusion\n\n"
        "{topic} is not a destination but a journey. Embrace the process, "
        "stay adaptable, and the results will follow.\n\n"
        "---\n*Generated in demo mode by the Multi-Model Content Orchestrator*"
    ),
]

_PRODUCT_DESCRIPTIONS = [
    (
        "Introducing **{topic}** -- the next generation of intelligent solutions "
        "designed for modern professionals.\n\n"
        "**Key Features:**\n"
        "- Lightning-fast performance with AI-powered optimization\n"
        "- Seamless integration with your existing workflow\n"
        "- Enterprise-grade security and compliance\n"
        "- Intuitive interface that requires zero training\n\n"
        "**Why Choose {topic}?**\n\n"
        "Built from the ground up with user feedback, {topic} delivers a "
        "30% productivity boost while reducing operational costs by up to 20%. "
        "Our proprietary adaptive engine learns your preferences and optimizes "
        "its recommendations in real time.\n\n"
        "**Pricing:** Starting at $29/month with a 14-day free trial.\n\n"
        "*Try {topic} today and experience the difference.*"
    ),
    (
        "**{topic}** -- Engineered for Excellence\n\n"
        "Discover a product that redefines what is possible. {topic} combines "
        "cutting-edge technology with thoughtful design to deliver an "
        "unmatched experience.\n\n"
        "**Highlights:**\n"
        "- AI-driven personalization that adapts to you\n"
        "- Cross-platform compatibility (Web, iOS, Android)\n"
        "- Real-time analytics dashboard\n"
        "- 24/7 priority support\n\n"
        "**What Our Customers Say:**\n"
        "\"Since switching to {topic}, our team productivity has doubled.\" "
        "-- Sarah M., Head of Operations at TechCorp\n\n"
        "**Get started free** at example.com/{topic_slug}."
    ),
]

_TECHNICAL_DOCS = [
    (
        "# {topic} -- Technical Reference\n\n"
        "## Overview\n\n"
        "{topic} provides a high-performance API for content orchestration "
        "across multiple large language models. This document covers "
        "architecture, configuration, and usage patterns.\n\n"
        "## Architecture\n\n"
        "```\n"
        "Client --> API Gateway --> Router --> Provider Pool\n"
        "                            |             |\n"
        "                        Metrics       LLM APIs\n"
        "```\n\n"
        "## Configuration\n\n"
        "| Parameter | Type | Default | Description |\n"
        "|-----------|------|---------|-------------|\n"
        "| `max_tokens` | int | 1024 | Maximum output tokens |\n"
        "| `temperature` | float | 0.7 | Sampling temperature |\n"
        "| `strategy` | str | balanced | Routing strategy |\n\n"
        "## Quick Start\n\n"
        "```python\n"
        "import httpx\n\n"
        "response = httpx.post(\n"
        "    'http://localhost:8002/generate',\n"
        "    json={{'prompt': 'Explain {topic}', 'max_tokens': 512}}\n"
        ")\n"
        "print(response.json()['content'])\n"
        "```\n\n"
        "## Error Handling\n\n"
        "All errors return a JSON body with `detail` describing the issue. "
        "HTTP status codes follow REST conventions (400 for bad requests, "
        "500 for server errors).\n\n"
        "---\n*Auto-generated technical documentation*"
    ),
]

_MARKETING_COPY = [
    (
        "**Unlock the Power of {topic}**\n\n"
        "Are you ready to take your business to the next level? {topic} "
        "is the secret weapon that top-performing companies use to stay "
        "ahead of the competition.\n\n"
        "**The Numbers Speak for Themselves:**\n"
        "- 10,000+ businesses trust {topic}\n"
        "- 98% customer satisfaction rate\n"
        "- ROI realized in under 30 days\n\n"
        "**Limited-Time Offer:** Sign up before the end of the month and "
        "receive 3 months free. No credit card required.\n\n"
        "**Don't get left behind.** Your competitors are already using {topic}. "
        "Join the movement today.\n\n"
        "[Get Started Now] | [Watch Demo] | [Read Case Studies]"
    ),
    (
        "**{topic}: Transform Your Workflow in Minutes**\n\n"
        "Imagine cutting your content creation time in half while doubling "
        "the quality. That is exactly what {topic} delivers.\n\n"
        "**Trusted by Industry Leaders:**\n"
        "Fortune 500 companies and ambitious startups alike rely on {topic} "
        "to power their content strategy.\n\n"
        "**What You Get:**\n"
        "- Instant AI-powered content generation\n"
        "- Multi-channel optimization\n"
        "- Brand voice consistency across every touchpoint\n"
        "- Actionable analytics and performance insights\n\n"
        "**Start your free trial today.** No strings attached.\n\n"
        "*{topic} -- Content that converts.*"
    ),
]

_SOCIAL_MEDIA = [
    (
        "Just discovered {topic} and it is a total game-changer! If you are "
        "looking to level up your content strategy, this is it. The AI-powered "
        "routing picks the best model for every task -- saving time AND money. "
        "Highly recommend checking it out. #AI #ContentStrategy #Innovation"
    ),
    (
        "Hot take: {topic} is the future of content creation. We tested it "
        "against manual workflows and saw 3x faster turnaround with better "
        "quality scores. The benchmark feature alone is worth it. "
        "#MachineLearning #Productivity #Tech"
    ),
]

_EMAIL = [
    (
        "Subject: Exciting Update About {topic}\n\n"
        "Hi there,\n\n"
        "I wanted to share some exciting news about {topic}. We have been "
        "working hard to bring you a solution that truly makes a difference, "
        "and I think you will love what we have built.\n\n"
        "Here is what is new:\n"
        "- Smarter model routing for faster, cheaper results\n"
        "- Side-by-side comparison so you can see the difference\n"
        "- Detailed cost analytics to keep budgets in check\n\n"
        "I would love to set up a quick 15-minute call to walk you through "
        "a live demo. Are you available this week?\n\n"
        "Best regards,\n"
        "The Orchestrator Team"
    ),
]

_GENERAL = [
    (
        "Here is a comprehensive overview of {topic}:\n\n"
        "{topic} encompasses a broad range of techniques and practices "
        "designed to improve efficiency, quality, and scalability. At its "
        "foundation, {topic} relies on intelligent automation, data-driven "
        "decision-making, and continuous iteration.\n\n"
        "**Core Concepts:**\n\n"
        "1. **Abstraction** -- Simplify complexity by hiding implementation "
        "details behind clean interfaces.\n"
        "2. **Composition** -- Build powerful systems by combining smaller, "
        "well-tested components.\n"
        "3. **Feedback loops** -- Use metrics and user signals to guide "
        "improvement.\n\n"
        "**Practical Applications:**\n\n"
        "- Automated content generation at scale\n"
        "- Intelligent routing across multiple AI models\n"
        "- Real-time quality assessment and optimization\n"
        "- Cost management and budget allocation\n\n"
        "In summary, {topic} is a discipline that rewards systematic thinking "
        "and a willingness to experiment. Start small, measure everything, "
        "and scale what works."
    ),
    (
        "Let me break down {topic} for you.\n\n"
        "{topic} is an important area that intersects technology, strategy, "
        "and execution. Understanding it requires looking at multiple "
        "dimensions:\n\n"
        "**The What:** {topic} refers to the systematic approach of "
        "orchestrating multiple AI models to produce optimal content output. "
        "Rather than relying on a single model, the system intelligently "
        "routes requests to the most suitable provider.\n\n"
        "**The Why:** Different models excel at different tasks. By leveraging "
        "a diverse model portfolio, organizations can maximize quality while "
        "minimizing costs. Benchmarks show up to 35% cost savings with "
        "comparable or better quality.\n\n"
        "**The How:** A routing engine evaluates each request against model "
        "capabilities, cost constraints, and quality requirements. It then "
        "dispatches the request to the optimal provider and scores the "
        "resulting output.\n\n"
        "This approach is already used by leading AI companies and is "
        "becoming the standard for production content systems."
    ),
]

_TEMPLATES = {
    "blog_post": _BLOG_POSTS,
    "product_description": _PRODUCT_DESCRIPTIONS,
    "technical_doc": _TECHNICAL_DOCS,
    "marketing_copy": _MARKETING_COPY,
    "social_media": _SOCIAL_MEDIA,
    "email": _EMAIL,
    "general": _GENERAL,
}

_MOCK_MODELS = [
    {
        "model_id": "mock-gpt-4o",
        "display_name": "Mock GPT-4o (Demo)",
        "quality_tier": "premium",
        "cost_input": 0.005,
        "cost_output": 0.015,
        "max_tokens": 128000,
    },
    {
        "model_id": "mock-claude-sonnet",
        "display_name": "Mock Claude Sonnet (Demo)",
        "quality_tier": "premium",
        "cost_input": 0.003,
        "cost_output": 0.015,
        "max_tokens": 200000,
    },
    {
        "model_id": "mock-mistral-7b",
        "display_name": "Mock Mistral 7B (Demo)",
        "quality_tier": "standard",
        "cost_input": 0.0001,
        "cost_output": 0.0001,
        "max_tokens": 8192,
    },
]


def _extract_topic(prompt: str) -> str:
    """Extract a short topic phrase from the prompt for template filling."""
    # Use the first meaningful clause (up to 60 chars)
    cleaned = prompt.strip().split("\n")[0]
    # Remove common prefixes
    for prefix in ("write ", "create ", "generate ", "draft ", "compose ", "explain ", "describe "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    # Remove leading articles
    for article in ("a ", "an ", "the "):
        if cleaned.lower().startswith(article):
            cleaned = cleaned[len(article):]
            break
    return cleaned[:80].strip().rstrip(".")


def _pick_template(prompt: str, content_type: str) -> str:
    """Deterministically select a template based on prompt hash so the same
    prompt always produces the same template (but different prompts vary)."""
    templates = _TEMPLATES.get(content_type, _GENERAL)
    idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(templates)
    return templates[idx]


class MockProvider(LLMProvider):
    """Fully functional mock provider that generates realistic content
    without any API keys.  This is the default provider in demo mode."""

    name = "mock"

    def __init__(self) -> None:
        self._call_count = 0

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return True

    async def health_check(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        self._call_count += 1
        content_type: str = kwargs.get("content_type", "general")
        model = kwargs.get("model", self._pick_model(prompt))

        start = self._timer()

        # Simulate realistic network + inference latency (150-800 ms)
        latency_base = 150 + (hash(prompt) % 400)
        jitter = random.uniform(-50, 100)
        simulated_ms = max(80, latency_base + jitter)
        # Actually sleep a fraction to make the demo feel realistic
        await asyncio.sleep(simulated_ms / 4000)

        topic = _extract_topic(prompt)
        topic_slug = topic.lower().replace(" ", "-")[:30]
        template = _pick_template(prompt, content_type)
        text = template.format(topic=topic, topic_slug=topic_slug)

        # Truncate to approximate max_tokens (4 chars per token)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "\n\n[...truncated]"

        elapsed = self._timer() - start
        # Report the simulated latency, not the real (much shorter) one
        reported_latency = max(elapsed, simulated_ms)

        input_tokens = self._estimate_token_count(prompt)
        output_tokens = self._estimate_token_count(text)

        model_info = self._model_info(model)
        cost = (input_tokens / 1000) * model_info["cost_input"] + \
               (output_tokens / 1000) * model_info["cost_output"]

        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=round(reported_latency, 2),
            cost_estimate=round(cost, 6),
            finish_reason="stop",
            raw={"demo": True, "call_count": self._call_count},
        )

    def get_models(self) -> List[dict]:
        return [
            {
                "provider": self.name,
                "model_id": m["model_id"],
                "display_name": m["display_name"],
                "capabilities": [
                    "text_generation", "summarization", "translation",
                    "blog_post", "product_description", "technical_doc",
                    "marketing_copy", "social_media", "email",
                ],
                "cost_per_1k_input_tokens": m["cost_input"],
                "cost_per_1k_output_tokens": m["cost_output"],
                "max_tokens": m["max_tokens"],
                "quality_tier": m["quality_tier"],
                "is_available": True,
            }
            for m in _MOCK_MODELS
        ]

    def estimate_cost(self, input_tokens: int, output_tokens: int, **kwargs) -> float:
        model = kwargs.get("model", _MOCK_MODELS[0]["model_id"])
        info = self._model_info(model)
        return (input_tokens / 1000) * info["cost_input"] + \
               (output_tokens / 1000) * info["cost_output"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(prompt: str) -> str:
        """Deterministically pick a mock model based on prompt hash."""
        idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(_MOCK_MODELS)
        return _MOCK_MODELS[idx]["model_id"]

    @staticmethod
    def _model_info(model_id: str) -> dict:
        for m in _MOCK_MODELS:
            if m["model_id"] == model_id:
                return m
        return _MOCK_MODELS[0]
