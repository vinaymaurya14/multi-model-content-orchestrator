"""End-to-end API tests using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Provide a synchronous test client with lifespan events."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health & system
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "mock" in data["available_providers"]

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert data["version"]


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        assert len(models) >= 3  # At least the 3 mock models
        for m in models:
            assert "provider" in m
            assert "model_id" in m
            assert "capabilities" in m

    def test_mock_models_present(self, client):
        models = client.get("/models").json()
        mock_models = [m for m in models if m["provider"] == "mock"]
        assert len(mock_models) >= 3


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    def test_generate_default(self, client):
        resp = client.post("/generate", json={
            "prompt": "Write a blog post about sustainable energy",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"]
        assert data["provider_used"] == "mock"
        assert data["tokens_used"] > 0
        assert data["quality_score"] > 0
        assert data["latency_ms"] > 0

    def test_generate_with_content_type(self, client):
        resp = client.post("/generate", json={
            "prompt": "Create a compelling product description for a smart thermostat",
            "content_type": "product_description",
            "quality_level": "premium",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "product_description"

    def test_generate_with_strategy(self, client):
        resp = client.post("/generate", json={
            "prompt": "Explain cloud computing",
            "strategy": "cost_optimized",
        })
        assert resp.status_code == 200

    def test_generate_empty_prompt_rejected(self, client):
        resp = client.post("/generate", json={"prompt": ""})
        assert resp.status_code == 422  # validation error

    def test_generate_marketing_copy(self, client):
        resp = client.post("/generate", json={
            "prompt": "Write marketing copy for an AI-powered writing assistant",
            "content_type": "marketing_copy",
        })
        assert resp.status_code == 200
        assert len(resp.json()["content"]) > 50

    def test_generate_technical_doc(self, client):
        resp = client.post("/generate", json={
            "prompt": "Document the REST API for a user management service",
            "content_type": "technical_doc",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class TestBenchmarkEndpoint:
    def test_benchmark_mock(self, client):
        resp = client.post("/benchmark", json={
            "prompt": "Summarize the benefits of remote work",
            "providers": ["mock"],
            "num_runs": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["provider"] == "mock"
        assert data["results"][0]["runs"] == 2
        assert data["fastest_provider"]
        assert data["cheapest_provider"]

    def test_benchmark_default_providers(self, client):
        resp = client.post("/benchmark", json={
            "prompt": "What is machine learning?",
            "num_runs": 1,
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

class TestCompareEndpoint:
    def test_compare_mock(self, client):
        resp = client.post("/compare", json={
            "prompt": "Explain the benefits of TypeScript over JavaScript",
            "providers": ["mock"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["comparisons"]) >= 1
        assert data["recommended_provider"]
        assert data["recommendation_reason"]
        comp = data["comparisons"][0]
        assert "quality_score" in comp
        assert "quality_breakdown" in comp


# ---------------------------------------------------------------------------
# Cost report
# ---------------------------------------------------------------------------

class TestCostReport:
    def test_cost_report_after_requests(self, client):
        # Generate some traffic first
        client.post("/generate", json={"prompt": "Test cost tracking"})
        resp = client.get("/costs/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_cost" in data
        assert "by_provider" in data
        assert "recommendations" in data
        assert data["monthly_limit"] > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_summary(self, client):
        resp = client.get("/metrics/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "total_cost_usd" in data
        assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Task queue
# ---------------------------------------------------------------------------

class TestTaskQueue:
    def test_submit_and_poll_task(self, client):
        # Submit
        resp = client.post("/tasks/generate", json={
            "prompt": "Write a haiku about coding",
        })
        assert resp.status_code == 200
        task = resp.json()
        assert "task_id" in task

        # Poll (may already be completed since mock is fast)
        import time
        time.sleep(0.5)
        resp2 = client.get(f"/tasks/{task['task_id']}")
        assert resp2.status_code == 200
        status = resp2.json()
        assert status["state"] in ("pending", "running", "completed")

    def test_get_nonexistent_task(self, client):
        resp = client.get("/tasks/does-not-exist")
        assert resp.status_code == 404
