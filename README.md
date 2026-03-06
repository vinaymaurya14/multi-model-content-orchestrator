# Multi-Model Content Orchestrator

A production-quality **FastAPI** service that intelligently routes content-generation requests across multiple LLM providers (OpenAI, Anthropic, HuggingFace, Ollama) based on content type, quality requirements, cost constraints, and latency targets.

Built as a portfolio project demonstrating **multi-model orchestration expertise** for AI/ML engineering roles.

---

## Architecture

```
                           +------------------+
                           |   FastAPI App     |
                           |   (main.py)       |
                           +--------+---------+
                                    |
                 +------------------+------------------+
                 |                  |                   |
          +------+------+   +------+------+   +--------+--------+
          |   Router    |   | Benchmarker |   | Cost Optimizer  |
          | (strategy)  |   | (compare)   |   | (budget/alerts) |
          +------+------+   +------+------+   +--------+--------+
                 |                  |                   |
          +------+------+   +------+------+   +--------+--------+
          |  Model      |   |  Quality    |   |  Metrics        |
          |  Registry   |   |  Scorer     |   |  Collector      |
          +------+------+   +-------------+   +-----------------+
                 |
    +------------+------------+------------+------------+
    |            |            |            |            |
+---+---+  +----+----+  +----+----+  +----+----+  +---+---+
| OpenAI|  |Anthropic|  |HuggingF.|  | Ollama  |  | Mock  |
| GPT   |  | Claude  |  |Inference|  | (local) |  |(demo) |
+-------+  +---------+  +---------+  +---------+  +-------+
```

## Key Features

- **Intelligent Routing** -- Four strategies (cost-optimized, quality-optimized, balanced, latency-optimized) score every available model and pick the best one for each request.
- **Multi-Provider Support** -- OpenAI, Anthropic, HuggingFace Inference API, and Ollama (local). All integrations use async `httpx`.
- **Demo Mode** -- Works out-of-the-box with **zero API keys** via a realistic mock provider that generates varied content for different content types.
- **Benchmarking** -- Run the same prompt across multiple providers and get statistical comparisons (mean, std, p50, p95 latency; quality scores; costs).
- **Side-by-Side Comparison** -- Compare outputs from different models with quality breakdowns.
- **Quality Scoring** -- Offline (no API calls) scoring across five dimensions: coherence, relevance, readability (Flesch-Kincaid), completeness, and SEO-friendliness.
- **Cost Tracking** -- Real-time cost analytics by provider and model, budget alerts, and recommendations for cheaper alternatives.
- **Async Task Queue** -- Submit long-running generation tasks and poll for results.
- **Comprehensive Tests** -- Full pytest suite covering routing logic, benchmarking, quality scoring, and every API endpoint.

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd multi-model-content-orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run in Demo Mode (no API keys needed)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

The server starts with the mock provider enabled by default.

### 3. Try the API

```bash
# Health check
curl http://localhost:8002/health | python -m json.tool

# Generate content
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a blog post about the future of AI in healthcare",
    "content_type": "blog_post",
    "quality_level": "premium",
    "max_tokens": 1024
  }' | python -m json.tool

# Benchmark across providers
curl -X POST http://localhost:8002/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning in simple terms",
    "num_runs": 3
  }' | python -m json.tool

# Side-by-side comparison
curl -X POST http://localhost:8002/compare \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a product description for a smart home assistant"
  }' | python -m json.tool

# List available models
curl http://localhost:8002/models | python -m json.tool

# Cost report
curl http://localhost:8002/costs/report | python -m json.tool

# Async task
curl -X POST http://localhost:8002/tasks/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a technical doc about REST APIs"}' | python -m json.tool

# Poll task status (replace TASK_ID)
curl http://localhost:8002/tasks/TASK_ID | python -m json.tool
```

### 4. Interactive API Docs

Visit **http://localhost:8002/docs** for the auto-generated Swagger UI.

---

## Using Real Providers

Copy `.env.example` to `.env` and uncomment the keys you want to use:

```bash
cp .env.example .env
```

```env
DEMO_MODE=false
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HF_API_TOKEN=hf_...
```

Restart the server. The router will now include real providers alongside the mock.

---

## Docker

```bash
# Build and run
docker compose up --build

# Or standalone
docker build -t orchestrator .
docker run -p 8002:8002 -e DEMO_MODE=true orchestrator
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use the mock provider so they run without API keys and complete in seconds.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with uptime and provider status |
| `GET` | `/models` | List all models with capabilities and costs |
| `POST` | `/generate` | Generate content via optimally routed model |
| `POST` | `/benchmark` | Benchmark a prompt across multiple providers |
| `POST` | `/compare` | Side-by-side model output comparison |
| `GET` | `/costs/report` | Detailed cost analysis report |
| `POST` | `/tasks/generate` | Submit async generation task |
| `GET` | `/tasks/{task_id}` | Poll task status and retrieve result |
| `GET` | `/metrics/summary` | Internal performance metrics |

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `balanced` | Equal weighting across quality, cost, and latency (default) |
| `cost_optimized` | Minimize cost while meeting minimum quality |
| `quality_optimized` | Maximize output quality regardless of cost |
| `latency_optimized` | Minimize response time |

### Content Types

`blog_post`, `product_description`, `technical_doc`, `marketing_copy`, `social_media`, `email`, `general`

### Quality Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Coherence | 20% | Sentence flow, consistency, transition words |
| Relevance | 25% | Keyword overlap with the original prompt |
| Readability | 20% | Flesch-Kincaid score mapped to quality |
| Completeness | 15% | Length adequacy and structural markers |
| SEO Score | 20% | Keyword density, headings, lists, emphasis |

---

## Project Structure

```
multi-model-content-orchestrator/
├── app/
│   ├── main.py                  # FastAPI app, routes, lifespan
│   ├── config.py                # Pydantic-settings configuration
│   ├── models/schemas.py        # All Pydantic request/response models
│   ├── providers/
│   │   ├── base.py              # Abstract LLMProvider interface
│   │   ├── openai_provider.py   # OpenAI GPT integration
│   │   ├── anthropic_provider.py# Anthropic Claude integration
│   │   ├── huggingface_provider.py # HuggingFace Inference API
│   │   ├── ollama_provider.py   # Ollama local models
│   │   └── mock_provider.py     # Realistic mock provider (demo mode)
│   ├── services/
│   │   ├── router.py            # Intelligent model routing engine
│   │   ├── benchmarker.py       # Multi-model benchmarking
│   │   ├── cost_optimizer.py    # Cost tracking & budget alerts
│   │   ├── quality_scorer.py    # Offline quality scoring (5 dimensions)
│   │   └── task_queue.py        # Async task processing
│   ├── core/
│   │   ├── routing_strategy.py  # Weighted scoring algorithms
│   │   ├── model_registry.py    # Model capabilities catalogue
│   │   └── metrics.py           # Performance metrics collection
│   └── utils/
│       └── text_utils.py        # Text processing utilities
├── tests/
│   ├── test_router.py           # Routing engine & strategy tests
│   ├── test_benchmarker.py      # Benchmarker, cost, quality tests
│   └── test_api.py              # End-to-end API endpoint tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Tech Stack

- **Python 3.11+**
- **FastAPI** -- async web framework
- **Pydantic v2** -- data validation and settings
- **httpx** -- async HTTP client for provider APIs
- **NumPy** -- statistical computations
- **NLTK** -- text tokenization for quality scoring
- **pytest + pytest-asyncio** -- testing

---

## License

MIT
