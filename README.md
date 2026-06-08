# nvidia-nim-agent-toolkit

[![CI](https://github.com/TylrDn/nvidia-nim-agent-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/nvidia-nim-agent-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Multi-agent coordination system powered by **NVIDIA NIM** inference microservices and **LangGraph** orchestration. Features a Planner → Executor → Reviewer loop with three specialist agents (REST API, SQL, Document Retrieval), a FastAPI server, and a Langfuse-traced evaluation harness.

> **Target role:** [Solutions Architect, Agentic AI — NVIDIA JR2014517](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Solutions-Architect--Agentic-AI_JR2014517)

---

## Architecture

```
POST /v1/run {intent}
  └─ LangGraph pipeline
       ├─ Planner   → decomposes intent into ordered task list
       ├─ Executor  → routes each task to: API Agent | SQL Agent | Doc Agent | LLM
       ├─ Reviewer  → scores output (0–1), drives retry or advance
       └─ Synthesiser → composes final answer from all task results
```

See [`docs/architecture.md`](docs/architecture.md) for full Mermaid diagram and component map.

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/TylrDn/nvidia-nim-agent-toolkit.git
cd nvidia-nim-agent-toolkit
cp .env.template .env
# Add your NIM_API_KEY to .env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API server

```bash
uvicorn api.server:app --reload --port 8080
```

### 4. Run a query

```bash
curl -X POST http://localhost:8080/v1/run \
  -H "Content-Type: application/json" \
  -d '{"intent": "What are the key principles of RAG?"}'
```

### 5. Docker (full stack)

```bash
cd deploy
docker-compose up --build
```

---

## Project Structure

```
nvidia-nim-agent-toolkit/
├── nim/                   # NIM client + health check + model configs
├── orchestrator/          # LangGraph StateGraph + Planner/Executor/Reviewer nodes
├── agents/                # API, SQL, and Document specialist agents
├── tools/                 # StructuredTool wrappers (HTTP, SQL, vector search)
├── api/                   # FastAPI server
├── configs/               # agents.yaml — swap models/personas without code changes
├── evals/                 # Evaluation harness with Langfuse tracing
├── deploy/                # Dockerfile + docker-compose.yml
├── docs/                  # Architecture docs with Mermaid diagrams
└── tests/                 # Unit tests (pytest)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| NIM via OpenAI-compat API | Drop-in swap between cloud NIM and self-hosted NIM — zero code change |
| LangGraph StateGraph | Explicit state schema + conditional edges → production-debuggable flow |
| YAML agent configs | Persona and model changes at runtime — no redeploy needed |
| StructuredTool wrappers | Pydantic-validated inputs — prevents malformed tool calls |
| Langfuse tracing | Full observability on all LLM paths — required for enterprise deployments |
| pgvector default | Lightweight vector store — Milvus swap via single env var |

---

## Environment Variables

See [`.env.template`](.env.template) for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NIM_API_KEY` | — | **Required.** NVIDIA API key |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM endpoint (cloud or self-hosted) |
| `NIM_DEFAULT_MODEL` | `meta/llama-3.1-70b-instruct` | Default inference model |
| `DATABASE_URL` | `sqlite:///./demo.db` | SQLAlchemy connection string |
| `PGVECTOR_URL` | — | PostgreSQL + pgvector connection |
| `VECTOR_BACKEND` | `pgvector` | `pgvector` or `milvus` |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse observability (optional) |

---

## Running Evals

```bash
python -m evals.agent_eval
# Reports written to evals/reports/eval_report_<timestamp>.json
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## CI/CD

GitHub Actions runs on every push:
- `ruff` lint
- `mypy` type check  
- `pytest` unit tests

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
