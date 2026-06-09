# nvidia-nim-agent-toolkit

[![CI](https://github.com/TylrDn/nvidia-nim-agent-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/nvidia-nim-agent-toolkit/actions/workflows/ci.yml)

Multi-agent coordination system powered by **NVIDIA NIM** inference microservices and **LangGraph** orchestration. Features tool-calling agents with REST API, SQL, and document store integrations — built as a reference for enterprise agentic AI deployment.

## Architecture

```
User Query
    ↓
FastAPI /query
    ↓
LangGraph StateGraph
    ↓
Planner → Executor → Reviewer
              ↓
    [APIAgent | SQLAgent | DocAgent]
              ↓
        NVIDIA NIM (OpenAI-compatible)
```

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram.

## Quickstart

```bash
cp .env.template .env
# Add your NIM_API_KEY to .env

pip install -r requirements.txt
uvicorn api.server:app --reload --port 8080
```

## Docker

The root `docker-compose.yml` exposes two profiles:

```bash
# Cloud NIM (build.nvidia.com) — orchestrator only
docker compose --profile dev up --build

# Local GPU NIM — also starts an on-host NIM container (requires NVIDIA GPU)
docker compose --profile full up --build
```

For the `full` profile, set `NIM_BASE_URL=http://nim:8000/v1` in `.env` so the
orchestrator targets the local NIM service. The production-parameterized compose
lives in [`deploy/docker-compose.yml`](deploy/docker-compose.yml).

The API will be available at `http://localhost:8080`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | NIM readiness check |
| `/query` | POST | Run the multi-agent pipeline |
| `/models` | GET | List available NIM models |
| `/docs` | GET | Interactive Swagger UI |

### Example

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current price of NVDA stock?"}'
```

## Key Components

| Module | Description |
|---|---|
| `nim/client.py` | NIM OpenAI-compatible client — all LLM calls route here |
| `orchestrator/graph.py` | LangGraph StateGraph: Planner → Executor → Reviewer loop |
| `orchestrator/state.py` | TypedDict state schema |
| `agents/api_agent.py` | REST API tool-calling agent |
| `agents/sql_agent.py` | Text-to-SQL agent |
| `agents/doc_agent.py` | Document retrieval agent |
| `tools/` | StructuredTool wrappers (http, sql, faiss) |
| `evals/agent_eval.py` | LangSmith evaluation harness |

## Agent Configuration

Agent personas, models, prompts, tools, and iteration caps live in
[`configs/agents.yaml`](configs/agents.yaml) and are validated at startup by
`configs/loader.py` — no Python code changes required to swap a model or edit a
prompt:

```yaml
agents:
  planner:
    model: meta/llama-3.1-70b-instruct
    system_prompt: |
      You are a task planning agent. ...
```

## Environment Variables

| Variable | Description |
|---|---|
| `NIM_API_KEY` | NVIDIA NIM API key (falls back to `NVIDIA_API_KEY`) |
| `NIM_BASE_URL` | NIM endpoint (default: build.nvidia.com) |
| `NIM_DEFAULT_MODEL` | Default model when not set per-agent |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Langfuse tracing |
| `DATABASE_URL` | SQLAlchemy DB URL for SQL agent |
| `FAISS_INDEX_PATH` | Path to FAISS vector index |
| `LOG_LEVEL` / `LOG_FORMAT` | Logging level and `text`/`json` format |

## Cross-Repo Integration

This toolkit is designed as the **foundation layer** for the NVIDIA SA demo portfolio:
- [`enterprise-rag-pipeline`](https://github.com/TylrDn/enterprise-rag-pipeline) — RAG backend for DocAgent
- [`agentic-guardrails-eval`](https://github.com/TylrDn/agentic-guardrails-eval) — safety eval suite targets this API
- [`inference-optimization-bench`](https://github.com/TylrDn/inference-optimization-bench) — benchmarks the NIM client layer

## Topics

`nvidia-nim` `langgraph` `multi-agent` `llm` `python` `agentic-ai` `nemo` `inference` `enterprise-ai` `tool-calling`
