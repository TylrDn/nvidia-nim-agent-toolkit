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
# Add your NVIDIA_API_KEY to .env

pip install -r requirements.txt
uvicorn api.server:app --reload --port 8080
```

## Docker

```bash
cd deploy
docker-compose up --build
```

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

## NIM Model Configuration

Swap models in `nim/config.yaml` — no Python code changes required:

```yaml
default_model: meta/llama-3.1-70b-instruct
```

## Environment Variables

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA build.nvidia.com API key |
| `NIM_BASE_URL` | NIM endpoint (default: build.nvidia.com) |
| `DATABASE_URL` | SQLAlchemy DB URL for SQL agent |
| `LANGSMITH_API_KEY` | LangSmith tracing + evals |
| `FAISS_INDEX_PATH` | Path to FAISS vector index |

## Cross-Repo Integration

This toolkit is designed as the **foundation layer** for the NVIDIA SA demo portfolio:
- [`enterprise-rag-pipeline`](https://github.com/TylrDn/enterprise-rag-pipeline) — RAG backend for DocAgent
- [`agentic-guardrails-eval`](https://github.com/TylrDn/agentic-guardrails-eval) — safety eval suite targets this API
- [`inference-optimization-bench`](https://github.com/TylrDn/inference-optimization-bench) — benchmarks the NIM client layer

## Topics

`nvidia-nim` `langgraph` `multi-agent` `llm` `python` `agentic-ai` `nemo` `inference` `enterprise-ai` `tool-calling`
