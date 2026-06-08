# nvidia-nim-agent-toolkit

[![CI](https://github.com/TylrDn/nvidia-nim-agent-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/TylrDn/nvidia-nim-agent-toolkit/actions/workflows/ci.yml)

Multi-agent coordination system powered by **NVIDIA NIM** inference microservices and **LangGraph** orchestration. Features tool-calling agents with REST API, SQL, and document store integrations — built as a reference for enterprise agentic AI deployment.

**JD Signal:** NIM + NeMo Agent Toolkit hands-on experience — the single highest differentiator for the NVIDIA Solutions Architect, Agentic AI role.

## Architecture

```
nvidia-nim-agent-toolkit/
├── nim/                     # NIM API client + config
├── orchestrator/            # LangGraph StateGraph + Planner/Executor/Reviewer nodes
├── agents/                  # API, SQL, and Doc tool-calling agents
├── tools/                   # StructuredTool wrappers
├── configs/                 # YAML agent + model configs
├── deploy/                  # Docker Compose + Kubernetes
├── evals/                   # LangSmith eval harness
├── tests/                   # pytest unit + integration tests
└── notebooks/               # Quickstart notebook
```

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/TylrDn/nvidia-nim-agent-toolkit
cd nvidia-nim-agent-toolkit
cp .env.template .env  # Fill in NIM_API_KEY
pip install -r requirements.txt

# 2. Run the agent
python -c "
from orchestrator.graph import build_graph
graph = build_graph()
result = graph.invoke({'user_input': 'Fetch the BTC price and summarise it.'})
print(result['final_output'])
"

# 3. Run tests
pytest tests/ -v

# 4. Docker
cd deploy && docker compose up
```

## Key Components

| Component | Description |
|---|---|
| `nim/client.py` | OpenAI-compatible NIM wrapper — routes any LangChain call through NIM |
| `orchestrator/graph.py` | LangGraph StateGraph — Planner → Executor → Reviewer loop |
| `agents/` | REST API, SQL, and document retrieval tool-calling agents |
| `tools/` | `StructuredTool` wrappers for HTTP, SQL, and vector search |
| `evals/agent_eval.py` | LangSmith-backed correctness + latency evaluation harness |

## Topics

`nvidia-nim` `langgraph` `multi-agent` `llm` `python` `agentic-ai` `nemo` `inference` `enterprise-ai` `tool-calling`
