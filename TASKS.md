# TASKS — nvidia-nim-agent-toolkit

**Completion:** 95%
**Last Audit:** 2025-01-30
**Repo Role:** NVIDIA SA reference implementation for NIM-powered multi-agent systems using LangGraph.

---

## Priority 1 — Critical Gaps (Must Fix)

These issues block production readiness or violate cross-repo standards.

### 1.1 — Add Langfuse Tracing to All LLM Call Paths

- [ ] **`nim/client.py`** — Add `get_langfuse_handler()` factory function; inject `CallbackHandler` into `ChatOpenAI` constructor.
  - Acceptance: `grep -n "CallbackHandler" nim/client.py` returns at least one result showing handler is instantiated and passed to `callbacks=[]`.
- [ ] **`agents/api_agent.py`** — Import `get_langfuse_handler` from `nim/client`; pass handler to LLM or chain invocations.
  - Acceptance: No `ChatOpenAI(` constructor call in this file lacks `callbacks=`.
- [ ] **`agents/sql_agent.py`** — Same as above.
  - Acceptance: Same as above.
- [ ] **`agents/doc_agent.py`** — Same as above.
  - Acceptance: Same as above.
- [ ] **`orchestrator/nodes/planner.py`** — Pass `config={"callbacks": [langfuse_handler]}` to all `llm.ainvoke()` calls.
  - Acceptance: Handler is present in all async LLM calls.
- [ ] **`orchestrator/nodes/executor.py`** — Same as above.
  - Acceptance: Same as above.
- [ ] **`orchestrator/nodes/reviewer.py`** — Same as above.
  - Acceptance: Same as above.
- [ ] **`.env.template`** — Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` variables with comments.
  - Acceptance: All three keys present in `.env.template`.
- [ ] **`requirements.txt`** — Add `langfuse>=2.0.0`.
  - Acceptance: `pip install -r requirements.txt` succeeds; `python -c "import langfuse"` exits 0.
- [ ] **`tests/conftest.py`** — Add `mock_langfuse` fixture to patch `langfuse.callback.CallbackHandler`.
  - Acceptance: All existing tests pass after Langfuse changes with no new network calls.

> Use the `.cursor/agents/add-langfuse-tracing.md` agent for this task.

---

### 1.2 — Add Root-Level docker-compose.yml

- [ ] **`docker-compose.yml`** (create at repo root) — Developer convenience compose pointing to `deploy/Dockerfile` with `env_file: .env`.
  - Acceptance: `docker compose config` from repo root exits 0; `docker compose up` starts the API on port 8000.
- [ ] **`README.md`** — Update "Getting Started" section to use `docker compose up` from root, not `cd deploy && docker compose up`.
  - Acceptance: README no longer instructs users to `cd` into `deploy/`.

> Use the `.cursor/agents/fix-docker-compose.md` agent for this task.

---

## Priority 2 — Polish (Should Fix)

These items improve code quality, maintainability, and developer experience.

### 2.1 — Add `.dockerignore` at Repo Root

- [ ] **`.dockerignore`** (create) — Exclude `.git`, `.env`, `*.pyc`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, `notebooks/`, `docs/`, `tests/`, `evals/`.
  - Acceptance: Docker build context size is reduced; sensitive files are not included in image.

### 2.2 — Improve Health Check Endpoint

- [ ] **`api/main.py`** or **`api/server.py`** — Ensure `/health` endpoint returns `{"status": "ok", "nim_reachable": bool}` by pinging NIM health check.
  - Acceptance: `GET /health` returns 200 with JSON body including `nim_reachable` field.
- [ ] **`nim/health_check.py`** — Expose `async def check_nim_health() -> bool` that the API endpoint calls.
  - Acceptance: Function exists, is typed, and has a docstring.

### 2.3 — Structured Logging Configuration

- [ ] **`api/main.py`** or a new **`core/logging.py`** — Configure `logging.basicConfig` with JSON-structured format for production use.
  - Acceptance: All log output is valid JSON with `level`, `message`, `timestamp`, `module` fields when `LOG_FORMAT=json` env var is set.

### 2.4 — Pin Dependency Versions in `requirements.txt`

- [ ] **`requirements.txt`** — Pin all dependencies with exact versions (e.g., `langchain-core==0.3.X`). Add a comment at the top: `# Generated with pip freeze — update with care`.
  - Acceptance: `pip install -r requirements.txt` is fully reproducible across environments.

### 2.5 — Validate `agents.yaml` on Startup

- [ ] **`configs/agents.yaml`** / loader module — Add explicit validation of required fields (`name`, `model`, `system_prompt`, `tools`, `max_iterations`) at startup. Raise `ValueError` with a clear message if any field is missing.
  - Acceptance: Starting the API with a malformed `agents.yaml` raises a `ValueError` with the name of the missing field before any LLM call is made.

### 2.6 — CI Coverage Gate

- [ ] **`.github/workflows/ci.yml`** — Add `--cov-fail-under=80` to the pytest command so CI fails if coverage drops below 80%.
  - Acceptance: `ci.yml` pytest step includes `--cov-fail-under=80`.

---

## Priority 3 — Enhancements (Nice to Have)

### 3.1 — Add Async Batch Inference Support

- [ ] **`nim/client.py`** — Add `async def batch_chat_completion(prompts: list[str]) -> list[str]` using `asyncio.gather`.
  - Acceptance: Method exists, is typed, has a docstring, and has a unit test.

### 3.2 — Eval Script Improvements

- [ ] **`evals/agent_eval.py`** — Replace any `print()` calls with `logger` output; add `--output-file` CLI argument to save eval results as JSON.
  - Acceptance: `ruff check evals/` passes; eval results can be saved to a file.

### 3.3 — Notebook Cleanup

- [ ] **`notebooks/quickstart.ipynb`** — Clear all output cells before committing. Add a note at the top cell linking to `docs/architecture.md`.
  - Acceptance: `jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True` run on the file produces no diff in cell outputs.

### 3.4 — Add LangSmith Integration (Optional Complement to Langfuse)

- [ ] **`nim/client.py`** — Add optional LangSmith tracing via `LANGCHAIN_TRACING_V2=true` environment variable support.
  - Acceptance: When `LANGCHAIN_TRACING_V2=true` is set in `.env`, traces appear in LangSmith. No code changes needed if env var is documented in `.env.template`.
- [ ] **`.env.template`** — Add `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` as optional variables with comments.

### 3.5 — ISV Customization Guide

- [ ] **`docs/isv-customization.md`** (create) — Document how an ISV would fork this repo, add a new agent to `agents/`, register it in `configs/agents.yaml`, and wire it into the LangGraph graph.
  - Acceptance: Document exists and covers: 1) creating a new StructuredTool, 2) registering in YAML, 3) adding a node to the graph, 4) writing a test.

---

## Cross-Repo Tasks

These tasks apply identically across `nvidia-nim-agent-toolkit`, `enterprise-rag-pipeline`, and `multi-agent-reference-architecture`.

- [ ] **All repos** — Confirm `langfuse>=2.0.0` is in each repo's `requirements.txt`.
- [ ] **All repos** — Confirm `docker-compose.yml` exists at each repo root.
- [ ] **All repos** — Confirm `.env.template` documents `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
- [ ] **All repos** — Confirm `ruff check . && mypy .` passes clean in each repo independently.
- [ ] **All repos** — Confirm LangGraph state is `TypedDict` (not `dict`) in each graph file.
