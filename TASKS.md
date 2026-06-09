# TASKS — nvidia-nim-agent-toolkit

**Completion:** 100%
**Last Audit:** 2026-06-09
**Repo Role:** NVIDIA SA reference implementation for NIM-powered multi-agent systems using LangGraph.

> All Priority 1–3 items are complete. The full gate passes locally:
> `ruff check .`, `mypy nim/ orchestrator/ agents/ tools/ api/ configs/ core/`,
> and `pytest -m "not integration" --cov --cov-fail-under=80` (88% coverage).

---

## Priority 1 — Critical Gaps (Done)

### 1.1 — Langfuse Tracing on All LLM Call Paths

- [x] **`nim/client.py`** — Public `get_langfuse_handler()` / `get_callbacks()`; `CallbackHandler` attached in `as_langchain_llm()`.
- [x] **`agents/api_agent.py`** — Uses `agents/base.py`, which attaches callbacks to every `ainvoke`.
- [x] **`agents/sql_agent.py`** — Same.
- [x] **`agents/doc_agent.py`** — Same.
- [x] **`orchestrator/nodes/planner.py`** — `config={"callbacks": get_callbacks()}` on every async LLM call.
- [x] **`orchestrator/nodes/executor.py`** — Dispatches to traced sub-agents.
- [x] **`orchestrator/nodes/reviewer.py`** — Callbacks attached.
- [x] **`.env.template`** — `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` documented.
- [x] **`requirements.txt`** — `langfuse==2.60.10` pinned.
- [x] **`tests/conftest.py`** — `mock_langfuse` fixture disables tracing in tests.

### 1.2 — Root-Level docker-compose.yml

- [x] **`docker-compose.yml`** — Root compose with `dev` and `full` profiles, `env_file: .env`, build from `deploy/Dockerfile`.
- [x] **`README.md`** — Quickstart uses `docker compose --profile dev up` from the repo root.

---

## Priority 2 — Polish (Done)

- [x] **2.1 `.dockerignore`** — Excludes VCS, caches, env, notebooks, docs, tests.
- [x] **2.2 Health endpoint** — `/health` returns `{"status": "ok", "nim_reachable": bool}`; `nim/health_check.py` exposes `async def check_nim_health() -> bool`.
- [x] **2.3 Structured logging** — `core/logging.py` emits JSON when `LOG_FORMAT=json`; wired into the API lifespan.
- [x] **2.4 Pinned dependencies** — `requirements.txt` pinned from a verified environment with a regeneration note.
- [x] **2.5 Validate `agents.yaml`** — `configs/loader.py` validates required agents and raises `ValueError` before any LLM call.
- [x] **2.6 CI coverage gate** — `--cov-fail-under=80` in `.github/workflows/ci.yml`.

---

## Priority 3 — Enhancements (Done)

- [x] **3.1 Async batch inference** — `NIMClient.batch_chat_completion()` via `asyncio.gather`, unit tested.
- [x] **3.2 Eval improvements** — `evals/agent_eval.py` uses logging and an `--output-file` argparse option.
- [x] **3.3 Notebook cleanup** — `notebooks/quickstart.ipynb` updated to the current API with a link to `docs/architecture.md`.
- [x] **3.4 LangSmith integration** — Optional via `LANGCHAIN_TRACING_V2`; documented in `.env.template`.
- [x] **3.5 ISV customization guide** — `docs/isv-customization.md` covers tool, YAML, dispatch, and test.

---

## Cross-Repo Tasks (this repo)

- [x] `langfuse` pinned in `requirements.txt`.
- [x] Root `docker-compose.yml` present.
- [x] `.env.template` documents the Langfuse keys.
- [x] `ruff check .` and `mypy` pass clean.
- [x] LangGraph state is a `TypedDict` (`orchestrator/state.py`).

> Portfolio-wide confirmation across all repos is tracked in
> `mission-control/NOTION.md`.
