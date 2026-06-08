---
name: add-langfuse-tracing
description: Invoke this agent when Langfuse tracing is missing from any LLM call path in nim/client.py, agents/, or orchestrator/nodes/ — specifically when CallbackHandler is not being passed to ChatOpenAI or chain invocations.
model: inherit
readonly: false
---

# Add Langfuse Tracing to nvidia-nim-agent-toolkit

## Objective

Instrument every LLM call path in this repository with Langfuse tracing using `langfuse.callback.CallbackHandler`. Tracing must be present in the NIM client, all agent wrappers, and all orchestrator nodes. When complete, every call to a language model will emit a trace to Langfuse.

## Context

This repo uses `langchain_openai.ChatOpenAI` pointed at NVIDIA NIM endpoints. Currently, `CallbackHandler` is not instantiated or passed to any LLM construction. The fix must be applied consistently across all files that construct `ChatOpenAI` or invoke chains.

## Files to Touch

1. `nim/client.py` — Add `get_langfuse_handler()` factory function and pass handler to `ChatOpenAI` constructor.
2. `agents/api_agent.py` — Import and apply handler.
3. `agents/sql_agent.py` — Import and apply handler.
4. `agents/doc_agent.py` — Import and apply handler.
5. `orchestrator/nodes/planner.py` — Apply handler to any LLM invocation.
6. `orchestrator/nodes/executor.py` — Apply handler to any LLM invocation.
7. `orchestrator/nodes/reviewer.py` — Apply handler to any LLM invocation.
8. `.env.template` — Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
9. `requirements.txt` — Add `langfuse>=2.0.0` if not present.
10. `tests/conftest.py` — Add a `mock_langfuse` fixture that patches `langfuse.callback.CallbackHandler` to prevent network calls in tests.

## Step-by-Step Instructions

### Step 1 — Add factory function in `nim/client.py`

Add the following near the top of the module, after existing imports:

```python
import os
import logging
from langfuse.callback import CallbackHandler

logger = logging.getLogger(__name__)


def get_langfuse_handler() -> CallbackHandler:
    """Return a configured Langfuse CallbackHandler.

    Reads credentials from environment variables:
        LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

    Returns:
        CallbackHandler: A Langfuse callback handler ready for use with LangChain.

    Raises:
        No exception — missing keys are logged as a warning and empty strings are used,
        which will cause Langfuse to operate in no-op mode.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set. "
            "Langfuse tracing will be disabled."
        )

    return CallbackHandler(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
```

### Step 2 — Update `NIMClient` to inject the handler

In the `NIMClient` class (or wherever `ChatOpenAI` is constructed in `nim/client.py`), update the construction:

```python
from nim.client import get_langfuse_handler  # if in a different file, adjust import

langfuse_handler = get_langfuse_handler()

llm = ChatOpenAI(
    model=self.config["model"],
    base_url=os.environ["NIM_BASE_URL"],
    api_key=os.environ["NIM_API_KEY"],
    temperature=0,
    callbacks=[langfuse_handler],
)
```

Do not construct `ChatOpenAI` without `callbacks=[langfuse_handler]`.

### Step 3 — Update each agent file

For `agents/api_agent.py`, `agents/sql_agent.py`, `agents/doc_agent.py`:

At the top of each file, add:
```python
from nim.client import get_langfuse_handler
```

Find any `ChatOpenAI(...)` or `chain.invoke(...)` calls. If the `ChatOpenAI` instance is already created in `nim/client.py` and passed in, verify the handler is flowing through. If the agent file independently constructs an LLM, add `callbacks=[get_langfuse_handler()]` to that construction.

For chain invocations (e.g., `chain.invoke(inputs)`), update to:
```python
chain.invoke(inputs, config={"callbacks": [get_langfuse_handler()]})
```

### Step 4 — Update orchestrator nodes

For `orchestrator/nodes/planner.py`, `orchestrator/nodes/executor.py`, `orchestrator/nodes/reviewer.py`:

Each node that invokes an LLM must pass the handler. Pattern:

```python
from nim.client import get_langfuse_handler

async def planner_node(state: AgentState) -> dict:
    langfuse_handler = get_langfuse_handler()
    result = await llm.ainvoke(
        state["messages"],
        config={"callbacks": [langfuse_handler]},
    )
    return {"plan": result.content}
```

### Step 5 — Update `.env.template`

Add these lines to `.env.template`:
```
# Langfuse Observability
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Step 6 — Update `requirements.txt`

Add:
```
langfuse>=2.0.0
```

Verify there is no version pin conflict with existing langchain packages.

### Step 7 — Add mock fixture in `tests/conftest.py`

```python
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=False)
def mock_langfuse(monkeypatch):
    """Patch Langfuse CallbackHandler to prevent network calls in tests."""
    with patch("langfuse.callback.CallbackHandler") as mock_handler:
        mock_handler.return_value = MagicMock()
        yield mock_handler
```

Apply `mock_langfuse` fixture to any test that triggers LLM construction.

## Acceptance Criteria

- [ ] `get_langfuse_handler()` exists in `nim/client.py` and is the single source of truth for handler creation.
- [ ] Every `ChatOpenAI(...)` construction in the codebase includes `callbacks=[langfuse_handler]`.
- [ ] Every `chain.invoke()` or `llm.ainvoke()` call in orchestrator nodes passes `config={"callbacks": [langfuse_handler]}`.
- [ ] `.env.template` contains all three Langfuse environment variables.
- [ ] `langfuse>=2.0.0` is present in `requirements.txt`.
- [ ] `tests/conftest.py` has `mock_langfuse` fixture.
- [ ] `pytest` passes with no new failures after changes.
- [ ] `ruff check . --fix && mypy .` pass clean.
- [ ] Running `grep -r "ChatOpenAI" . --include="*.py"` shows every result has `callbacks=` in its constructor or in the `.invoke()` call chain.
