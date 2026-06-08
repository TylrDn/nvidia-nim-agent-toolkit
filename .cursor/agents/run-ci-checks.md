---
name: run-ci-checks
description: Invoke this agent to run the full local CI check suite — linting, type-checking, and tests — and fix any failures before pushing to GitHub, or when the GitHub Actions CI workflow is failing and you need to diagnose and resolve issues.
model: inherit
readonly: false
---

# Run CI Checks — nvidia-nim-agent-toolkit

## Objective

Execute the complete local CI pipeline (ruff, mypy, pytest) in the correct order, interpret any failures, apply fixes, and confirm the suite passes clean. This mirrors the checks run in `.github/workflows/ci.yml` so that CI never fails after a push.

## Context

The CI pipeline for this repo runs three stages: lint (`ruff`), type-check (`mypy`), and test (`pytest`). All three must pass before a PR can be merged. This agent runs them locally, finds root causes of failures, and fixes them.

## Pre-flight Checks

Before running anything, verify the environment:

1. Confirm Python 3.11 is active: `python --version`
2. Confirm virtual environment is activated or dependencies are installed: `pip list | grep langchain`
3. If dependencies are missing, install them: `pip install -e ".[dev]"` or `pip install -r requirements.txt`
4. Confirm `.env` exists (copy from `.env.template` if not): the test suite may need env vars, and some tests may rely on `load_dotenv()`.

## Step-by-Step Instructions

### Stage 1 — Ruff Lint

Run:
```bash
ruff check . --fix
```

Expected output: no errors after `--fix`. If errors remain after `--fix`:
- `E501` (line too long): Manually wrap the offending line to ≤100 chars.
- `F401` (unused import): Remove the import or add `# noqa: F401` only if the import is needed for side effects (document why).
- `F811` (redefinition): Identify which definition is the correct one and remove the duplicate.
- `UP` (pyupgrade) rules: These are auto-fixed; if they appear, re-run `ruff check . --fix`.

After fixing, run `ruff check .` (no `--fix`) and confirm zero errors.

### Stage 2 — Mypy Type Check

Run:
```bash
mypy . --ignore-missing-imports --python-version 3.11
```

Common failure patterns and fixes:

| Error pattern | Fix |
|---|---|
| `error: Function is missing a return type annotation` | Add `-> ReturnType:` to the function signature |
| `error: Argument 1 to "X" has incompatible type "Y"; expected "Z"` | Fix the type mismatch or add an explicit cast |
| `error: Item "None" of "X \| None" has no attribute "Y"` | Add a `None` guard: `if x is not None:` |
| `error: Module "langfuse" has no attribute "callback"` | Verify `langfuse>=2.0.0` is installed; add `# type: ignore[attr-defined]` with a comment only as last resort |
| `error: Cannot find implementation or library stub for module named "yaml"` | Install `types-PyYAML`: `pip install types-PyYAML` |

After all fixes, rerun `mypy .` and confirm zero errors or only expected `[import-untyped]` notes.

### Stage 3 — Pytest

Run:
```bash
pytest tests/ -v --cov=. --cov-report=term-missing -m "not integration"
```

The `-m "not integration"` flag skips tests marked `@pytest.mark.integration` that require live NIM endpoints.

If tests fail:

1. Read the full traceback. Identify the file and line.
2. Check if the failure is a mock setup issue (missing `mock_langfuse` fixture — see `add-langfuse-tracing` agent).
3. Check if the failure is an import error — this usually means a missing dependency or circular import.
4. Check if the failure is an assertion error — read the expected vs actual values and trace back to the logic.
5. Fix the root cause. Do not comment out tests or use `pytest.skip()` without a documented reason.

For async test failures (`RuntimeError: no running event loop`), confirm the test function is decorated `@pytest.mark.asyncio` and `pytest-asyncio` is installed.

### Stage 4 — Report

After all three stages pass, output a summary:

```
CI Check Summary
================
Ruff:   PASSED (0 errors)
Mypy:   PASSED (0 errors)
Pytest: PASSED (X tests passed, Y warnings)
Coverage: Z% overall

Files modified during this run:
- list any files that were changed
```

If any stage cannot be made to pass (e.g., a pre-existing broken test that requires external service), document the blocker clearly:

```
BLOCKED: tests/test_nim_client.py::test_chat_completion_integration
Reason: Requires live NIM endpoint. Mark with @pytest.mark.integration and exclude from default run.
Suggested fix: Add @pytest.mark.integration decorator to the test function.
```

## Acceptance Criteria

- [ ] `ruff check .` exits with code 0 and outputs no errors.
- [ ] `mypy . --ignore-missing-imports --python-version 3.11` exits with code 0.
- [ ] `pytest tests/ -v -m "not integration"` exits with code 0.
- [ ] Test coverage for `nim/`, `orchestrator/`, `agents/` is ≥ 80%.
- [ ] No tests are skipped or commented out without a documented reason.
- [ ] A summary of all changes made during this run is provided.
- [ ] If the CI workflow YAML was the source of a failure, `.github/workflows/ci.yml` is updated with the fix and the fix is documented in the summary.
