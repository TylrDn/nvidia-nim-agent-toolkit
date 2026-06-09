"""Unit tests for the agents config loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from configs.loader import (
    AgentConfig,
    get_agent_config,
    get_routing_config,
    load_agents_config,
    validate_agents_config,
)

VALID_YAML = """
default_agent_model: meta/llama-3.1-70b-instruct
agents:
  planner:
    model: m1
    system_prompt: plan
  executor:
    model: m2
    system_prompt: exec
  reviewer:
    model: m3
    system_prompt: review
  api_agent:
    model: m4
    system_prompt: api
    tools: [http_get]
    max_iterations: 3
  sql_agent:
    model: m5
    system_prompt: sql
  doc_agent:
    model: m6
    system_prompt: doc
routing:
  max_retries: 4
  reviewer_accept_threshold: 0.9
  fallback_agent: doc_agent
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(text)
    return path


def test_default_config_loads_all_agents() -> None:
    config = validate_agents_config()
    assert {"planner", "executor", "reviewer", "api_agent", "sql_agent", "doc_agent"} <= set(
        config.agents
    )


def test_get_agent_config_returns_typed_object() -> None:
    cfg = get_agent_config("api_agent")
    assert isinstance(cfg, AgentConfig)
    assert "http_get" in cfg.tools


def test_get_agent_config_unknown_raises() -> None:
    with pytest.raises(ValueError, match="No agent named"):
        get_agent_config("nonexistent_agent")


def test_validate_loads_custom_path(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID_YAML)
    config = load_agents_config(path, force=True)
    assert get_routing_config().max_retries == 4
    assert config.agents["api_agent"].max_iterations == 3


def test_missing_required_agent_raises(tmp_path: Path) -> None:
    text = VALID_YAML.replace(
        "  sql_agent:\n    model: m5\n    system_prompt: sql\n", ""
    )
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match="missing required agents"):
        validate_agents_config(path)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "default_agent_model: x\nagents: [unbalanced")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_agents_config(path, force=True)


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Reload the default config after each test so cache state doesn't leak."""
    yield
    validate_agents_config()
