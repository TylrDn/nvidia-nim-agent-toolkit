"""Typed loader and validator for ``configs/agents.yaml``.

Loads the agent persona configuration once at startup, validates it with
Pydantic, and exposes typed accessors so the rest of the codebase never reads
raw YAML or hardcodes model names, prompts, or tool lists.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "agents.yaml"

# Agents that must be present for the orchestrator to run.
REQUIRED_AGENTS = ("planner", "executor", "reviewer", "api_agent", "sql_agent", "doc_agent")


class AgentConfig(BaseModel):
    """Configuration for a single agent persona.

    Attributes:
        model: NIM model name, e.g. ``meta/llama-3.1-70b-instruct``.
        system_prompt: Persona and instructions for the agent.
        tools: Tool names this agent may call (empty for orchestrator nodes).
        max_iterations: Cap on the tool-calling loop.
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.
        description: Human-readable summary of the agent's role.
    """

    model: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = 5
    temperature: float = 0.0
    max_tokens: int = 1024
    description: str = ""


class RoutingConfig(BaseModel):
    """Reviewer-loop routing thresholds shared across the graph."""

    max_retries: int = 2
    reviewer_accept_threshold: float = 0.75
    fallback_agent: str = "doc_agent"


class AgentsConfig(BaseModel):
    """Top-level parsed representation of ``agents.yaml``."""

    default_agent_model: str
    agents: dict[str, AgentConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


_config: AgentsConfig | None = None


def load_agents_config(path: Path | None = None, force: bool = False) -> AgentsConfig:
    """Load and cache the agents configuration from YAML.

    Args:
        path: Path to the YAML file. Defaults to ``configs/agents.yaml``.
        force: Reload from disk even if a cached config exists.

    Returns:
        AgentsConfig: The parsed, validated configuration.

    Raises:
        ValueError: If the file is missing, unparseable, or fails validation.
    """
    global _config
    if _config is not None and not force:
        return _config

    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ValueError(f"Agents config not found at {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        logger.error("Failed to parse %s: %s", config_path, exc)
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    try:
        config = AgentsConfig.model_validate(raw)
    except ValidationError as exc:
        logger.error("Agents config validation failed: %s", exc)
        raise ValueError(f"Invalid agents config in {config_path}: {exc}") from exc

    _config = config
    return config


def validate_agents_config(path: Path | None = None) -> AgentsConfig:
    """Validate the config and assert all required agents are present.

    Intended to be called from the FastAPI lifespan so misconfiguration fails
    fast, before the first LLM call.

    Args:
        path: Path to the YAML file. Defaults to ``configs/agents.yaml``.

    Returns:
        AgentsConfig: The validated configuration.

    Raises:
        ValueError: If a required agent entry is missing.
    """
    config = load_agents_config(path, force=True)
    missing = [name for name in REQUIRED_AGENTS if name not in config.agents]
    if missing:
        raise ValueError(f"agents.yaml is missing required agents: {', '.join(missing)}")
    logger.info("Agents config validated: %d agents loaded", len(config.agents))
    return config


def get_agent_config(name: str) -> AgentConfig:
    """Return the configuration for a single agent by name.

    Args:
        name: Agent key as defined under ``agents:`` in the YAML.

    Returns:
        AgentConfig: The configuration for the requested agent.

    Raises:
        ValueError: If no agent with that name is configured.
    """
    config = load_agents_config()
    if name not in config.agents:
        raise ValueError(f"No agent named '{name}' in agents config")
    return config.agents[name]


def get_routing_config() -> RoutingConfig:
    """Return the shared reviewer-loop routing configuration.

    Returns:
        RoutingConfig: Retry caps and acceptance thresholds.
    """
    return load_agents_config().routing
