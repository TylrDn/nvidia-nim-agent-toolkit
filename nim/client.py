"""NIM OpenAI-compatible API wrapper.

Routes all LangChain LLM calls through a NVIDIA NIM inference microservice
endpoint via its OpenAI-compatible REST interface.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class NIMConfig(BaseModel):
    base_url: str = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("NIM_API_KEY", "not-used")
    model: str = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 120


class NIMClient:
    """Thin wrapper that returns a LangChain ChatOpenAI wired to NIM."""

    def __init__(self, config: NIMConfig | None = None) -> None:
        self.config = config or NIMConfig()

    def get_llm(self, **overrides: Any) -> ChatOpenAI:
        """Return a ChatOpenAI instance pointing at the NIM endpoint."""
        cfg = self.config
        return ChatOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=overrides.get("model", cfg.model),
            temperature=overrides.get("temperature", cfg.temperature),
            max_tokens=overrides.get("max_tokens", cfg.max_tokens),
            timeout=overrides.get("timeout", cfg.timeout),
        )

    def health_check(self) -> dict[str, Any]:
        """Probe /health on the NIM container."""
        url = self.config.base_url.rstrip("/v1").rstrip("/") + "/health"
        try:
            resp = httpx.get(url, timeout=5)
            return {"status": "ok" if resp.status_code == 200 else "degraded",
                    "code": resp.status_code}
        except httpx.ConnectError:
            return {"status": "unreachable", "code": None}
