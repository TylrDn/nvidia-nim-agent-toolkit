"""NIM OpenAI-compatible API client wrapper.

Routes all LangChain LLM calls through NVIDIA NIM inference microservices
using the OpenAI-compatible REST interface.
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Iterator

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
DEFAULT_MODEL = os.getenv("NIM_DEFAULT_MODEL", "meta/llama-3.1-70b-instruct")


class NIMClient:
    """Thin wrapper around NIM's OpenAI-compatible endpoint.

    Provides a LangChain-compatible ChatOpenAI instance plus helpers
    for health checking and streaming.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        streaming: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.streaming = streaming
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazily initialise and cache the LangChain LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                openai_api_key=NIM_API_KEY,
                openai_api_base=NIM_BASE_URL,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.timeout,
                streaming=self.streaming,
            )
        return self._llm

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """Invoke the NIM endpoint synchronously."""
        start = time.perf_counter()
        result = self.llm.invoke(messages)
        elapsed = time.perf_counter() - start
        logger.debug("NIM invoke: model=%s latency=%.3fs", self.model, elapsed)
        return result

    def stream(self, messages: list[BaseMessage]) -> Iterator[str]:
        """Stream token chunks from the NIM endpoint."""
        for chunk in self.llm.stream(messages):
            yield chunk.content

    def health_check(self) -> dict[str, Any]:
        """Probe the NIM /v1/models endpoint and return status info."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{NIM_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {NIM_API_KEY}"},
                )
                resp.raise_for_status()
                models = [m["id"] for m in resp.json().get("data", [])]
                return {"status": "ok", "models": models}
        except Exception as exc:  # noqa: BLE001
            logger.error("NIM health check failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NIMClient":
        """Construct a NIMClient from a YAML-loaded config dict."""
        return cls(
            model=config.get("model", DEFAULT_MODEL),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 2048),
            timeout=config.get("timeout", 60.0),
            streaming=config.get("streaming", False),
        )


# Module-level default client — importable directly.
default_client = NIMClient()
