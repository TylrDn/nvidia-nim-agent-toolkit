"""NIM OpenAI-compatible API client wrapper."""
from __future__ import annotations

import os
import time
from typing import Any, Iterator

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from nim.health_check import wait_for_nim


NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")


class NIMClient:
    """Thin wrapper around NVIDIA NIM that exposes an OpenAI-compatible interface.

    All LangChain LLM call paths route through this class so swapping the
    underlying NIM endpoint requires changing only ``config.yaml``.
    """

    def __init__(
        self,
        model: str = "meta/llama-3.1-70b-instruct",
        base_url: str = NIM_BASE_URL,
        api_key: str = NIM_API_KEY,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._openai = OpenAI(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # LangChain-compatible LLM (preferred call path)
    # ------------------------------------------------------------------

    def as_langchain_llm(self) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance pointed at NIM."""
        return ChatOpenAI(
            model=self.model,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.timeout,
        )

    # ------------------------------------------------------------------
    # Raw OpenAI SDK call paths
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a chat completion request to NIM."""
        return self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            **kwargs,
        )

    def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Yield token chunks from a streaming NIM completion."""
        response = self.chat(messages, stream=True, **kwargs)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Tool / function calling
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """NIM function-calling — passes OpenAI tool spec directly."""
        return self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], model: str = "nvidia/nv-embedqa-e5-v5") -> list[list[float]]:
        """Return dense embeddings via NIM embeddings endpoint."""
        resp = self._openai.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return available NIM model IDs."""
        resp = self._openai.models.list()
        return [m.id for m in resp.data]

    def __repr__(self) -> str:
        return f"NIMClient(model={self.model!r}, base_url={self.base_url!r})"


# Module-level singleton — import and use directly
_default_client: NIMClient | None = None


def get_client(model: str | None = None) -> NIMClient:
    """Return a shared NIMClient instance (lazy init)."""
    global _default_client
    if _default_client is None or model is not None:
        _default_client = NIMClient(model=model or "meta/llama-3.1-70b-instruct")
    return _default_client
