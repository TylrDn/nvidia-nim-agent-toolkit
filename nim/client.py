"""NIM client — wraps NVIDIA NIM inference endpoints.

Langfuse tracing is supported via the LangChain path (`as_langchain_llm()`).
The raw OpenAI SDK path (`chat()`, `stream_chat()`, `chat_with_tools()`) does
NOT support LangChain callbacks; set LANGFUSE_PUBLIC_KEY to enable tracing
only when calling through `as_langchain_llm()`.
"""

from __future__ import annotations

import os
from typing import Any, Generator, Iterator

from langchain_openai import ChatOpenAI
from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional Langfuse import
# ---------------------------------------------------------------------------
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:  # langfuse not installed — tracing silently disabled
    LangfuseCallbackHandler = None  # type: ignore[assignment,misc]
    LANGFUSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_langfuse_handler() -> "LangfuseCallbackHandler | None":
    """Return a Langfuse CallbackHandler if credentials are configured.

    Returns ``None`` (and never raises) when:
    - ``langfuse`` package is not installed
    - ``LANGFUSE_PUBLIC_KEY`` env var is not set
    """
    if not LANGFUSE_AVAILABLE:
        return None
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return LangfuseCallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return None


# ---------------------------------------------------------------------------
# NIMClient
# ---------------------------------------------------------------------------

class NIMClient:
    """Client for NVIDIA NIM inference endpoints.

    Environment variables
    ---------------------
    NVIDIA_API_KEY : str
        API key for NVIDIA NIM / NGC.
    NIM_BASE_URL : str
        Base URL for the NIM endpoint (default: ``https://integrate.api.nvidia.com/v1``).
    LANGFUSE_PUBLIC_KEY : str, optional
        Enable Langfuse tracing on LangChain calls.
    LANGFUSE_SECRET_KEY : str, optional
        Langfuse secret key.
    LANGFUSE_HOST : str, optional
        Langfuse server URL (default: ``https://cloud.langfuse.com``).
    """

    def __init__(
        self,
        model: str = "meta/llama-3.1-8b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv(
            "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")

    # ------------------------------------------------------------------
    # LangChain integration (Langfuse tracing supported here)
    # ------------------------------------------------------------------

    def as_langchain_llm(self, **kwargs: Any) -> ChatOpenAI:
        """Return a LangChain ``ChatOpenAI`` instance pointed at the NIM endpoint.

        Langfuse tracing is automatically attached as a callback when
        ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set.
        """
        handler = _get_langfuse_handler()
        callbacks = [handler] if handler else []

        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            callbacks=callbacks,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Raw OpenAI SDK methods
    # NOTE: Langfuse LangChain callbacks are NOT supported on these paths.
    #       Use as_langchain_llm() for traced calls.
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> OpenAI:
        """Return a raw OpenAI SDK client pointed at the NIM endpoint."""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat request and return the assistant's message content.

        Note: Langfuse tracing is **not** available on this raw SDK path.
        Switch to ``as_langchain_llm()`` for traced inference.
        """
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a chat response, yielding content chunks as they arrive.

        Note: Langfuse tracing is **not** available on this raw SDK path.
        Switch to ``as_langchain_llm()`` for traced inference.
        """
        client = self._get_openai_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request with tool/function definitions.

        Returns the raw ``choices[0].message`` as a dict so callers can
        inspect ``tool_calls`` and ``content``.

        Note: Langfuse tracing is **not** available on this raw SDK path.
        Switch to ``as_langchain_llm()`` for traced inference.
        """
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        )
        message = response.choices[0].message
        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "role": message.role,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client_instance: NIMClient | None = None


def get_client(
    model: str = "meta/llama-3.1-8b-instruct",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> NIMClient:
    """Return (or lazily create) the module-level ``NIMClient`` singleton.

    Subsequent calls with the same arguments return the cached instance.
    Pass different arguments to force a new instance by constructing
    ``NIMClient(...)`` directly.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = NIMClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    return _client_instance
