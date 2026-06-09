"""NIM client — wraps NVIDIA NIM inference endpoints.

All LLM calls in this repository route through :class:`NIMClient`. Tracing is
enabled by attaching the Langfuse :class:`CallbackHandler` (via
:func:`get_callbacks`) to every LangChain invocation. The async raw-SDK paths
use ``openai.AsyncOpenAI`` and are wrapped with ``tenacity`` retries.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, APIError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = os.getenv("NIM_DEFAULT_MODEL", "meta/llama-3.1-70b-instruct")

# ---------------------------------------------------------------------------
# Optional Langfuse import — tracing degrades gracefully if unavailable.
# ---------------------------------------------------------------------------
try:
    from langfuse.callback import CallbackHandler

    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without langfuse
    CallbackHandler = None  # type: ignore[assignment,misc]
    LANGFUSE_AVAILABLE = False


class NIMClientError(Exception):
    """Base error for NIM client failures."""


class NIMRateLimitError(NIMClientError):
    """Raised when the NIM endpoint reports rate limiting."""


def get_langfuse_handler() -> "CallbackHandler | None":
    """Return a configured Langfuse callback handler, or ``None`` if disabled.

    Reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_HOST``
    from the environment. Logs a warning and returns ``None`` when the package
    is not installed or credentials are absent, so tracing never raises.

    Returns:
        CallbackHandler | None: Handler when configured, otherwise ``None``.
    """
    if not LANGFUSE_AVAILABLE:
        logger.warning("langfuse not installed — tracing disabled.")
        return None

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled."
        )
        return None

    return CallbackHandler(public_key=public_key, secret_key=secret_key, host=host)


def get_callbacks() -> list[Any]:
    """Return the list of active LangChain callbacks for an invocation.

    Returns:
        list[Any]: ``[langfuse_handler]`` when tracing is configured, else ``[]``.
    """
    handler = get_langfuse_handler()
    return [handler] if handler else []


def _resolve_api_key(explicit: str | None) -> str:
    """Resolve the NIM API key, preferring ``NIM_API_KEY``.

    Args:
        explicit: An explicitly passed key, if any.

    Returns:
        str: The resolved key (may be empty if unset).
    """
    return explicit or os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""


class NIMClient:
    """Client for NVIDIA NIM inference endpoints (OpenAI-compatible).

    Environment variables:
        NIM_API_KEY: API key for NVIDIA NIM (falls back to ``NVIDIA_API_KEY``).
        NIM_BASE_URL: Base URL for the NIM endpoint.
        LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST: Tracing.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize a NIM client.

        Args:
            model: NIM model name.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            base_url: Override for the NIM base URL.
            api_key: Override for the NIM API key.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv("NIM_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = _resolve_api_key(api_key)
        self._async_client: AsyncOpenAI | None = None

    # ------------------------------------------------------------------
    # LangChain integration (Langfuse tracing supported here)
    # ------------------------------------------------------------------

    def as_langchain_llm(self, **kwargs: Any) -> ChatOpenAI:
        """Return a ``ChatOpenAI`` pointed at NIM with Langfuse tracing attached.

        Args:
            **kwargs: Extra keyword arguments forwarded to ``ChatOpenAI``.

        Returns:
            ChatOpenAI: Configured LLM with ``callbacks`` set.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "callbacks": get_callbacks(),
            **kwargs,
        }
        return ChatOpenAI(**params)

    # ------------------------------------------------------------------
    # Async raw OpenAI SDK methods
    # ------------------------------------------------------------------

    def _get_async_client(self) -> AsyncOpenAI:
        """Return (and cache) the underlying ``AsyncOpenAI`` client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._async_client

    @retry(
        retry=retry_if_exception_type((APIError, NIMRateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat request and return the assistant's message content.

        Args:
            messages: OpenAI-style message dicts.
            temperature: Override the client default.
            max_tokens: Override the client default.
            **kwargs: Extra arguments forwarded to the API.

        Returns:
            str: The assistant message content.

        Raises:
            NIMRateLimitError: If the endpoint reports rate limiting.
            NIMClientError: On other API failures.
        """
        client = self._get_async_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                **kwargs,
            )
        except RateLimitError as exc:
            logger.error("NIM rate limit hit: %s", exc)
            raise NIMRateLimitError(str(exc)) from exc
        except APIError as exc:
            logger.error("NIM API error: %s", exc)
            raise
        return response.choices[0].message.content or ""

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat response, yielding content chunks as they arrive.

        Args:
            messages: OpenAI-style message dicts.
            temperature: Override the client default.
            max_tokens: Override the client default.
            **kwargs: Extra arguments forwarded to the API.

        Yields:
            str: Content deltas.
        """
        client = self._get_async_client()
        stream: Any = await client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    async def batch_chat_completion(
        self,
        batch: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[str]:
        """Run multiple chat completions concurrently.

        Args:
            batch: A list of message lists, one per request.
            **kwargs: Extra arguments forwarded to each ``chat_completion``.

        Returns:
            list[str]: Completion content in the same order as ``batch``.
        """
        tasks = [self.chat_completion(messages, **kwargs) for messages in batch]
        return await asyncio.gather(*tasks)

    @retry(
        retry=retry_if_exception_type(APIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def list_models(self) -> list[str]:
        """List model IDs available at the NIM endpoint.

        Returns:
            list[str]: Sorted model identifiers.

        Raises:
            NIMClientError: If the models endpoint is unreachable.
        """
        client = self._get_async_client()
        try:
            response = await client.models.list()
        except APIError as exc:
            logger.error("Failed to list NIM models: %s", exc)
            raise
        return sorted(model.id for model in response.data)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client_instance: NIMClient | None = None


def get_client(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> NIMClient:
    """Return (or lazily create) the module-level ``NIMClient`` singleton.

    The singleton is created on first call. For a client bound to a different
    model, construct ``NIMClient(model=...)`` directly.

    Args:
        model: NIM model name for the initial construction.
        temperature: Sampling temperature for the initial construction.
        max_tokens: Maximum completion tokens for the initial construction.
        **kwargs: Extra arguments forwarded to ``NIMClient``.

    Returns:
        NIMClient: The shared client instance.
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
