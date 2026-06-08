"""NIM OpenAI-compatible API wrapper with Langfuse tracing."""
from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

load_dotenv()


class NIMClient:
    """Thin wrapper around the NVIDIA NIM OpenAI-compatible endpoint.

    Exposes a LangChain-compatible ``ChatOpenAI`` instance that routes
    all inference through the configured NIM base URL.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = os.environ["NIM_BASE_URL"]
        self.api_key = os.environ["NIM_API_KEY"]
        self.model = model or os.environ.get("NIM_DEFAULT_MODEL", "meta/llama3-70b-instruct")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._langfuse_handler = self._build_langfuse_handler()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_llm(self) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance pointed at NIM."""
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            callbacks=[self._langfuse_handler] if self._langfuse_handler else [],
        )

    def list_models(self) -> list[dict[str, Any]]:
        """Return available models from the NIM endpoint."""
        resp = httpx.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not (public_key and secret_key):
            return None
        return LangfuseCallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
