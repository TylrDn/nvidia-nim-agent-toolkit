"""NIM OpenAI-compatible API wrapper.

Wraps NVIDIA NIM endpoints behind a standard LangChain-compatible interface
so any LangChain LLM call routes through NIM without code changes.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class NIMClient:
    """Thin wrapper around NIM's OpenAI-compatible REST API.

    Usage::

        client = NIMClient(model="meta/llama-3.1-70b-instruct")
        llm = client.get_llm()
        response = llm.invoke("Hello from NIM")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model or os.environ["NIM_MODEL"]
        self.base_url = base_url or os.environ["NIM_BASE_URL"]
        self.api_key = api_key or os.environ["NIM_API_KEY"]
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance pointed at NIM."""
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Convenience one-shot invoke — returns string content."""
        llm = self.get_llm()
        response = llm.invoke(prompt, **kwargs)
        return response.content  # type: ignore[return-value]


def get_default_llm() -> ChatOpenAI:
    """Module-level helper — returns NIM LLM with env defaults."""
    return NIMClient().get_llm()
