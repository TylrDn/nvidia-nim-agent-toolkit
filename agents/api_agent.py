"""REST API tool-calling agent."""
from __future__ import annotations

import os
from typing import Any

import httpx
from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from nim.client import NIMClient


class APIQueryInput(BaseModel):
    url: str = Field(..., description="Full URL to GET")
    params: dict[str, str] = Field(default_factory=dict,
                                   description="Query parameters")


def _http_get(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        resp = httpx.get(url, params=params or {}, timeout=15)
        resp.raise_for_status()
        return {"status": resp.status_code, "body": resp.json()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


_api_tool = StructuredTool.from_function(
    func=_http_get,
    name="http_get",
    description="Perform an HTTP GET request and return the JSON response.",
    args_schema=APIQueryInput,
)

_llm = NIMClient().get_llm().bind_tools([_api_tool])

_SYSTEM = (
    "You are an API agent. Use the http_get tool to answer the user's request. "
    "Always return structured JSON in your final answer."
)


def run(task_description: str) -> dict[str, Any]:
    """Execute the API agent for a single task."""
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=task_description),
    ]
    response = _llm.invoke(messages)
    return {"output": response.content, "tool": "api"}
