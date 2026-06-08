"""Standalone HTTP tool wrappers importable outside the agent context."""
from __future__ import annotations

from typing import Any

import httpx
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class HTTPGetInput(BaseModel):
    url: str = Field(..., description="Full URL to GET")
    params: dict[str, str] = Field(default_factory=dict)


def http_get(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        r = httpx.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        return {"status": r.status_code, "body": r.json()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


HTTP_GET_TOOL = StructuredTool.from_function(
    func=http_get,
    name="http_get",
    description="Perform a GET request and return JSON response.",
    args_schema=HTTPGetInput,
)
