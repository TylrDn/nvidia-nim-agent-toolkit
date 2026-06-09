"""REST API StructuredTool wrappers for the API agent."""
from __future__ import annotations

import json
from typing import Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class HTTPGetInput(BaseModel):
    url: str = Field(description="Full URL to GET")
    headers: Optional[dict] = Field(default=None, description="Optional request headers")
    params: Optional[dict] = Field(default=None, description="Optional query parameters")


class HTTPPostInput(BaseModel):
    url: str = Field(description="Full URL to POST")
    payload: dict = Field(description="JSON body to send")
    headers: Optional[dict] = Field(default=None, description="Optional request headers")


def _http_get(url: str, headers: dict | None = None, params: dict | None = None) -> str:
    try:
        r = httpx.get(url, headers=headers or {}, params=params or {}, timeout=15)
        r.raise_for_status()
        try:
            return json.dumps(r.json(), indent=2)
        except Exception:
            return r.text[:4000]
    except Exception as e:
        return f"HTTP GET error: {e}"


def _http_post(url: str, payload: dict, headers: dict | None = None) -> str:
    try:
        r = httpx.post(url, json=payload, headers=headers or {}, timeout=15)
        r.raise_for_status()
        try:
            return json.dumps(r.json(), indent=2)
        except Exception:
            return r.text[:4000]
    except Exception as e:
        return f"HTTP POST error: {e}"


def get_api_tools() -> list[StructuredTool]:
    """Return the REST API StructuredTools for the API agent."""
    return [
        StructuredTool.from_function(
            func=_http_get,
            name="http_get",
            description="Make an HTTP GET request to a URL. Returns the JSON or text response.",
            args_schema=HTTPGetInput,
        ),
        StructuredTool.from_function(
            func=_http_post,
            name="http_post",
            description="Make an HTTP POST request with a JSON payload. Returns the response.",
            args_schema=HTTPPostInput,
        ),
    ]
