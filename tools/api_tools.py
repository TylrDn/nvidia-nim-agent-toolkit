"""StructuredTool wrappers for REST API calls."""
from __future__ import annotations

from typing import Any

import httpx
from langchain.tools import StructuredTool
from pydantic import BaseModel


class GetRequestInput(BaseModel):
    url: str
    params: dict[str, str] = {}


def _http_get(url: str, params: dict[str, str] = {}) -> str:
    """Perform an HTTP GET and return the response body as a string."""
    resp = httpx.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.text[:4000]  # cap at 4k chars to stay within context


class PostRequestInput(BaseModel):
    url: str
    payload: dict[str, Any] = {}


def _http_post(url: str, payload: dict[str, Any] = {}) -> str:
    """Perform an HTTP POST with a JSON payload."""
    resp = httpx.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.text[:4000]


def get_api_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_http_get,
            name="http_get",
            description="Perform an HTTP GET request to a URL with optional query params.",
            args_schema=GetRequestInput,
        ),
        StructuredTool.from_function(
            func=_http_post,
            name="http_post",
            description="Perform an HTTP POST request with a JSON payload.",
            args_schema=PostRequestInput,
        ),
    ]
