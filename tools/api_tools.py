"""REST API tool wrappers as LangChain StructuredTools.

All tools are read-only by default. Configure base URLs and auth
via environment variables or the .env file.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import httpx
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.getenv("API_TOOL_TIMEOUT", "15"))
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


class HttpGetInput(BaseModel):
    url: str = Field(description="Full URL to GET")
    params: Optional[dict] = Field(default=None, description="Query parameters")
    bearer_token: Optional[str] = Field(default=None, description="Bearer token if required")


class HttpPostInput(BaseModel):
    url: str = Field(description="Full URL to POST")
    payload: dict = Field(description="JSON body payload")
    bearer_token: Optional[str] = Field(default=None, description="Bearer token if required")


def http_get(url: str, params: Optional[dict] = None, bearer_token: Optional[str] = None) -> str:
    """Perform an authenticated HTTP GET and return the response body."""
    headers = dict(DEFAULT_HEADERS)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.text
    except httpx.HTTPStatusError as exc:
        return f"HTTP error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:  # noqa: BLE001
        return f"Request failed: {exc}"


def http_post(url: str, payload: dict, bearer_token: Optional[str] = None) -> str:
    """Perform an authenticated HTTP POST and return the response body."""
    headers = dict(DEFAULT_HEADERS)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.text
    except httpx.HTTPStatusError as exc:
        return f"HTTP error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:  # noqa: BLE001
        return f"Request failed: {exc}"


def get_api_tools() -> list[StructuredTool]:
    """Return the list of API StructuredTools for agent use."""
    return [
        StructuredTool.from_function(
            func=http_get,
            name="http_get",
            description="Make an HTTP GET request to a REST API endpoint. Returns response body.",
            args_schema=HttpGetInput,
        ),
        StructuredTool.from_function(
            func=http_post,
            name="http_post",
            description="Make an HTTP POST request with a JSON payload. Returns response body.",
            args_schema=HttpPostInput,
        ),
    ]
