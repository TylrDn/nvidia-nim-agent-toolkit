"""FastAPI server exposing the NIM multi-agent toolkit."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from configs.loader import validate_agents_config
from core.logging import configure_logging
from nim.client import get_client
from nim.health_check import check_nim_health
from orchestrator.graph import run_agent_async

load_dotenv()
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("api.access")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate configuration at startup so misconfiguration fails fast.

    Raises:
        ValueError: If ``agents.yaml`` is missing required agents or invalid.
    """
    configure_logging()
    validate_agents_config()
    logger.info("NIM Agent Toolkit API ready")
    yield


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with a request id and latency, and echo the id back."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Attach a request id, time the handler, and log the outcome."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        access_logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response


app = FastAPI(
    title="NVIDIA NIM Agent Toolkit",
    description="Multi-agent coordination system powered by NVIDIA NIM + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(StructuredLoggingMiddleware)


class QueryRequest(BaseModel):
    """Request body for the ``/query`` endpoint."""

    query: str
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response body for the ``/query`` endpoint."""

    thread_id: str
    answer: str


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness/readiness probe including NIM reachability."""
    return {"status": "ok", "nim_reachable": await check_nim_health()}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the multi-agent pipeline for a single query."""
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        answer = await run_agent_async(request.query, thread_id=thread_id)
    except Exception as exc:  # noqa: BLE001 - return a clean 500 to the client
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(thread_id=thread_id, answer=answer)


@app.get("/models")
async def list_models() -> dict[str, list[str]]:
    """List available NIM models."""
    try:
        models = await get_client().list_models()
    except Exception as exc:  # noqa: BLE001 - upstream NIM failure → 502
        logger.exception("Failed to list models")
        raise HTTPException(status_code=502, detail=f"NIM unavailable: {exc}") from exc
    return {"models": models}
