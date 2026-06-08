"""FastAPI server — exposes the multi-agent pipeline over HTTP.

Endpoints:
  POST /v1/run     — run a full agentic pipeline for a given intent
  GET  /v1/health  — NIM and service health check
  GET  /           — liveness probe
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NVIDIA NIM Agent Toolkit",
    description="Multi-agent coordination powered by NVIDIA NIM inference microservices.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    intent: str = Field(..., description="User intent / query to execute")
    model: str | None = Field(default=None, description="Override default NIM model")


class RunResponse(BaseModel):
    intent: str
    final_answer: str
    task_count: int
    reviewer_score: float
    latency_sec: float


@app.get("/", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/health", tags=["health"])
def health_check() -> dict[str, Any]:
    from nim.client import NIMClient

    client = NIMClient()
    nim_status = client.health_check()
    return {"service": "ok", "nim": nim_status}


@app.post("/v1/run", response_model=RunResponse, tags=["agent"])
def run_agent(request: RunRequest) -> RunResponse:
    from nim.client import NIMClient
    from orchestrator.graph import run as run_pipeline

    client = NIMClient(model=request.model) if request.model else NIMClient()

    start = time.perf_counter()
    try:
        state = run_pipeline(intent=request.intent, client=client)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = round(time.perf_counter() - start, 3)
    return RunResponse(
        intent=request.intent,
        final_answer=state.get("final_answer", ""),
        task_count=len(state.get("task_list", [])),
        reviewer_score=state.get("reviewer_score", 0.0),
        latency_sec=latency,
    )
