"""FastAPI server exposing the NIM multi-agent toolkit."""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nim.health_check import is_nim_ready
from orchestrator.graph import run_agent, build_graph

app = FastAPI(
    title="NVIDIA NIM Agent Toolkit",
    description="Multi-agent coordination system powered by NVIDIA NIM + LangGraph",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    thread_id: str
    answer: str


@app.get("/health")
def health() -> dict:
    nim_ok = is_nim_ready()
    return {"status": "ok", "nim_ready": nim_ok}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the multi-agent pipeline for a single query."""
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        answer = run_agent(request.query, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(thread_id=thread_id, answer=answer)


@app.get("/models")
def list_models() -> dict:
    """List available NIM models."""
    from nim.client import get_client
    try:
        models = get_client().list_models()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM unavailable: {e}")
    return {"models": models}
