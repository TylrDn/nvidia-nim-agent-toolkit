"""FastAPI entrypoint — wraps the LangGraph multi-agent graph over HTTP."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from orchestrator.graph import graph
from orchestrator.state import AgentState

app = FastAPI(
    title="NVIDIA NIM Agent Toolkit",
    description="Multi-agent LangGraph orchestrator backed by NVIDIA NIM.",
    version="0.1.0",
)


class AgentRequest(BaseModel):
    user_request: str


class AgentResponse(BaseModel):
    final_answer: str
    results: list[dict]
    reviewer_score: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(req: AgentRequest) -> AgentResponse:
    initial_state: AgentState = {
        "messages": [],
        "user_request": req.user_request,
        "tasks": [],
        "current_task_idx": 0,
        "results": [],
        "reviewer_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
    }
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentResponse(
        final_answer=final_state["final_answer"],
        results=final_state["results"],
        reviewer_score=final_state["reviewer_score"],
    )
