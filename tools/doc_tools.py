"""Standalone document retrieval tools importable outside the agent context."""
from __future__ import annotations

from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class DocRetrieveInput(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=4)


def _stub_retrieve(query: str, top_k: int = 4) -> dict[str, Any]:
    """Stub — replace with live vectorstore in agents/doc_agent.py."""
    return {"chunks": [f"[stub chunk {i} for: {query}]" for i in range(top_k)]}


DOC_RETRIEVE_TOOL = StructuredTool.from_function(
    func=_stub_retrieve,
    name="retrieve_docs",
    description="Retrieve relevant document chunks from the vector store.",
    args_schema=DocRetrieveInput,
)
