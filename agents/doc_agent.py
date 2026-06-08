"""Document store retrieval agent using FAISS for local dev and pgvector for prod."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain.tools import StructuredTool
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from nim.client import NIMClient

_NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
_NIM_API_KEY = os.getenv("NIM_API_KEY", "not-used")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
_VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", ".vector_store")

_embeddings = OpenAIEmbeddings(
    base_url=_NIM_BASE_URL,
    api_key=_NIM_API_KEY,
    model=_EMBEDDING_MODEL,
)


def _get_vectorstore() -> FAISS | None:
    p = Path(_VECTOR_INDEX_PATH)
    if p.exists():
        return FAISS.load_local(str(p), _embeddings,
                                allow_dangerous_deserialization=True)
    return None


class DocQueryInput(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=4, description="Number of chunks to retrieve")


def _retrieve_docs(query: str, top_k: int = 4) -> dict[str, Any]:
    vs = _get_vectorstore()
    if vs is None:
        return {"error": "Vector store not initialised. Run ingestion first."}
    docs = vs.similarity_search(query, k=top_k)
    return {"chunks": [d.page_content for d in docs]}


_doc_tool = StructuredTool.from_function(
    func=_retrieve_docs,
    name="retrieve_docs",
    description="Search the document store and return the most relevant text chunks.",
    args_schema=DocQueryInput,
)

_llm = NIMClient().get_llm().bind_tools([_doc_tool])

_SYSTEM = (
    "You are a document-retrieval agent. Use the retrieve_docs tool to find "
    "relevant passages, then synthesise a concise answer grounded in the retrieved text."
)


def run(task_description: str) -> dict[str, Any]:
    """Execute the doc retrieval agent for a single task."""
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=task_description),
    ]
    response = _llm.invoke(messages)
    return {"output": response.content, "tool": "doc"}
