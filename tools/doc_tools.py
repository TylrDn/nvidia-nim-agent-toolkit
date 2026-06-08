"""Document store StructuredTool wrappers for the Doc agent."""
from __future__ import annotations

import os
from typing import Optional

from langchain.tools import StructuredTool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            openai_api_base=NIM_BASE_URL,
            openai_api_key=NIM_API_KEY,
        )
        if os.path.exists(FAISS_INDEX_PATH):
            _vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            # Empty index for demo — populate via ingest pipeline
            _vectorstore = FAISS.from_texts(["placeholder"], embeddings)
    return _vectorstore


class SearchInput(BaseModel):
    query: str = Field(description="Semantic search query")
    k: Optional[int] = Field(default=5, description="Number of results to return")


def _search_docs(query: str, k: int = 5) -> str:
    try:
        vs = _get_vectorstore()
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] ({source})\n{doc.page_content[:800]}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Document search error: {e}"


def get_doc_tools() -> list:
    return [
        StructuredTool.from_function(
            func=_search_docs,
            name="search_documents",
            description="Semantic search over the document store. Returns top-k relevant passages.",
            args_schema=SearchInput,
        ),
    ]
