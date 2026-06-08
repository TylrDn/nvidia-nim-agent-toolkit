"""Document retrieval tool wrappers as LangChain StructuredTools.

Provides semantic search over a configured vector store (pgvector by default).
Swap the backend by setting VECTOR_BACKEND=milvus and the relevant env vars.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql://postgres:password@localhost:5432/vectordb")
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NIM_API_KEY", "")


class RetrieveInput(BaseModel):
    query: str = Field(description="Natural language search query")
    top_k: int = Field(default=5, description="Number of passages to retrieve")
    collection: Optional[str] = Field(
        default=None, description="Collection/namespace to search. Defaults to env config."
    )


def _get_embeddings():
    """Return a LangChain embeddings instance pointed at NIM."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=NIM_API_KEY,
        openai_api_base=f"{NIM_BASE_URL}/embeddings",
    )


def retrieve_documents(query: str, top_k: int = 5, collection: Optional[str] = None) -> str:
    """Retrieve top-k relevant passages from the vector store."""
    col = collection or COLLECTION_NAME
    try:
        embeddings = _get_embeddings()

        if VECTOR_BACKEND == "pgvector":
            from langchain_community.vectorstores import PGVector

            store = PGVector(
                connection_string=PGVECTOR_URL,
                embedding_function=embeddings,
                collection_name=col,
            )
        elif VECTOR_BACKEND == "milvus":
            from langchain_community.vectorstores import Milvus

            store = Milvus(
                embedding_function=embeddings,
                collection_name=col,
            )
        else:
            return f"Unsupported vector backend: {VECTOR_BACKEND}"

        docs = store.similarity_search(query, k=top_k)
        if not docs:
            return "No relevant documents found."

        passages = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            passages.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n".join(passages)

    except Exception as exc:  # noqa: BLE001
        logger.error("Document retrieval error: %s", exc)
        return f"Retrieval error: {exc}"


def get_doc_tools() -> list[StructuredTool]:
    """Return the list of document retrieval StructuredTools."""
    return [
        StructuredTool.from_function(
            func=retrieve_documents,
            name="retrieve_documents",
            description=(
                "Retrieve relevant document passages from the vector store using semantic search. "
                "Use this before answering any factual question."
            ),
            args_schema=RetrieveInput,
        ),
    ]
