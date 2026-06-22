"""Shared Qdrant vector store (hybrid dense + sparse) for retrieval and ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_qdrant import QdrantVectorStore

_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    """Return a singleton ``QdrantVectorStore`` aligned with hybrid retrieval settings.

    Creates the collection on first use if it does not exist.
    Heavy imports (langchain_openai, qdrant_client, fastembed) happen here,
    not at module level, so tests can patch this function without triggering
    those imports.
    """
    global _store  # noqa: PLW0603
    if _store is not None:
        return _store

    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore as _QVS
    from langchain_qdrant import RetrievalMode
    from langchain_qdrant.fastembed_sparse import FastEmbedSparse

    from crag.config import settings

    client_options: dict[str, str] = {"url": settings.qdrant_url}
    if (settings.qdrant_api_key or "").strip():
        client_options["api_key"] = settings.qdrant_api_key.strip()
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )
    sparse = FastEmbedSparse()
    try:
        _store = _QVS.construct_instance(
            client_options=client_options,
            collection_name=settings.collection_name,
            embedding=embedding,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    except Exception as exc:
        from qdrant_client.http.exceptions import UnexpectedResponse

        if isinstance(exc, UnexpectedResponse) and exc.status_code == 404:
            raise RuntimeError(
                "Qdrant returned HTTP 404 — nothing at QDRANT_URL looks like the Qdrant REST API. "
                "For Qdrant Cloud use https://<cluster>.cloud.qdrant.io:6333 (TLS on port 6333 is required; "
                "omitting :6333 often hits port 443 and returns this 404). "
                "For local Docker: `docker compose up -d` then http://127.0.0.1:6333. "
                f"Configured QDRANT_URL after normalization: {settings.qdrant_url!r}."
            ) from exc
        raise
    return _store
