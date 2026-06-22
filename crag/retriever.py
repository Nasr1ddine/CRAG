"""Hybrid (BM25 + dense) retrieval against Qdrant Cloud."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from crag.config import settings
from crag.qdrant_store import get_vector_store

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 3_000


class HybridRetriever:
    """Retrieves documents from Qdrant using hybrid BM25 + dense search."""

    def retrieve(self, query: str) -> list[Document]:
        """Run hybrid search and return top-k documents.

        Args:
            query: Natural-language search query.

        Returns:
            List of ``Document`` objects with content truncated to
            ``_MAX_DOC_CHARS`` and retrieval metadata attached.
        """
        try:
            results: list[Document] = get_vector_store().similarity_search(
                query,
                k=settings.top_k,
            )
        except Exception:
            logger.exception("Qdrant retrieval failed for query: %s", query)
            return []

        for doc in results:
            doc.page_content = doc.page_content[:_MAX_DOC_CHARS]
            doc.metadata["retrieval_method"] = "hybrid"

        logger.info("Retrieved %d documents for query: %s", len(results), query)
        return results
