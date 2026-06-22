"""Tavily-powered web search fallback for when retrieval fails."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from tavily import TavilyClient

from crag.config import settings

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Performs advanced web searches via the Tavily API."""

    def __init__(self) -> None:
        key = (settings.tavily_api_key or "").strip()
        self._client: TavilyClient | None
        self._client = TavilyClient(api_key=key) if key else None

    def search(self, query: str, max_results: int = 3) -> list[Document]:
        """Run a web search and return results as LangChain Documents.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            List of ``Document`` objects with source metadata.  Returns an
            empty list on API failure.
        """
        if self._client is None:
            logger.info("Web search skipped: TAVILY_API_KEY is not set")
            return []
        try:
            response = self._client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
        except Exception:
            logger.exception("Tavily web search failed for query: %s", query)
            return []

        docs: list[Document] = []
        for result in response.get("results", []):
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("url", ""),
                    "title": result.get("title", ""),
                    "origin": "web",
                    "tavily_score": result.get("score", 0.0),
                },
            )
            docs.append(doc)

        logger.info("Web search returned %d results for query: %s", len(docs), query)
        return docs
