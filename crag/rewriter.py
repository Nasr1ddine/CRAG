"""Query rewriter — transforms vague queries into specific, keyword-rich ones."""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from crag.config import settings

logger = logging.getLogger(__name__)

_REWRITER_SYSTEM = (
    "You are a search-query optimiser. Your sole job is to rewrite the user's "
    "question so that a vector / keyword search engine returns better results.\n\n"
    "Guidelines:\n"
    "- Make the query more specific and keyword-rich.\n"
    "- Resolve pronouns and ambiguous references using the provided context.\n"
    "- Remove conversational filler ('um', 'I was wondering', etc.).\n"
    "- Keep the rewritten query concise (one or two sentences max).\n"
    "- Output ONLY the rewritten query — no explanation, no preamble."
)

_REWRITER_HUMAN = (
    "Original query: {query}\n\n"
    "Context about why retrieval failed: {context}\n\n"
    "Rewritten query:"
)


class QueryRewriter:
    """Rewrites underperforming queries to improve retrieval recall."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=settings.grader_model,
            temperature=0.3,
            openai_api_key=settings.openai_api_key,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _REWRITER_SYSTEM),
                ("human", _REWRITER_HUMAN),
            ]
        )
        self._chain = self._prompt | self._llm

    def rewrite(self, query: str, context: str = "") -> str:
        """Produce an improved search query.

        Args:
            query: The original user question.
            context: Optional hint about what went wrong (e.g. "previous
                retrieval returned irrelevant docs about X").

        Returns:
            A rewritten query string.
        """
        result = self._chain.invoke({"query": query, "context": context})
        rewritten = result.content.strip().strip('"').strip("'")
        logger.info("Rewrote query: '%s' → '%s'", query, rewritten)
        return rewritten
