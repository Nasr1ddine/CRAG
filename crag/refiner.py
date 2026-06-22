"""Context refiner — strips irrelevant sentences from AMBIGUOUS documents."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from crag.config import settings

logger = logging.getLogger(__name__)

_MAX_REFINE_CHARS = 2_000

_REFINER_SYSTEM = (
    "You are a precise context filter. Given a user question and a document, "
    "extract ONLY the sentences that are directly relevant to answering the "
    "question. Remove everything else. If no sentences are relevant, return "
    "an empty string.\n\n"
    "Output the relevant sentences only — no commentary."
)

_REFINER_HUMAN = (
    "Question: {question}\n\n"
    "Document:\n{document}\n\n"
    "Relevant sentences:"
)


class ContextRefiner:
    """Refines AMBIGUOUS docs by keeping only query-relevant sentences."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.grader_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _REFINER_SYSTEM),
                ("human", _REFINER_HUMAN),
            ]
        )
        self._chain = self._prompt | self._llm

    def refine_doc(self, query: str, doc: Document, score: str) -> str:
        """Refine a single document based on its grading score.

        Args:
            query: The user question.
            doc: The retrieved document.
            score: One of ``RELEVANT``, ``AMBIGUOUS``, ``IRRELEVANT``.

        Returns:
            Full content for RELEVANT docs, filtered content for AMBIGUOUS,
            empty string for IRRELEVANT.
        """
        if score == "IRRELEVANT":
            return ""
        if score == "RELEVANT":
            return doc.page_content

        truncated = doc.page_content[:_MAX_REFINE_CHARS]
        try:
            result = self._chain.invoke(
                {"question": query, "document": truncated}
            )
            return result.content.strip()
        except Exception:
            logger.exception("Refinement failed, returning full content as fallback")
            return doc.page_content

    def refine_all(self, query: str, graded_docs: list[dict]) -> str:
        """Build a single context string from graded documents.

        RELEVANT docs are included in full; AMBIGUOUS docs go through
        sentence-level filtering; IRRELEVANT docs are dropped entirely.
        Each section is annotated with its source metadata.

        Args:
            query: The user question.
            graded_docs: List of dicts with keys ``doc``, ``score``, ``reason``.

        Returns:
            Combined context string ready for the generator.
        """
        sections: list[str] = []
        for entry in graded_docs:
            doc: Document = entry["doc"]
            score: str = entry["score"]
            refined = self.refine_doc(query, doc, score)
            if not refined:
                continue
            source = doc.metadata.get("source", "unknown")
            origin = doc.metadata.get("origin", "retrieval")
            header = f"[Source: {source} | Origin: {origin}]"
            sections.append(f"{header}\n{refined}")

        context = "\n\n---\n\n".join(sections)
        logger.info(
            "Refined context: %d sections, %d chars", len(sections), len(context)
        )
        return context
