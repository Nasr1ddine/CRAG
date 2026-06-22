"""Retrieval grader — classifies each document as RELEVANT / AMBIGUOUS / IRRELEVANT."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from crag.config import settings

logger = logging.getLogger(__name__)

_MAX_GRADING_CHARS = 2_000

_GRADER_SYSTEM = (
    "You are a strict retrieval-relevance grader. Given a user question and a "
    "retrieved document, decide whether the document is relevant to answering "
    "the question.\n\n"
    "Rules:\n"
    "- RELEVANT  — the document directly answers or substantially supports an answer.\n"
    "- AMBIGUOUS — the document is partially related but not clearly sufficient on its own.\n"
    "- IRRELEVANT — the document has no meaningful connection to the question.\n\n"
    "Be STRICT: if the document only tangentially mentions a keyword without "
    "providing useful information, mark it AMBIGUOUS or IRRELEVANT."
)

_GRADER_HUMAN = (
    "Question: {question}\n\n"
    "Document content:\n{document}\n\n"
    "Provide your assessment."
)


class GradeDoc(BaseModel):
    """Structured grading output for a single document."""

    score: Literal["RELEVANT", "AMBIGUOUS", "IRRELEVANT"] = Field(
        description="Relevance classification."
    )
    reason: str = Field(description="One-sentence justification for the score.")


class RetrievalGrader:
    """Grades retrieved documents against a query using structured LLM output."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=settings.grader_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        ).with_structured_output(GradeDoc)

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _GRADER_SYSTEM),
                ("human", _GRADER_HUMAN),
            ]
        )

    def grade(self, query: str, doc: Document) -> GradeDoc:
        """Grade a single document's relevance to the query.

        Args:
            query: The user question.
            doc: A retrieved document.

        Returns:
            ``GradeDoc`` with score and reason.
        """
        truncated = doc.page_content[:_MAX_GRADING_CHARS]
        result: GradeDoc = (self._prompt | self._llm).invoke(
            {"question": query, "document": truncated}
        )
        return result  # type: ignore[return-value]

    def grade_all(
        self, query: str, docs: list[Document]
    ) -> list[dict]:
        """Grade every document in the list.

        Args:
            query: The user question.
            docs: Retrieved documents.

        Returns:
            List of dicts with keys ``doc``, ``score``, ``reason``.
        """
        graded: list[dict] = []
        for doc in docs:
            try:
                grade = self.grade(query, doc)
                graded.append(
                    {"doc": doc, "score": grade.score, "reason": grade.reason}
                )
            except Exception:
                logger.exception("Grading failed for doc: %s", doc.metadata)
                graded.append(
                    {
                        "doc": doc,
                        "score": "IRRELEVANT",
                        "reason": "Grading failed — treated as irrelevant.",
                    }
                )
        relevant = sum(1 for g in graded if g["score"] == "RELEVANT")
        ambiguous = sum(1 for g in graded if g["score"] == "AMBIGUOUS")
        logger.info(
            "Graded %d docs: %d RELEVANT, %d AMBIGUOUS, %d IRRELEVANT",
            len(graded),
            relevant,
            ambiguous,
            len(graded) - relevant - ambiguous,
        )
        return graded
