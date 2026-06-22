"""Unit tests for the retrieval grader."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from crag.grader import GradeDoc, RetrievalGrader


@pytest.fixture()
def grader() -> RetrievalGrader:
    """Create a grader instance with mocked LLM (no real langchain_openai import)."""
    stub = types.ModuleType("langchain_openai")
    stub.ChatOpenAI = MagicMock()
    with patch.dict(sys.modules, {"langchain_openai": stub}):
        return RetrievalGrader()


def test_grade_relevant_doc(grader: RetrievalGrader) -> None:
    """A clearly relevant document should be graded RELEVANT."""
    mock_result = GradeDoc(
        score="RELEVANT",
        reason="Document directly addresses the question about remote work policy.",
    )
    grader._llm = MagicMock(return_value=mock_result)

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch.object(grader, "_prompt") as mock_prompt:
        (mock_prompt.__or__).return_value = mock_chain

        doc = Document(
            page_content="Our remote work policy allows employees to work from home up to 3 days per week.",
            metadata={"source": "hr_handbook.pdf"},
        )
        result = grader.grade("What is the remote work policy?", doc)

    assert result.score == "RELEVANT"
    assert len(result.reason) > 0


def test_grade_irrelevant_doc(grader: RetrievalGrader) -> None:
    """An off-topic document should be graded IRRELEVANT."""
    mock_result = GradeDoc(
        score="IRRELEVANT",
        reason="Document discusses cafeteria menus, unrelated to deployment procedures.",
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch.object(grader, "_prompt") as mock_prompt:
        (mock_prompt.__or__).return_value = mock_chain

        doc = Document(
            page_content="The cafeteria serves lunch from 11:30 AM to 1:30 PM. Monday's special is pasta.",
            metadata={"source": "cafeteria_menu.pdf"},
        )
        result = grader.grade("How do we deploy to production?", doc)

    assert result.score == "IRRELEVANT"
    assert len(result.reason) > 0


def test_grade_all_handles_failure(grader: RetrievalGrader) -> None:
    """If grading a single doc raises, it should be marked IRRELEVANT and not crash."""
    with patch.object(grader, "grade", side_effect=RuntimeError("LLM timeout")):
        docs = [
            Document(page_content="Some content", metadata={"source": "test.pdf"}),
        ]
        results = grader.grade_all("test query", docs)

    assert len(results) == 1
    assert results[0]["score"] == "IRRELEVANT"
    assert "failed" in results[0]["reason"].lower()
