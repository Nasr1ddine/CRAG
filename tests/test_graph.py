"""Integration tests for the CRAG graph pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from crag.graph.state import make_initial_state


@pytest.fixture()
def irrelevant_docs() -> list[Document]:
    """A set of clearly irrelevant documents."""
    return [
        Document(
            page_content="Best pasta recipes from Italian chefs.",
            metadata={"source": "cooking_blog.com"},
        ),
        Document(
            page_content="Top 10 vacation destinations in Europe.",
            metadata={"source": "travel_guide.com"},
        ),
    ]


@pytest.fixture()
def relevant_doc() -> Document:
    """A single clearly relevant document."""
    return Document(
        page_content="The company uses Kubernetes for container orchestration across all environments.",
        metadata={"source": "arch_docs.pdf"},
    )


def test_all_irrelevant_triggers_web_search(irrelevant_docs: list[Document]) -> None:
    """When all retrieved docs are IRRELEVANT, routing should select web_search."""
    from crag.graph.nodes import route_after_grading

    state = {
        "query": "What is the deployment process?",
        "iteration": 0,
        "graded_docs": [
            {"doc": doc, "score": "IRRELEVANT", "reason": "Off-topic"}
            for doc in irrelevant_docs
        ],
    }
    decision = route_after_grading(state)  # type: ignore[arg-type]
    assert decision == "web_search"


def test_ambiguous_only_triggers_rewrite(irrelevant_docs: list[Document]) -> None:
    """When no docs are RELEVANT but some are AMBIGUOUS, routing should rewrite."""
    from crag.graph.nodes import route_after_grading

    state = {
        "query": "Tell me about deployments",
        "iteration": 0,
        "graded_docs": [
            {"doc": irrelevant_docs[0], "score": "AMBIGUOUS", "reason": "Partially related"},
            {"doc": irrelevant_docs[1], "score": "IRRELEVANT", "reason": "Off-topic"},
        ],
    }
    decision = route_after_grading(state)  # type: ignore[arg-type]
    assert decision == "rewrite"


def test_relevant_doc_triggers_generate(relevant_doc: Document) -> None:
    """When at least one doc is RELEVANT, routing should generate."""
    from crag.graph.nodes import route_after_grading

    state = {
        "query": "What is our container orchestration platform?",
        "iteration": 0,
        "graded_docs": [
            {"doc": relevant_doc, "score": "RELEVANT", "reason": "Directly answers the question"},
        ],
    }
    decision = route_after_grading(state)  # type: ignore[arg-type]
    assert decision == "generate"


def test_iteration_cap_forces_generate(irrelevant_docs: list[Document]) -> None:
    """When max iterations reached, should generate even with IRRELEVANT docs."""
    from crag.graph.nodes import route_after_grading

    state = {
        "query": "obscure question",
        "iteration": 2,
        "graded_docs": [
            {"doc": doc, "score": "IRRELEVANT", "reason": "Off-topic"}
            for doc in irrelevant_docs
        ],
    }
    decision = route_after_grading(state)  # type: ignore[arg-type]
    assert decision == "generate"


def test_full_graph_with_mocked_components() -> None:
    """End-to-end graph run with all external services mocked."""
    import crag.graph.nodes as nodes_mod

    mock_docs = [
        Document(
            page_content="Employees can work remotely up to 3 days per week.",
            metadata={"source": "hr_handbook.pdf"},
        ),
    ]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_docs

    mock_grader = MagicMock()
    mock_grader.grade_all.return_value = [
        {"doc": mock_docs[0], "score": "RELEVANT", "reason": "Directly relevant"}
    ]

    mock_refiner = MagicMock()
    mock_refiner.refine_all.return_value = "Employees can work remotely up to 3 days per week."

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Employees may work from home up to 3 days weekly."

    mock_faith = MagicMock()
    mock_faith.score = 1.0
    mock_faith.issues = []

    mock_judge = MagicMock()
    mock_judge.evaluate.return_value = mock_faith

    with (
        patch.object(nodes_mod, "_get_retriever", return_value=mock_retriever),
        patch.object(nodes_mod, "_get_grader", return_value=mock_grader),
        patch.object(nodes_mod, "_get_refiner", return_value=mock_refiner),
        patch.object(nodes_mod, "_get_generator", return_value=mock_generator),
        patch.object(nodes_mod, "_get_judge", return_value=mock_judge),
    ):
        from crag.graph.graph import build_graph

        graph = build_graph()
        result = graph.invoke(make_initial_state("What is the remote work policy?"))

    assert result["answer"] == "Employees may work from home up to 3 days weekly."
    assert result["faithfulness_score"] == 1.0
    assert result["routing_decision"] == "generate"
