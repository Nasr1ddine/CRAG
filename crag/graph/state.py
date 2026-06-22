"""Graph state definition shared across all CRAG pipeline nodes."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from langchain_core.documents import Document


class GradedDocument(TypedDict):
    """A retrieved document paired with its relevance assessment."""

    doc: Document
    score: Literal["RELEVANT", "AMBIGUOUS", "IRRELEVANT"]
    reason: str


class GraphState(TypedDict, total=False):
    """Mutable state that flows through every node in the CRAG graph.

    Fields use ``total=False`` so nodes only need to set the keys they
    modify; LangGraph merges updates automatically.
    """

    query: str
    rewritten_query: Optional[str]
    documents: list[Document]
    graded_docs: list[GradedDocument]
    web_results: list[Document]
    refined_context: str
    answer: str
    faithfulness_score: float
    faithfulness_issues: list[str]
    routing_decision: Literal["generate", "rewrite", "web_search"]
    iteration: int
    trace_id: Optional[str]


def make_initial_state(query: str, trace_id: Optional[str] = None) -> GraphState:
    """Create the complete starting state shared by API, eval, and tests."""
    return {
        "query": query,
        "rewritten_query": None,
        "documents": [],
        "graded_docs": [],
        "web_results": [],
        "refined_context": "",
        "answer": "",
        "faithfulness_score": 0.0,
        "faithfulness_issues": [],
        "routing_decision": "generate",
        "iteration": 0,
        "trace_id": trace_id,
    }
