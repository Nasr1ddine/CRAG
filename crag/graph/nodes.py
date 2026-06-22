"""Individual node functions for the CRAG LangGraph pipeline.

Every function takes ``GraphState``, performs one logical step, and returns
a partial state dict that LangGraph merges back.

Components are lazily initialised on first use so that modules can be
imported without requiring a valid ``.env`` at parse time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from crag.config import settings
from crag.graph.state import GraphState

if TYPE_CHECKING:
    from crag.evaluator import FaithfulnessJudge
    from crag.generator import AnswerGenerator
    from crag.grader import RetrievalGrader
    from crag.refiner import ContextRefiner
    from crag.retriever import HybridRetriever
    from crag.rewriter import QueryRewriter
    from crag.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_retriever: HybridRetriever | None = None
_grader: RetrievalGrader | None = None
_rewriter: QueryRewriter | None = None
_refiner: ContextRefiner | None = None
_generator: AnswerGenerator | None = None
_judge: FaithfulnessJudge | None = None
_web_search: WebSearchTool | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        from crag.retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


def _get_grader() -> RetrievalGrader:
    global _grader  # noqa: PLW0603
    if _grader is None:
        from crag.grader import RetrievalGrader
        _grader = RetrievalGrader()
    return _grader


def _get_rewriter() -> QueryRewriter:
    global _rewriter  # noqa: PLW0603
    if _rewriter is None:
        from crag.rewriter import QueryRewriter
        _rewriter = QueryRewriter()
    return _rewriter


def _get_refiner() -> ContextRefiner:
    global _refiner  # noqa: PLW0603
    if _refiner is None:
        from crag.refiner import ContextRefiner
        _refiner = ContextRefiner()
    return _refiner


def _get_generator() -> AnswerGenerator:
    global _generator  # noqa: PLW0603
    if _generator is None:
        from crag.generator import AnswerGenerator
        _generator = AnswerGenerator()
    return _generator


def _get_judge() -> FaithfulnessJudge:
    global _judge  # noqa: PLW0603
    if _judge is None:
        from crag.evaluator import FaithfulnessJudge
        _judge = FaithfulnessJudge()
    return _judge


def _get_web_search() -> WebSearchTool:
    global _web_search  # noqa: PLW0603
    if _web_search is None:
        from crag.tools.web_search import WebSearchTool
        _web_search = WebSearchTool()
    return _web_search


def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents using hybrid search.

    Uses the rewritten query when available, otherwise the original query.
    """
    query = state.get("rewritten_query") or state["query"]
    logger.info("Node [retrieve]: query='%s'", query)
    documents = _get_retriever().retrieve(query)
    return {"documents": documents}  # type: ignore[return-value]


def grade_docs(state: GraphState) -> GraphState:
    """Grade every retrieved document for relevance."""
    query = state.get("rewritten_query") or state["query"]
    documents = state.get("documents", [])
    logger.info("Node [grade_docs]: grading %d documents", len(documents))
    graded = _get_grader().grade_all(query, documents)
    return {"graded_docs": graded}  # type: ignore[return-value]


def route_after_grading(state: GraphState) -> str:
    """Conditional edge: decide next step based on grading results.

    Returns:
        ``"generate"`` | ``"rewrite"`` | ``"web_search"``
    """
    iteration = state.get("iteration", 0)
    graded_docs = state.get("graded_docs", [])
    scores = [g["score"] for g in graded_docs]

    if iteration >= settings.max_rewrite_iterations:
        logger.info("Node [route]: iteration cap reached (%d) — generating best-effort answer", iteration)
        return "generate"

    if all(s == "IRRELEVANT" for s in scores):
        logger.info("Node [route]: all docs IRRELEVANT — falling back to web search")
        return "web_search"

    has_relevant = any(s == "RELEVANT" for s in scores)
    has_ambiguous = any(s == "AMBIGUOUS" for s in scores)

    if not has_relevant and has_ambiguous:
        logger.info("Node [route]: no RELEVANT docs, some AMBIGUOUS — rewriting query")
        return "rewrite"

    logger.info("Node [route]: at least one RELEVANT doc — proceeding to generate")
    return "generate"


def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite the query to improve retrieval on the next iteration."""
    query = state.get("rewritten_query") or state["query"]
    iteration = state.get("iteration", 0)

    irrelevant_topics = ", ".join(
        g["doc"].page_content[:80]
        for g in state.get("graded_docs", [])
        if g["score"] == "IRRELEVANT"
    )
    context_hint = (
        f"Previous retrieval returned irrelevant docs about: {irrelevant_topics}"
        if irrelevant_topics
        else ""
    )

    rewritten = _get_rewriter().rewrite(query, context=context_hint)
    new_iteration = iteration + 1
    logger.info("Node [rewrite_query]: iteration %d → '%s'", new_iteration, rewritten)
    return {  # type: ignore[return-value]
        "rewritten_query": rewritten,
        "iteration": new_iteration,
        "routing_decision": "rewrite",
    }


def web_search(state: GraphState) -> GraphState:
    """Fall back to web search and merge results into the document pool."""
    query = state.get("rewritten_query") or state["query"]
    logger.info("Node [web_search]: searching web for '%s'", query)

    web_docs: list[Document] = _get_web_search().search(query)

    existing_docs = list(state.get("documents", []))
    existing_graded = list(state.get("graded_docs", []))

    web_graded = [
        {"doc": doc, "score": "RELEVANT", "reason": "Sourced from web search."}
        for doc in web_docs
    ]

    return {  # type: ignore[return-value]
        "web_results": web_docs,
        "documents": existing_docs + web_docs,
        "graded_docs": existing_graded + web_graded,
        "routing_decision": "web_search",
    }


def refine_context(state: GraphState) -> GraphState:
    """Build refined context from graded documents."""
    query = state.get("rewritten_query") or state["query"]
    graded_docs = state.get("graded_docs", [])
    logger.info("Node [refine_context]: refining %d graded docs", len(graded_docs))

    context = _get_refiner().refine_all(query, graded_docs)
    return {"refined_context": context}  # type: ignore[return-value]


def generate(state: GraphState) -> GraphState:
    """Generate the final answer from refined context."""
    query = state.get("rewritten_query") or state["query"]
    context = state.get("refined_context", "")
    logger.info("Node [generate]: generating answer (%d chars of context)", len(context))

    if not context.strip():
        return {  # type: ignore[return-value]
            "answer": (
                "I could not find enough relevant information to answer your "
                "question. Please try rephrasing or providing more context."
            ),
            "routing_decision": state.get("routing_decision", "generate"),
        }

    answer = _get_generator().generate(query, context)
    return {  # type: ignore[return-value]
        "answer": answer,
        "routing_decision": state.get("routing_decision", "generate"),
    }


def evaluate(state: GraphState) -> GraphState:
    """Evaluate the faithfulness of the generated answer."""
    context = state.get("refined_context", "")
    answer = state.get("answer", "")
    logger.info("Node [evaluate]: checking faithfulness")

    result = _get_judge().evaluate(context, answer)
    return {  # type: ignore[return-value]
        "faithfulness_score": result.score,
        "faithfulness_issues": result.issues,
    }
