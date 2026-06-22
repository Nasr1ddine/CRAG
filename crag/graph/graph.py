"""Builds and compiles the CRAG LangGraph state graph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from crag.graph.nodes import (
    evaluate,
    generate,
    grade_docs,
    refine_context,
    retrieve,
    rewrite_query,
    route_after_grading,
    web_search,
)
from crag.graph.state import GraphState


def build_graph() -> CompiledStateGraph:
    """Construct and compile the full CRAG pipeline.

    Returns:
        A compiled ``StateGraph`` ready to be invoked.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_docs", grade_docs)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("refine_context", refine_context)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_docs")

    workflow.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {
            "generate": "refine_context",
            "rewrite": "rewrite_query",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "refine_context")
    workflow.add_edge("refine_context", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()
