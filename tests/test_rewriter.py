"""Unit tests for the query rewriter."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from crag.rewriter import QueryRewriter


@pytest.fixture()
def rewriter() -> QueryRewriter:
    """Create a rewriter instance with mocked LLM (no real langchain_openai import)."""
    stub = types.ModuleType("langchain_openai")
    stub.ChatOpenAI = MagicMock()
    with patch.dict(sys.modules, {"langchain_openai": stub}):
        return QueryRewriter()


def test_rewrite_produces_improved_query(rewriter: QueryRewriter) -> None:
    """A vague query should be rewritten into something more specific."""
    improved = "What is the company's official remote work policy including allowed WFH days per week?"

    mock_result = MagicMock()
    mock_result.content = improved

    rewriter._chain = MagicMock()
    rewriter._chain.invoke.return_value = mock_result

    result = rewriter.rewrite("Tell me about working from home")

    assert len(result) > len("Tell me about working from home")
    assert "remote work" in result.lower() or "wfh" in result.lower()
    rewriter._chain.invoke.assert_called_once()


def test_rewrite_strips_quotes(rewriter: QueryRewriter) -> None:
    """The rewriter should strip leading/trailing quotes from the output."""
    mock_result = MagicMock()
    mock_result.content = '"What are the deployment procedures for production releases?"'

    rewriter._chain = MagicMock()
    rewriter._chain.invoke.return_value = mock_result

    result = rewriter.rewrite("how do we deploy stuff")

    assert not result.startswith('"')
    assert not result.endswith('"')


def test_rewrite_uses_context_hint(rewriter: QueryRewriter) -> None:
    """When context is provided, it should be passed to the chain."""
    mock_result = MagicMock()
    mock_result.content = "Kubernetes production deployment CI/CD pipeline process"

    rewriter._chain = MagicMock()
    rewriter._chain.invoke.return_value = mock_result

    context_hint = "Previous retrieval returned irrelevant docs about cafeteria menus"
    result = rewriter.rewrite("how do we deploy", context=context_hint)

    call_args = rewriter._chain.invoke.call_args[0][0]
    assert call_args["context"] == context_hint
    assert len(result) > 0
