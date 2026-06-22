"""Shared test fixtures — provide dummy settings so tests run without .env."""

from __future__ import annotations

import pytest

from crag.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Clear the ``get_settings`` LRU cache between tests."""
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal dummy settings so pydantic validation never fails in tests."""
    envs = {
        "OPENAI_API_KEY": "sk-test",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "test-key",
        "TAVILY_API_KEY": "tvly-test",
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
        "LANGFUSE_HOST": "https://localhost",
        "COLLECTION_NAME": "test_crag",
    }
    for k, v in envs.items():
        monkeypatch.setenv(k, v)
