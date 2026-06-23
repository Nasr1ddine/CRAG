"""Tests for Qdrant URL normalization."""

from __future__ import annotations

from crag.config import normalize_qdrant_url
from crag.frontend_client import CRAGApiClient
from crag.frontend_state import FALLBACK_API_BASE_URL, normalize_api_base_url


def test_qdrant_cloud_adds_default_port() -> None:
    assert normalize_qdrant_url(
        "https://acf20d98-01d1-4d90-9f73-c62957de28ca.europe-west3-0.gcp.cloud.qdrant.io"
    ) == (
        "https://acf20d98-01d1-4d90-9f73-c62957de28ca.europe-west3-0.gcp.cloud.qdrant.io:6333"
    )


def test_qdrant_cloud_preserves_explicit_port() -> None:
    u = "https://mycluster.eu-west.aws.cloud.qdrant.io:6333"
    assert normalize_qdrant_url(u) == u


def test_localhost_adds_port() -> None:
    assert normalize_qdrant_url("http://127.0.0.1") == "http://127.0.0.1:6333"
    assert normalize_qdrant_url("http://localhost") == "http://localhost:6333"


def test_localhost_with_port_unchanged() -> None:
    u = "http://127.0.0.1:6333"
    assert normalize_qdrant_url(u) == u


def test_frontend_api_url_uses_fallback_for_blank_values() -> None:
    assert normalize_api_base_url("") == FALLBACK_API_BASE_URL
    assert normalize_api_base_url("   ") == FALLBACK_API_BASE_URL
    assert normalize_api_base_url(None) == FALLBACK_API_BASE_URL


def test_frontend_api_url_adds_default_protocol() -> None:
    assert normalize_api_base_url("localhost:8001") == "http://localhost:8001"


def test_frontend_client_normalizes_base_url() -> None:
    client = CRAGApiClient("localhost:8001/")
    try:
        assert client.base_url == "http://localhost:8001"
    finally:
        client.close()
