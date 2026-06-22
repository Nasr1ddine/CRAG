"""Tests for Qdrant URL normalization."""

from __future__ import annotations

from crag.config import normalize_qdrant_url


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
