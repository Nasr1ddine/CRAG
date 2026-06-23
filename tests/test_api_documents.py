"""API regression tests for document ingestion error handling."""

from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main


def test_upload_reports_text_extraction_failure(monkeypatch) -> None:
    def broken_extract(filename: str, data: bytes) -> str:
        raise RuntimeError("bad pdf structure")

    monkeypatch.setattr(api_main, "extract_text", broken_extract)

    client = TestClient(api_main.app)
    response = client.post(
        "/documents/upload",
        files=[("files", ("broken.pdf", b"not a real pdf", "application/pdf"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not extract text from 'broken.pdf': bad pdf structure"


def test_text_ingest_reports_pre_index_cleanup_failure(monkeypatch) -> None:
    def broken_count(source: str) -> int:
        raise RuntimeError("qdrant offline")

    monkeypatch.setattr(api_main, "count_by_source", broken_count)

    client = TestClient(api_main.app)
    response = client.post(
        "/documents/text",
        json={"source": "notes.txt", "text": "hello world"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Pre-index cleanup error for source 'notes.txt': qdrant offline"
