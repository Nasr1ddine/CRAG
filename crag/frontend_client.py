"""HTTP client used by the Streamlit frontend.

This module intentionally has no Streamlit imports so request/response handling
can be unit-tested without a running Streamlit session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


@dataclass(frozen=True)
class QueryResult:
    """Normalized response from the CRAG query endpoint."""

    answer: str
    faithfulness_score: float
    faithfulness_issues: list[str]
    routing_decision: str
    sources: list[str]
    iteration_count: int
    query_used: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "QueryResult":
        return cls(
            answer=str(payload.get("answer", "")),
            faithfulness_score=float(payload.get("faithfulness_score", 0.0)),
            faithfulness_issues=[str(item) for item in payload.get("faithfulness_issues", [])],
            routing_decision=str(payload.get("routing_decision", "")),
            sources=[str(item) for item in payload.get("sources", [])],
            iteration_count=int(payload.get("iteration_count", 0)),
            query_used=str(payload.get("query_used", "")),
        )


@dataclass(frozen=True)
class IngestResult:
    """Normalized response from document ingest endpoints."""

    batch_id: str
    sources: list[str]
    chunks_indexed: int
    point_ids: list[str]
    replaced_sources: list[str]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IngestResult":
        return cls(
            batch_id=str(payload.get("batch_id", "")),
            sources=[str(item) for item in payload.get("sources", [])],
            chunks_indexed=int(payload.get("chunks_indexed", 0)),
            point_ids=[str(item) for item in payload.get("point_ids", [])],
            replaced_sources=[str(item) for item in payload.get("replaced_sources", [])],
        )


@dataclass(frozen=True)
class DeleteResult:
    """Normalized response from document deletion endpoints."""

    deleted_count: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DeleteResult":
        return cls(deleted_count=int(payload.get("deleted_count", 0)))


@dataclass(frozen=True)
class DebugConfig:
    """Safe backend configuration values exposed by the API."""

    qdrant_url: str
    collection_name: str
    has_qdrant_api_key: bool
    has_openai_api_key: bool
    has_tavily_api_key: bool
    has_langfuse_public_key: bool
    has_langfuse_secret_key: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DebugConfig":
        return cls(
            qdrant_url=str(payload.get("qdrant_url", "")),
            collection_name=str(payload.get("collection_name", "")),
            has_qdrant_api_key=bool(payload.get("has_qdrant_api_key", False)),
            has_openai_api_key=bool(payload.get("has_openai_api_key", False)),
            has_tavily_api_key=bool(payload.get("has_tavily_api_key", False)),
            has_langfuse_public_key=bool(payload.get("has_langfuse_public_key", False)),
            has_langfuse_secret_key=bool(payload.get("has_langfuse_secret_key", False)),
        )


@dataclass(frozen=True)
class UploadFilePayload:
    """File payload accepted by the multipart upload endpoint."""

    filename: str
    content: bytes
    content_type: str


class CRAGApiError(RuntimeError):
    """Raised when the CRAG API returns an error or cannot be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CRAGApiClient:
    """Small typed client for the CRAG FastAPI service."""

    def __init__(self, base_url: str, *, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> dict[str, str]:
        payload = self._request("GET", "/health")
        return {str(key): str(value) for key, value in payload.items()}

    def debug_config(self) -> DebugConfig:
        return DebugConfig.from_payload(self._request("GET", "/debug/config"))

    def query(self, query: str, *, user_id: str | None = None) -> QueryResult:
        payload = {"query": query, "user_id": user_id or None}
        return QueryResult.from_payload(self._request("POST", "/query", json=payload))

    def upload_documents(self, uploads: list[UploadFilePayload]) -> IngestResult:
        files = [
            ("files", (upload.filename, upload.content, upload.content_type))
            for upload in uploads
        ]
        return IngestResult.from_payload(self._request("POST", "/documents/upload", files=files))

    def ingest_text(self, text: str, *, source: str) -> IngestResult:
        payload = {"text": text, "source": source}
        return IngestResult.from_payload(self._request("POST", "/documents/text", json=payload))

    def delete_source(self, source: str) -> DeleteResult:
        encoded_source = quote(source, safe="")
        return DeleteResult.from_payload(self._request("DELETE", f"/documents/source/{encoded_source}"))

    def delete_batch(self, batch_id: str) -> DeleteResult:
        encoded_batch_id = quote(batch_id, safe="")
        return DeleteResult.from_payload(self._request("DELETE", f"/documents/batch/{encoded_batch_id}"))

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _extract_error_detail(exc.response)
            raise CRAGApiError(detail, status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise CRAGApiError(f"Could not reach CRAG API at {self.base_url}: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise CRAGApiError("CRAG API returned a non-JSON response.") from exc

        if not isinstance(payload, dict):
            raise CRAGApiError("CRAG API returned an unexpected response shape.")
        return payload


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"CRAG API returned HTTP {response.status_code}."

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    return f"CRAG API returned HTTP {response.status_code}."
