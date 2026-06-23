"""FastAPI application exposing the CRAG pipeline as a REST API."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from crag.config import settings
from crag.ingest import (
    count_by_source,
    delete_by_batch_id,
    delete_by_source,
    extract_text,
    index_documents,
    split_into_documents,
)
from crag.graph.state import make_initial_state

logger = logging.getLogger(__name__)

_graph: CompiledStateGraph | None = None

_SWAGGER_STATIC_DIR = Path(__file__).resolve().parent / "static" / "swagger-ui"
_SWAGGER_JS = _SWAGGER_STATIC_DIR / "swagger-ui-bundle.js"
_SWAGGER_CSS = _SWAGGER_STATIC_DIR / "swagger-ui.css"
_SWAGGER_CDN = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.11.0"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build and cache the CRAG graph on startup."""
    global _graph  # noqa: PLW0603
    from crag.graph.graph import build_graph

    _graph = build_graph()
    logger.info("CRAG graph compiled and ready")
    yield
    _graph = None


app = FastAPI(
    title="CRAG API",
    description="Self-Correcting Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
)

if _SWAGGER_JS.is_file():
    app.mount(
        "/static/swagger-ui",
        StaticFiles(directory=str(_SWAGGER_STATIC_DIR)),
        name="swagger-ui-static",
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui_docs(request: Request) -> HTMLResponse:
    """Swagger UI with locally hosted assets when available (avoids blank /docs when CDNs are blocked)."""
    root_path = request.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    if _SWAGGER_JS.is_file() and _SWAGGER_CSS.is_file():
        js_url = f"{root_path}/static/swagger-ui/swagger-ui-bundle.js"
        css_url = f"{root_path}/static/swagger-ui/swagger-ui.css"
        favicon_url = f"{_SWAGGER_CDN}/favicon-32x32.png"
    else:
        logger.warning(
            "Swagger UI static files missing under %s; falling back to CDN (may fail offline).",
            _SWAGGER_STATIC_DIR,
        )
        js_url = f"{_SWAGGER_CDN}/swagger-ui-bundle.js"
        css_url = f"{_SWAGGER_CDN}/swagger-ui.css"
        favicon_url = f"{_SWAGGER_CDN}/favicon-32x32.png"

    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url=js_url,
        swagger_css_url=css_url,
        swagger_favicon_url=favicon_url,
        swagger_ui_parameters=app.swagger_ui_parameters,
    )


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
async def document_upload_page() -> HTMLResponse:
    """Simple browser UI to pick files and POST multipart data to ``/documents/upload``."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRAG — Upload documents</title>
  <style>
    :root { font-family: system-ui, sans-serif; background: #0f1419; color: #e6edf3; }
    body { max-width: 40rem; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; font-weight: 600; }
    p, label { color: #8b949e; font-size: 0.9rem; line-height: 1.5; }
    input[type=file] { margin: 1rem 0; width: 100%; }
    button {
      background: #238636; color: #fff; border: 0; padding: 0.6rem 1.2rem;
      border-radius: 6px; font-size: 1rem; cursor: pointer;
    }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #out { margin-top: 1.5rem; white-space: pre-wrap; word-break: break-word;
      background: #161b22; padding: 1rem; border-radius: 6px; font-size: 0.8rem;
      border: 1px solid #30363d; min-height: 3rem; }
    .err { color: #f85149; }
    .ok { color: #3fb950; }
    a { color: #58a6ff; }
  </style>
</head>
<body>
  <h1>Upload documents to Qdrant</h1>
  <p>Choose one or more <strong>.txt</strong>, <strong>.md</strong>, or <strong>.pdf</strong> files.
     This sends a normal browser upload (multipart) to <code>/documents/upload</code> — not a file path string.</p>
  <label for="f">Files</label>
  <input id="f" type="file" multiple accept=".txt,.md,.markdown,.pdf,text/plain,application/pdf">
  <div><button type="button" id="go">Upload</button></div>
  <div id="out"></div>
  <p><a href="/docs">Open API docs</a> · <a href="/">Home</a></p>
  <script>
    const f = document.getElementById('f');
    const go = document.getElementById('go');
    const out = document.getElementById('out');
    go.onclick = async () => {
      out.textContent = '';
      out.className = '';
      if (!f.files || !f.files.length) {
        out.className = 'err';
        out.textContent = 'Select at least one file.';
        return;
      }
      const fd = new FormData();
      for (const file of f.files) fd.append('files', file, file.name);
      go.disabled = true;
      try {
        const r = await fetch('/documents/upload', { method: 'POST', body: fd });
        const text = await r.text();
        let pretty = text;
        try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch (e) {}
        out.className = r.ok ? 'ok' : 'err';
        out.textContent = (r.ok ? '' : r.status + ' ') + pretty;
      } catch (e) {
        out.className = 'err';
        out.textContent = String(e);
      }
      go.disabled = false;
    };
  </script>
</body>
</html>"""
    return HTMLResponse(html)


class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question.")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier for tracing.")


class QueryResponse(BaseModel):
    """Response payload returned after pipeline execution."""

    answer: str
    faithfulness_score: float
    faithfulness_issues: list[str]
    routing_decision: str
    sources: list[str]
    iteration_count: int
    query_used: str


class TextIngestRequest(BaseModel):
    """Plain-text body to chunk and index (no file upload)."""

    text: str = Field(..., min_length=1, max_length=2_000_000, description="Raw text to index.")
    source: str = Field(
        default="inline_text",
        min_length=1,
        max_length=500,
        description="Label stored as document metadata (e.g. notebook name).",
    )


class IngestResponse(BaseModel):
    """Result of indexing uploaded or inline text."""

    batch_id: str = Field(description="Unique ID for this upload batch (use to delete all its chunks).")
    sources: list[str] = Field(description="Logical names of indexed inputs (e.g. filenames).")
    chunks_indexed: int
    point_ids: list[str]
    replaced_sources: list[str] = Field(
        default_factory=list,
        description="Sources whose old chunks were deleted before re-indexing (dedup).",
    )


def _extract_upload_text(label: str, raw: bytes) -> str:
    """Extract uploaded text and convert parser failures into client errors."""
    try:
        return extract_text(label, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Text extraction failed for upload '%s'", label)
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract text from '{label}': {exc}",
        ) from exc


def _replace_existing_source_if_present(source: str) -> bool:
    """Remove previous chunks for a source before a fresh ingest."""
    try:
        if count_by_source(source) > 0:
            delete_by_source(source)
            return True
    except Exception as exc:
        logger.exception("Pre-index cleanup failed for source '%s'", source)
        raise HTTPException(
            status_code=500,
            detail=f"Pre-index cleanup error for source '{source}': {exc}",
        ) from exc
    return False


def _build_langfuse_handler(
    trace_id: str,
    user_id: Optional[str] = None,
) -> LangfuseCallbackHandler:
    """Create a per-request Langfuse callback handler (Langfuse v2+ API)."""
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    trace = client.trace(id=trace_id, user_id=user_id)
    return trace.get_langchain_handler()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Run a query through the full CRAG pipeline.

    Args:
        request: The incoming query with optional user_id.

    Returns:
        ``QueryResponse`` containing the answer, scores, and metadata.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised yet.")

    trace_id = str(uuid.uuid4())
    callbacks: list[LangfuseCallbackHandler] = []
    pk = (settings.langfuse_public_key or "").strip()
    sk = (settings.langfuse_secret_key or "").strip()
    if pk and sk:
        callbacks.append(_build_langfuse_handler(trace_id, request.user_id))

    initial_state = make_initial_state(request.query, trace_id=trace_id)

    try:
        result = _graph.invoke(
            initial_state,
            config={"callbacks": callbacks},
        )
    except Exception as exc:
        logger.exception("Graph execution failed for query: %s", request.query)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {exc}",
        ) from exc

    sources: list[str] = []
    for g in result.get("graded_docs", []):
        src = g["doc"].metadata.get("source", "")
        if src and src not in sources:
            sources.append(src)

    query_used = result.get("rewritten_query") or result.get("query", request.query)

    return QueryResponse(
        answer=result.get("answer", ""),
        faithfulness_score=result.get("faithfulness_score", 0.0),
        faithfulness_issues=result.get("faithfulness_issues", []),
        routing_decision=result.get("routing_decision", "generate"),
        sources=sources,
        iteration_count=result.get("iteration", 0),
        query_used=query_used,
    )


@app.post("/documents/upload", response_model=IngestResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> IngestResponse:
    """Chunk and index uploaded files into the Qdrant collection used by ``/query``.

    Supported types: ``.txt``, ``.md``, ``.markdown``, ``.pdf``.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    batch_id = uuid.uuid4().hex
    all_docs: list[Document] = []
    sources: list[str] = []
    replaced: list[str] = []
    max_b = settings.max_upload_bytes

    for upload in files:
        if upload.size is not None and upload.size > max_b:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename or 'unnamed'}' exceeds max size of {max_b} bytes.",
            )
        raw = await upload.read()
        if len(raw) > max_b:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename or 'unnamed'}' exceeds max size of {max_b} bytes.",
            )
        label = upload.filename or "upload.txt"
        text = _extract_upload_text(label, raw)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No extractable text in '{label}'.",
            )

        if _replace_existing_source_if_present(label):
            replaced.append(label)

        all_docs.extend(split_into_documents(text, source=label, batch_id=batch_id))
        sources.append(label)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No text chunks produced from uploads.")

    try:
        ids = index_documents(all_docs)
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(
            status_code=500,
            detail=f"Indexing error: {exc}",
        ) from exc

    return IngestResponse(
        batch_id=batch_id,
        sources=sources,
        chunks_indexed=len(all_docs),
        point_ids=ids,
        replaced_sources=replaced,
    )


@app.post("/documents/text", response_model=IngestResponse)
async def ingest_text_body(request: TextIngestRequest) -> IngestResponse:
    """Chunk and index a JSON body of raw text (alternative to multipart upload)."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty after trimming.")

    if len(text.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds max size of {settings.max_upload_bytes} bytes.",
        )

    batch_id = uuid.uuid4().hex
    replaced: list[str] = []

    if _replace_existing_source_if_present(request.source):
        replaced.append(request.source)

    docs = split_into_documents(text, source=request.source, batch_id=batch_id)
    if not docs:
        raise HTTPException(status_code=400, detail="No text chunks produced.")

    try:
        ids = index_documents(docs)
    except Exception as exc:
        logger.exception("Inline text indexing failed")
        raise HTTPException(
            status_code=500,
            detail=f"Indexing error: {exc}",
        ) from exc

    return IngestResponse(
        batch_id=batch_id,
        sources=[request.source],
        chunks_indexed=len(docs),
        point_ids=ids,
        replaced_sources=replaced,
    )


class DeleteResponse(BaseModel):
    """Result of a deletion request."""

    deleted_count: int = Field(description="Number of Qdrant points removed.")


class DebugConfigResponse(BaseModel):
    """Safe runtime config snapshot for troubleshooting deployment/env issues."""

    qdrant_url: str
    collection_name: str
    has_qdrant_api_key: bool
    has_openai_api_key: bool
    has_tavily_api_key: bool
    has_langfuse_public_key: bool
    has_langfuse_secret_key: bool


def _delete_documents(
    delete_fn: Callable[[str], int],
    value: str,
    operation: str,
    field_name: str,
) -> DeleteResponse:
    try:
        count = delete_fn(value)
    except Exception as exc:
        logger.exception("%s failed for '%s'", operation, value)
        raise HTTPException(status_code=500, detail=f"Deletion error: {exc}") from exc
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No documents found with {field_name} '{value}'.")
    return DeleteResponse(deleted_count=count)


@app.delete("/documents/source/{source}", response_model=DeleteResponse)
async def delete_documents_by_source(source: str) -> DeleteResponse:
    """Delete all chunks whose ``metadata.source`` matches *source* (e.g. a filename)."""
    return _delete_documents(delete_by_source, source, "delete_by_source", "source")


@app.delete("/documents/batch/{batch_id}", response_model=DeleteResponse)
async def delete_documents_by_batch(batch_id: str) -> DeleteResponse:
    """Delete all chunks belonging to a specific upload batch (returned by ``/documents/upload`` or ``/documents/text``)."""
    return _delete_documents(delete_by_batch_id, batch_id, "delete_by_batch_id", "batch_id")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/debug/config", response_model=DebugConfigResponse, include_in_schema=False)
async def debug_config() -> DebugConfigResponse:
    """Expose non-secret runtime config values for fast diagnosis of 500s."""
    return DebugConfigResponse(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.collection_name,
        has_qdrant_api_key=bool((settings.qdrant_api_key or "").strip()),
        has_openai_api_key=bool((settings.openai_api_key or "").strip()),
        has_tavily_api_key=bool((settings.tavily_api_key or "").strip()),
        has_langfuse_public_key=bool((settings.langfuse_public_key or "").strip()),
        has_langfuse_secret_key=bool((settings.langfuse_secret_key or "").strip()),
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Service metadata."""
    return {
        "name": "CRAG API",
        "version": "1.0.0",
        "upload_ui": "/upload",
        "docs": "/docs",
    }
