"""Load file bytes, chunk text, and index into Qdrant."""

from __future__ import annotations

import logging
import uuid
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from crag.config import settings

logger = logging.getLogger(__name__)

# Matches langchain_qdrant QdrantVectorStore.add_texts default batch_size.
_INDEX_BATCH_SIZE = 64

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
_PDF_EXTENSION = ".pdf"


def extract_text(filename: str, data: bytes) -> str:
    """Decode plain text / markdown or extract text from PDF."""
    suffix = Path(filename).suffix.lower()
    if suffix in _TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace").strip()
    if suffix == _PDF_EXTENSION:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts).strip()
    msg = f"Unsupported file type '{suffix}'. Allowed: {sorted(_TEXT_EXTENSIONS | {_PDF_EXTENSION})}"
    raise ValueError(msg)


def split_into_documents(
    text: str,
    *,
    source: str,
    batch_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split ``text`` into chunked ``Document`` instances with metadata.

    Args:
        batch_id: Optional upload-batch identifier attached to every chunk so
            the whole upload can be deleted later.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = [c for c in splitter.split_text(text) if c.strip()]
    meta_base: dict[str, object] = {"source": source}
    if batch_id is not None:
        meta_base["batch_id"] = batch_id
    return [
        Document(
            page_content=chunk,
            metadata={**meta_base, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def index_documents(documents: list[Document]) -> list[str]:
    """Embed and upsert documents into the configured Qdrant collection.

    Upserts in batches. On failure, any points written in earlier batches are
    rolled back so Qdrant does not keep partial orphan chunks.

    Returns:
        Point ids assigned in Qdrant.
    """
    if not documents:
        return []
    from crag.qdrant_store import get_vector_store

    store = get_vector_store()
    point_ids = [uuid.uuid4().hex for _ in documents]
    added: list[str | int] = []
    try:
        for start in range(0, len(documents), _INDEX_BATCH_SIZE):
            end = start + _INDEX_BATCH_SIZE
            batch = documents[start:end]
            batch_ids = point_ids[start:end]
            part = store.add_documents(batch, ids=batch_ids)
            added.extend(part)
    except Exception:
        if added:
            _rollback_points(store, added)
        raise
    logger.info("Indexed %d chunks into collection '%s'", len(documents), settings.collection_name)
    return [str(x) for x in added]


def _rollback_points(store: object, point_ids: list[str | int]) -> None:
    """Best-effort removal of already-written points after a partial failure."""
    try:
        store.delete(ids=point_ids)  # type: ignore[union-attr]
        logger.warning("Rolled back %d orphan points after indexing failure", len(point_ids))
    except Exception:
        logger.exception(
            "Failed to roll back %d orphan points — manual cleanup required: %s",
            len(point_ids),
            point_ids,
        )


def count_by_source(source: str) -> int:
    """Return the number of Qdrant points with ``metadata.source == source``."""
    from crag.qdrant_store import get_vector_store

    store = get_vector_store()
    filt = _build_metadata_filter("source", source)
    result = store.client.count(
        collection_name=settings.collection_name,
        count_filter=filt,
        exact=True,
    )
    return result.count


def _build_metadata_filter(key: str, value: str) -> object:
    """Build a Qdrant REST ``Filter`` matching ``metadata.<key> == value``.

    Uses lazy import so ``qdrant_client`` is only pulled in when actually needed.
    """
    from qdrant_client.http.models.models import (  # type: ignore[import-untyped]
        FieldCondition,
        Filter,
        MatchValue,
    )

    return Filter(
        must=[
            FieldCondition(
                key=f"metadata.{key}",
                match=MatchValue(value=value),
            )
        ]
    )


def _delete_by_metadata(key: str, value: str, operation: str) -> int:
    from crag.qdrant_store import get_vector_store

    store = get_vector_store()
    client = store.client
    filt = _build_metadata_filter(key, value)

    count_result = client.count(
        collection_name=settings.collection_name,
        count_filter=filt,
        exact=True,
    )
    matched = count_result.count

    if matched == 0:
        logger.info("%s('%s'): no points matched", operation, value)
        return 0

    client.delete(
        collection_name=settings.collection_name,
        points_selector=filt,
    )
    logger.info("%s('%s'): deleted %d points", operation, value, matched)
    return matched


def delete_by_source(source: str) -> int:
    """Delete all Qdrant points whose ``metadata.source`` equals *source*.

    Returns:
        Number of points deleted (0 if none matched).
    """
    return _delete_by_metadata("source", source, "delete_by_source")


def delete_by_batch_id(batch_id: str) -> int:
    """Delete all Qdrant points whose ``metadata.batch_id`` equals *batch_id*.

    Returns:
        Number of points deleted (0 if none matched).
    """
    return _delete_by_metadata("batch_id", batch_id, "delete_by_batch_id")
