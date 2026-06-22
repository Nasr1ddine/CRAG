"""Unit tests for document text extraction, chunking, indexing, and deletion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from crag.ingest import extract_text, split_into_documents

import crag.qdrant_store as _qs_mod
import crag.ingest as _ingest_mod


# --- extract_text ---

def test_extract_plain_text() -> None:
    assert extract_text("notes.txt", b"  hello world  ") == "hello world"


def test_extract_markdown() -> None:
    assert extract_text("README.md", b"# Title\nBody") == "# Title\nBody"


def test_extract_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("data.docx", b"x")


def test_extract_empty_txt_returns_empty() -> None:
    assert extract_text("empty.txt", b"   ") == ""


# --- split_into_documents ---

def test_split_into_documents_metadata() -> None:
    docs = split_into_documents("alpha " * 500, source="big.txt", chunk_size=100, chunk_overlap=0)
    assert len(docs) >= 2
    assert all(d.metadata["source"] == "big.txt" for d in docs)
    assert docs[0].metadata["chunk_index"] == 0


def test_split_with_batch_id() -> None:
    docs = split_into_documents("hello world", source="f.txt", batch_id="abc123", chunk_size=5000, chunk_overlap=0)
    assert len(docs) == 1
    assert docs[0].metadata["batch_id"] == "abc123"
    assert docs[0].metadata["source"] == "f.txt"


def test_split_without_batch_id_omits_key() -> None:
    docs = split_into_documents("hello world", source="f.txt", chunk_size=5000, chunk_overlap=0)
    assert "batch_id" not in docs[0].metadata


def test_split_filters_empty_chunks() -> None:
    docs = split_into_documents("a", source="x.txt", chunk_size=5000, chunk_overlap=0)
    assert all(d.page_content.strip() for d in docs)


# --- Fixture: mock vector store + filter builder ---

@pytest.fixture()
def mock_store():
    mock = MagicMock()
    with patch.object(_qs_mod, "get_vector_store", return_value=mock), \
         patch.object(_ingest_mod, "_build_metadata_filter", return_value="FAKE_FILTER"):
        yield mock


# --- index_documents ---

def test_index_documents_calls_add_documents(mock_store: MagicMock) -> None:
    mock_store.add_documents.return_value = ["abc123"]
    docs = [Document(page_content="test chunk", metadata={"source": "t.txt", "chunk_index": 0})]

    ids = _ingest_mod.index_documents(docs)
    assert ids == ["abc123"]
    mock_store.add_documents.assert_called_once()


def test_index_documents_empty_list() -> None:
    assert _ingest_mod.index_documents([]) == []


def test_index_documents_rollback_on_failure(mock_store: MagicMock) -> None:
    """When add_documents raises before any batch succeeds, no rollback is needed."""
    mock_store.add_documents.side_effect = RuntimeError("network timeout")

    docs = [Document(page_content="chunk", metadata={"source": "x.txt", "chunk_index": 0})]
    with pytest.raises(RuntimeError, match="network timeout"):
        _ingest_mod.index_documents(docs)

    mock_store.delete.assert_not_called()


def test_index_documents_rollback_after_partial_batches(mock_store: MagicMock) -> None:
    """When batch 2 fails, batch 1 IDs are deleted via store.delete."""
    first_batch_size = _ingest_mod._INDEX_BATCH_SIZE
    total = first_batch_size + 5
    fixed_ids = [f"pt{i:04x}" for i in range(total)]

    class _FakeUUID:
        __slots__ = ("hex",)

        def __init__(self, h: str) -> None:
            self.hex = h

    docs = [
        Document(page_content=f"c{i}", metadata={"source": "big.txt", "chunk_index": i})
        for i in range(total)
    ]

    def add_side_effect(batch: list, ids: list | None = None, **kwargs: object) -> list:
        assert ids is not None
        if len(batch) == first_batch_size:
            return list(ids)
        raise RuntimeError("fail on second batch")

    mock_store.add_documents.side_effect = add_side_effect

    with patch.object(_ingest_mod.uuid, "uuid4", side_effect=[_FakeUUID(h) for h in fixed_ids]):
        with pytest.raises(RuntimeError, match="fail on second batch"):
            _ingest_mod.index_documents(docs)

    mock_store.delete.assert_called_once()
    call_kw = mock_store.delete.call_args.kwargs
    assert call_kw["ids"] == fixed_ids[:first_batch_size]


# --- delete functions ---

def test_delete_by_source_calls_qdrant_delete(mock_store: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.count.return_value = MagicMock(count=3)
    mock_store.client = mock_client

    deleted = _ingest_mod.delete_by_source("handbook.pdf")
    assert deleted == 3
    mock_client.count.assert_called_once()
    mock_client.delete.assert_called_once()


def test_delete_by_source_zero_matches(mock_store: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.count.return_value = MagicMock(count=0)
    mock_store.client = mock_client

    deleted = _ingest_mod.delete_by_source("nonexistent.txt")
    assert deleted == 0
    mock_client.delete.assert_not_called()


def test_delete_by_batch_id_calls_qdrant_delete(mock_store: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.count.return_value = MagicMock(count=5)
    mock_store.client = mock_client

    deleted = _ingest_mod.delete_by_batch_id("batchabc")
    assert deleted == 5
    mock_client.delete.assert_called_once()


# --- count_by_source ---

def test_count_by_source(mock_store: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.count.return_value = MagicMock(count=7)
    mock_store.client = mock_client

    assert _ingest_mod.count_by_source("notes.txt") == 7
    mock_client.count.assert_called_once()
