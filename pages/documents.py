"""Document management page for the CRAG Streamlit frontend."""

from __future__ import annotations

import streamlit as st

from crag.frontend_client import (
    CRAGApiClient,
    CRAGApiError,
    IngestResult,
    UploadFilePayload,
)
from crag.frontend_state import StateKey


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> CRAGApiClient:
    """Cache the API client per backend URL to reuse HTTP connections."""
    return CRAGApiClient(base_url)


def render_api_error(error: CRAGApiError) -> None:
    label = f"API error {error.status_code}" if error.status_code else "API connection error"
    st.error(f"{label}: {error}")


def render_ingest_result(result: IngestResult | None) -> None:
    with st.container(border=True):
        st.subheader("Latest ingest result")
        if result is None:
            st.info("No ingest operation has completed in this session.")
            return

        metric_col, source_col = st.columns([0.32, 0.68], gap="large")
        with metric_col:
            st.metric("Chunks indexed", result.chunks_indexed)
            st.metric("Sources", len(result.sources))
            st.caption("Batch ID")
            st.code(result.batch_id, language="text")
        with source_col:
            st.caption("Indexed sources")
            for source in result.sources:
                st.write(f"- `{source}`")
            if result.replaced_sources:
                st.warning(
                    "Existing chunks were replaced for: "
                    + ", ".join(f"`{source}`" for source in result.replaced_sources)
                )
            with st.expander("Point IDs"):
                st.json(result.point_ids)


def _uploaded_payloads(uploaded_files: list[object]) -> list[UploadFilePayload]:
    payloads: list[UploadFilePayload] = []
    for uploaded_file in uploaded_files:
        payloads.append(
            UploadFilePayload(
                filename=uploaded_file.name,
                content=uploaded_file.getvalue(),
                content_type=uploaded_file.type or "application/octet-stream",
            )
        )
    return payloads


@st.fragment
def render_documents_fragment() -> None:
    """Run ingest/delete operations inside a fragment because they can be slow."""
    client = get_api_client(st.session_state[StateKey.API_BASE_URL])

    upload_col, text_col = st.columns(2, gap="large")

    with upload_col:
        with st.container(border=True):
            st.subheader("Upload files")
            st.caption("Use for `.txt`, `.md`, `.markdown`, and `.pdf` files that should be chunked and indexed.")
            with st.form("upload_documents_form", clear_on_submit=True):
                uploaded_files = st.file_uploader(
                    "Files",
                    type=["txt", "md", "markdown", "pdf"],
                    accept_multiple_files=True,
                )
                upload_submit = st.form_submit_button("Upload and index", type="primary", use_container_width=True)

            if upload_submit:
                if not uploaded_files:
                    st.warning("Select at least one file to upload.")
                else:
                    with st.spinner("Extracting text, chunking documents, and indexing vectors..."):
                        try:
                            result = client.upload_documents(_uploaded_payloads(uploaded_files))
                        except CRAGApiError as error:
                            render_api_error(error)
                        else:
                            st.session_state[StateKey.LAST_INGEST_RESULT] = result
                            st.success(f"Indexed {result.chunks_indexed} chunks.")

    with text_col:
        with st.container(border=True):
            st.subheader("Paste text")
            st.caption("Use for quick analyst notes, copied docs, or generated test content.")
            with st.form("ingest_text_form", clear_on_submit=True):
                source = st.text_input("Source label", value="inline_text")
                text = st.text_area("Text", height=178)
                text_submit = st.form_submit_button("Index text", type="primary", use_container_width=True)

            if text_submit:
                cleaned_text = text.strip()
                cleaned_source = source.strip()
                if not cleaned_source:
                    st.warning("Source label is required.")
                elif not cleaned_text:
                    st.warning("Paste text before indexing.")
                else:
                    with st.spinner("Chunking text and indexing vectors..."):
                        try:
                            result = client.ingest_text(cleaned_text, source=cleaned_source)
                        except CRAGApiError as error:
                            render_api_error(error)
                        else:
                            st.session_state[StateKey.LAST_INGEST_RESULT] = result
                            st.success(f"Indexed {result.chunks_indexed} chunks.")

    render_ingest_result(st.session_state[StateKey.LAST_INGEST_RESULT])

    st.subheader("Cleanup")
    delete_source_col, delete_batch_col = st.columns(2, gap="large")
    with delete_source_col:
        with st.container(border=True):
            st.caption("Delete every chunk for a source label, such as an uploaded filename.")
            with st.form("delete_source_form"):
                source_to_delete = st.text_input("Source to delete")
                delete_source_submit = st.form_submit_button("Delete by source", use_container_width=True)
            if delete_source_submit:
                cleaned_source = source_to_delete.strip()
                if not cleaned_source:
                    st.warning("Enter a source label to delete.")
                else:
                    with st.spinner("Deleting source chunks..."):
                        try:
                            result = client.delete_source(cleaned_source)
                        except CRAGApiError as error:
                            render_api_error(error)
                        else:
                            st.success(f"Deleted {result.deleted_count} chunks for `{cleaned_source}`.")

    with delete_batch_col:
        with st.container(border=True):
            st.caption("Delete one ingest batch using the batch id returned after upload or text ingest.")
            with st.form("delete_batch_form"):
                batch_to_delete = st.text_input("Batch ID to delete")
                delete_batch_submit = st.form_submit_button("Delete by batch", use_container_width=True)
            if delete_batch_submit:
                cleaned_batch = batch_to_delete.strip()
                if not cleaned_batch:
                    st.warning("Enter a batch id to delete.")
                else:
                    with st.spinner("Deleting batch chunks..."):
                        try:
                            result = client.delete_batch(cleaned_batch)
                        except CRAGApiError as error:
                            render_api_error(error)
                        else:
                            st.success(f"Deleted {result.deleted_count} chunks for batch `{cleaned_batch}`.")


st.title("Documents")
st.write(
    "Upload files, index pasted text, and remove stale chunks from the Qdrant collection "
    "used by the CRAG retriever."
)
render_documents_fragment()
