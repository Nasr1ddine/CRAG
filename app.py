"""Streamlit entry point for the CRAG RAG Inspector."""

from __future__ import annotations

import streamlit as st

from crag.frontend_client import CRAGApiClient, CRAGApiError
from crag.frontend_state import StateKey, default_state_values, normalize_api_base_url


st.set_page_config(
    page_title="CRAG Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Install explicit defaults without overwriting an active session."""
    for key, value in default_state_values().items():
        st.session_state.setdefault(key, value)
    st.session_state[StateKey.API_BASE_URL] = normalize_api_base_url(
        st.session_state[StateKey.API_BASE_URL]
    )


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> CRAGApiClient:
    """Cache the HTTP client so reruns reuse the same connection pool."""
    return CRAGApiClient(base_url)


def active_client() -> CRAGApiClient:
    """Return the API client for the currently configured backend URL."""
    return get_api_client(st.session_state[StateKey.API_BASE_URL])


def render_api_error(error: CRAGApiError) -> None:
    """Display backend failures without exposing a raw traceback."""
    label = f"API error {error.status_code}" if error.status_code else "API connection error"
    st.error(f"{label}: {error}")


def normalize_sidebar_api_url() -> None:
    """Keep user-edited sidebar values valid before pages use them."""
    st.session_state[StateKey.API_BASE_URL] = normalize_api_base_url(
        st.session_state[StateKey.API_BASE_URL]
    )


def render_sidebar() -> None:
    """Render global controls shared by all pages."""
    with st.sidebar:
        st.title("CRAG Inspector")
        st.caption("Internal workspace for querying and maintaining the CRAG pipeline.")

        st.text_input(
            "API base URL",
            key=StateKey.API_BASE_URL,
            help="FastAPI backend URL, for example http://localhost:8000 or http://localhost:8001.",
            on_change=normalize_sidebar_api_url,
        )
        st.text_input(
            "User ID for tracing",
            key=StateKey.USER_ID,
            help="Sent with queries so backend traces can be grouped by analyst.",
        )

        st.divider()
        st.caption("Navigation uses Streamlit pages because query, document, and system workflows have distinct state and intent.")


def main() -> None:
    initialize_session_state()
    render_sidebar()

    pages = [
        st.Page("pages/query.py", title="Query Workspace"),
        st.Page("pages/documents.py", title="Documents"),
        st.Page("pages/system.py", title="System"),
    ]
    selected_page = st.navigation(pages, position="sidebar")
    selected_page.run()


if __name__ == "__main__":
    main()
