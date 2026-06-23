"""System diagnostics page for the CRAG Streamlit frontend."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from crag.frontend_client import CRAGApiClient, CRAGApiError, DebugConfig
from crag.frontend_state import StateKey

DIAGNOSTIC_TTL_SECONDS = 30


@st.cache_data(ttl=DIAGNOSTIC_TTL_SECONDS, show_spinner=False)
def fetch_health(base_url: str) -> dict[str, str]:
    """Cache health checks briefly so deploy changes show up quickly."""
    client = CRAGApiClient(base_url)
    try:
        return client.health()
    finally:
        client.close()


@st.cache_data(ttl=DIAGNOSTIC_TTL_SECONDS, show_spinner=False)
def fetch_debug_config(base_url: str) -> DebugConfig:
    """Cache safe config diagnostics briefly because env changes may be recent."""
    client = CRAGApiClient(base_url)
    try:
        return client.debug_config()
    finally:
        client.close()


def render_api_error(error: CRAGApiError) -> None:
    label = f"API error {error.status_code}" if error.status_code else "API connection error"
    st.error(f"{label}: {error}")


def render_secret_presence(label: str, present: bool) -> None:
    status = "Configured" if present else "Missing"
    st.metric(label, status)


st.title("System")
st.write(
    "Check backend reachability and safe runtime configuration values. Results are cached "
    f"for {DIAGNOSTIC_TTL_SECONDS} seconds so transient deploy issues remain visible."
)

base_url = st.session_state[StateKey.API_BASE_URL]
refresh_col, _ = st.columns([0.2, 0.8])
with refresh_col:
    if st.button("Refresh diagnostics", use_container_width=True):
        fetch_health.clear()
        fetch_debug_config.clear()
        st.rerun()

health_col, config_col = st.columns([0.35, 0.65], gap="large")

with health_col:
    with st.container(border=True):
        st.subheader("Backend health")
        st.caption(f"Target: `{base_url}`")
        with st.spinner("Checking backend health..."):
            try:
                health = fetch_health(base_url)
            except CRAGApiError as error:
                render_api_error(error)
            else:
                if health.get("status") == "ok":
                    st.success("Backend is reachable.")
                else:
                    st.warning("Backend responded, but did not report an ok status.")
                st.json(health)

with config_col:
    with st.container(border=True):
        st.subheader("Runtime config")
        st.caption("Only non-secret values and secret-presence booleans are exposed by the API.")
        with st.spinner("Loading safe config snapshot..."):
            try:
                config = fetch_debug_config(base_url)
            except CRAGApiError as error:
                render_api_error(error)
            else:
                value_col, secret_col = st.columns(2, gap="large")
                with value_col:
                    st.caption("Vector store")
                    st.write(f"Qdrant URL: `{config.qdrant_url}`")
                    st.write(f"Collection: `{config.collection_name}`")
                with secret_col:
                    render_secret_presence("OpenAI key", config.has_openai_api_key)
                    render_secret_presence("Qdrant key", config.has_qdrant_api_key)
                    render_secret_presence("Tavily key", config.has_tavily_api_key)
                    render_secret_presence(
                        "Langfuse keys",
                        config.has_langfuse_public_key and config.has_langfuse_secret_key,
                    )
                with st.expander("Raw safe config"):
                    st.json(asdict(config))

with st.container(border=True):
    st.subheader("Operational notes")
    st.write(
        "Query responses and ingest/delete operations are intentionally not cached because they "
        "represent fresh LLM output or side effects. Health and config diagnostics use a short "
        "TTL because deployment settings can change while an internal operator is debugging."
    )
