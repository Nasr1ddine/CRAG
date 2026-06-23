"""Query workspace for the CRAG Streamlit frontend."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

from crag.frontend_client import CRAGApiClient, CRAGApiError, QueryResult
from crag.frontend_state import ChatTurn, StateKey


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> CRAGApiClient:
    """Cache the API client per backend URL to reuse HTTP connections."""
    return CRAGApiClient(base_url)


def render_api_error(error: CRAGApiError) -> None:
    label = f"API error {error.status_code}" if error.status_code else "API connection error"
    st.error(f"{label}: {error}")


def render_result(turn: ChatTurn, *, expanded: bool) -> None:
    result: QueryResult = turn.result
    with st.container(border=True):
        st.caption("Question")
        st.markdown(turn.question)

        answer_col, metric_col = st.columns([0.68, 0.32], gap="large")
        with answer_col:
            st.subheader("Answer")
            st.markdown(result.answer or "_The backend returned an empty answer._")
        with metric_col:
            st.subheader("Diagnostics")
            st.metric("Faithfulness", f"{result.faithfulness_score:.2f}")
            st.metric("Route", result.routing_decision or "unknown")
            st.metric("Iterations", result.iteration_count)
            if result.query_used and result.query_used != turn.question:
                st.caption("Query used after rewrite")
                st.code(result.query_used, language="text")

        sources_tab, faithfulness_tab, raw_tab = st.tabs(["Sources", "Faithfulness", "Raw response"])
        with sources_tab:
            if result.sources:
                for source in result.sources:
                    st.write(f"- `{source}`")
            else:
                st.info("No source metadata was returned for this answer.")
        with faithfulness_tab:
            if result.faithfulness_issues:
                for issue in result.faithfulness_issues:
                    st.warning(issue)
            else:
                st.success("No faithfulness issues were reported.")
        with raw_tab:
            st.json(asdict(result), expanded=expanded)


@st.fragment
def render_query_fragment() -> None:
    """Run slow LLM-backed queries inside a fragment to avoid full-page reruns."""
    client = get_api_client(st.session_state[StateKey.API_BASE_URL])
    chat_history: list[ChatTurn] = st.session_state[StateKey.CHAT_HISTORY]

    with st.container(border=True):
        st.subheader("Ask the CRAG pipeline")
        st.caption("Use a form so composing a question does not call the backend on every keystroke.")
        with st.form("query_form", clear_on_submit=True):
            question = st.text_area(
                "Question",
                placeholder="Ask about the indexed corpus, policies, product docs, or uploaded files.",
                height=120,
            )
            submit = st.form_submit_button("Run query", type="primary", use_container_width=True)

        if submit:
            cleaned_question = question.strip()
            if not cleaned_question:
                st.warning("Enter a question before running the pipeline.")
            else:
                with st.spinner("Running retrieval, grading, generation, and faithfulness checks..."):
                    try:
                        result = client.query(
                            cleaned_question,
                            user_id=st.session_state[StateKey.USER_ID].strip() or None,
                        )
                    except CRAGApiError as error:
                        render_api_error(error)
                    else:
                        chat_history.append(ChatTurn(question=cleaned_question, result=result))
                        st.success("Query completed.")

    if not chat_history:
        with st.container(border=True):
            st.subheader("No queries yet")
            st.write(
                "Ask a question to see the generated answer, route decision, faithfulness score, "
                "and source list in one reviewable workspace."
            )
        return

    action_col, _ = st.columns([0.2, 0.8])
    with action_col:
        if st.button("Clear history", use_container_width=True):
            chat_history.clear()
            st.rerun(scope="fragment")

    st.subheader("Query history")
    for index, turn in enumerate(reversed(chat_history), start=1):
        render_result(turn, expanded=index == 1)


st.title("Query Workspace")
st.write(
    "Run questions against CRAG and inspect the answer, routing choice, rewrite behavior, "
    "sources, and faithfulness checks."
)
render_query_fragment()
