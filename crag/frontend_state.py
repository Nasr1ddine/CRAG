"""Session-state keys and defaults for the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_API_BASE_URL = "http://localhost:8000"


class StateKey:
    """Named keys used with ``st.session_state``."""

    API_BASE_URL = "crag_api_base_url"
    CHAT_HISTORY = "crag_chat_history"
    LAST_INGEST_RESULT = "crag_last_ingest_result"
    USER_ID = "crag_user_id"


@dataclass
class ChatTurn:
    """One submitted query and its rendered API result."""

    question: str
    result: Any


@dataclass(frozen=True)
class StateDefaults:
    """Initial Streamlit session values."""

    api_base_url: str = DEFAULT_API_BASE_URL
    chat_history: list[ChatTurn] = field(default_factory=list)
    last_ingest_result: Any | None = None
    user_id: str = "analyst"


def default_state_values() -> dict[str, Any]:
    """Return a fresh mapping of keys to default values."""
    defaults = StateDefaults()
    return {
        StateKey.API_BASE_URL: defaults.api_base_url,
        StateKey.CHAT_HISTORY: defaults.chat_history,
        StateKey.LAST_INGEST_RESULT: defaults.last_ingest_result,
        StateKey.USER_ID: defaults.user_id,
    }
