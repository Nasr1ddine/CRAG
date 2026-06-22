"""Centralised application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


def normalize_qdrant_url(url: str) -> str:
    """Return Qdrant REST base URL: strip paths, default port 6333 when missing.

    Qdrant Cloud serves the HTTP API on port 6333. A URL without ``:6333`` hits
    443 instead and often returns ``404 page not found``.
    """
    u = url.strip().rstrip("/")
    parsed = urlparse(u)
    if not parsed.scheme or not parsed.hostname:
        return u
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    host_l = hostname.lower()
    port = parsed.port
    if port is None:
        if "cloud.qdrant.io" in host_l or host_l.endswith(".qdrant.io"):
            port = 6333
        elif host_l in ("localhost", "127.0.0.1", "0.0.0.0"):
            port = 6333
    netloc = f"{hostname}:{port}" if port is not None else parsed.netloc
    return urlunparse((scheme, netloc, "", "", "", ""))

_CONFIG_DIR = Path(__file__).resolve().parent.parent
_REPO_PARENT = _CONFIG_DIR.parent
_DOTENV_FILES = tuple(
    str(p)
    for p in (_REPO_PARENT / ".env", _CONFIG_DIR / ".env")
    if p.is_file()
)


class Settings(BaseSettings):
    """Application-wide configuration backed by env vars / .env file."""

    model_config = ConfigDict(
        env_file=_DOTENV_FILES if _DOTENV_FILES else (".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str = ""
    collection_name: str = "crag_docs"
    tavily_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    top_k: int = 5
    max_rewrite_iterations: int = 2

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_bytes: int = 15 * 1024 * 1024

    grader_model: str = "gpt-4o-mini"
    generator_model: str = "gpt-4o"

    @field_validator("qdrant_url")
    @classmethod
    def _normalize_qdrant_url(cls, v: str) -> str:
        return normalize_qdrant_url(v)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (created on first call)."""
    return Settings()  # type: ignore[call-arg]


class _SettingsProxy:
    """Lazy proxy so ``settings`` can be imported without triggering
    validation until the first attribute access. This lets tests and
    Docker builds import modules before ``.env`` is available."""

    def __getattr__(self, name: str) -> object:
        return getattr(get_settings(), name)


settings: Settings = _SettingsProxy()  # type: ignore[assignment]
