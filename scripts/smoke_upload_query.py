"""Smoke test: upload test_data/test.md and POST /query (run from crag-system root).

Usage (local Qdrant):
  set QDRANT_URL=http://127.0.0.1:6333
  set QDRANT_API_KEY=
  python scripts/smoke_upload_query.py

Or rely on .env / parent .env for keys; override QDRANT_URL for Docker Qdrant.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Default local Qdrant if not set (docker compose)
if not os.environ.get("QDRANT_URL"):
    os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"
if "QDRANT_API_KEY" not in os.environ:
    os.environ["QDRANT_API_KEY"] = ""

_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crag.config import get_settings

get_settings.cache_clear()
import crag.qdrant_store as qsmod

qsmod._store = None

from fastapi.testclient import TestClient

from api.main import app


def main() -> int:
    md_path = _ROOT / "test_data" / "test.md"
    if not md_path.is_file():
        print("Missing", md_path, file=sys.stderr)
        return 1

    with TestClient(app) as client:
        r = client.get("/health")
        print("health", r.status_code, r.json())

        content = md_path.read_bytes()
        up = client.post(
            "/documents/upload",
            files=[("files", ("test.md", content, "text/markdown"))],
        )
        print("upload", up.status_code)
        if up.status_code != 200:
            print(up.text, file=sys.stderr)
            return 1
        print(json.dumps(up.json(), indent=2)[:500])

        q = client.post(
            "/query",
            json={
                "query": "What is Acme Corp official snack policy about bananas?",
                "user_id": "smoke-script",
            },
        )
        print("query", q.status_code)
        if q.status_code != 200:
            print(q.text, file=sys.stderr)
            return 1
        body = q.json()
        print("answer:", body.get("answer", "")[:400])
        print("sources:", body.get("sources"))
        print("routing:", body.get("routing_decision"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
