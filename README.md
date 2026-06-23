# CRAG System

Production-grade Self-Correcting Retrieval-Augmented Generation (CRAG) pipeline.

## Architecture

1. Hybrid retrieval from Qdrant (BM25 + dense)
2. Retrieval grading (RELEVANT, AMBIGUOUS, IRRELEVANT)
3. Routing:
  - all IRRELEVANT -> web search fallback (Tavily)
  - no RELEVANT and any AMBIGUOUS -> rewrite query and re-retrieve (bounded loop)
  - at least one RELEVANT -> continue
4. Context refinement for ambiguous docs
5. Grounded generation from verified context only
6. Faithfulness judging of answer against context
7. Per-request tracing with Langfuse

## Stack

- LangGraph, LangChain
- Qdrant Cloud
- OpenAI models and embeddings
- Tavily web search
- RAGAS evaluation
- FastAPI + Pydantic v2
- Streamlit frontend
- Poetry + Docker + Render

## Setup

### 1) Install

With Poetry:

```bash
poetry install
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Fill `.env` with valid API keys and settings.

**Where `.env` is read from:** `crag-system/.env`, then the parent directory’s `.env` (e.g. repo root), so a single file can sit next to `crag-system/` if you prefer.

Required settings:

- `OPENAI_API_KEY`
- `QDRANT_URL`

Optional settings:

- `QDRANT_API_KEY`
- `COLLECTION_NAME` (default: `crag_docs`)
- `TAVILY_API_KEY`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `TOP_K` (default: `5`)
- `MAX_REWRITE_ITERATIONS` (default: `2`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_UPLOAD_BYTES`
- `GRADER_MODEL`, `GENERATOR_MODEL`
- `CRAG_API_PORT` (default: `8001` for Docker Compose host access)

**Local Qdrant (optional):** if your `QDRANT_URL` points at the wrong host, uploads and queries will fail against Qdrant. For a local instance:

```bash
docker compose up -d
```

Set `QDRANT_URL=http://127.0.0.1:6333` and leave `QDRANT_API_KEY` empty (or unset).

**Optional keys:** `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` may be left empty; web search fallback and Langfuse callbacks are skipped when unset.

**Windows / Poetry:** if `poetry run uvicorn` or `poetry run pytest` fails with `ImportError: DLL load failed while importing _ssl`, your Poetry environment is using a broken Python build. Point Poetry at a working interpreter, for example: `poetry env use path\to\python.exe`, then reinstall.

## Run API

```bash
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Browser file upload (multipart): open [http://localhost:8000/upload](http://localhost:8000/upload) to choose `.txt` / `.md` / `.pdf` files — avoids sending a path string in JSON.

API docs are served at [http://localhost:8000/docs](http://localhost:8000/docs). The app uses bundled Swagger UI JavaScript/CSS when present and falls back to CDN-hosted assets when needed.

## Run Streamlit Inspector

The Streamlit frontend runs as a separate process and calls the FastAPI backend over HTTP. Start the API first, then launch the inspector:

```bash
poetry run streamlit run app.py
```

If the API is running through Docker Compose, set the sidebar API base URL to `http://localhost:8001`. For a local `uvicorn` process, use `http://localhost:8000`.

The inspector provides:

- Query workspace with answer, route, rewrite, source, and faithfulness diagnostics.
- Document workspace for multipart upload, pasted-text ingest, and deletion by source or batch id.
- System workspace for cached `/health` and safe `/debug/config` checks.

The Streamlit UI keeps all business logic in plain Python modules and uses fragments around slow query/ingest/delete actions so routine page navigation does not repeat expensive backend calls.

Health check:

```bash
curl http://localhost:8000/health
```

Query example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the remote work policy?","user_id":"u-123"}'
```

Quick check (upload `test_data/test.md` and run one query; requires working Qdrant + `OPENAI_API_KEY`):

```bash
# optional: docker compose up -d  then  export QDRANT_URL=http://127.0.0.1:6333
poetry run python scripts/smoke_upload_query.py
```

## Evaluation

Run CRAG vs naive baseline with RAGAS:

```bash
poetry run python -m evaluation.ragas_eval
```

Outputs metrics and saves `evaluation/results.json`.

## Tests

```bash
poetry run pytest -v
```

## Docker

```bash
docker build -t crag-system .
docker run --env-file .env -p 8000:8000 crag-system
```

Docker Compose runs the API, Streamlit inspector, and Qdrant:

```bash
docker compose up --build
```

By default, Compose publishes:

- Streamlit inspector: `http://localhost:8501`
- CRAG API: `http://localhost:8001`
- Qdrant: `http://localhost:6333`

Inside Compose, the Streamlit service talks to the API at `http://crag-api:8000` through `CRAG_FRONTEND_API_BASE_URL`. Set `CRAG_STREAMLIT_PORT=8502`, `CRAG_API_PORT=8080`, or another free host port in `.env` if needed.

## Deploy on Dokploy

Use `docker-compose.yml` as the Dokploy Compose source. It defines three services:

- `crag-api` serves the FastAPI backend on container port `8000`.
- `crag-streamlit` serves the Streamlit inspector frontend on container port `8501`.
- `qdrant` provides the local vector store on container port `6333`.

Configure secret values (`OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `TAVILY_API_KEY`, and Langfuse keys) in Dokploy. The Streamlit service uses `CRAG_FRONTEND_API_BASE_URL`; by default it calls the API through Docker networking at `http://crag-api:8000`.