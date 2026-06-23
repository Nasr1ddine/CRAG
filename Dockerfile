FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.4 \
    && poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main --no-root

COPY . .

# ── runtime stage ──
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
