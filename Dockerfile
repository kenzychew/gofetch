FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

COPY . .

FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

COPY --from=builder /app /app

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
