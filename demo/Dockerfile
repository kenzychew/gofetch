# Build context must be the repo root (not demo/), because this image needs
# src/, configs/, and data/ from alongside demo/ -- see demo/README.md.
#
#   docker build -f demo/Dockerfile -t gofetch-demo .

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

COPY demo/requirements-demo.txt demo/requirements-demo.txt
RUN pip install -r demo/requirements-demo.txt

COPY src ./src
COPY configs ./configs
COPY data ./data
COPY demo ./demo

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
