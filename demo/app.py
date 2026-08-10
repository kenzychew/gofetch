"""Single-process demo entrypoint: the real GoFetch RAG API plus a static UI.

Reuses `src.api.main:app` unmodified (same /query, /ingest, /health routes,
same lifespan-managed pipeline) and adds two demo-only concerns on top: a
per-IP rate limit on the expensive endpoints, and a static file mount for the
minimal frontend in demo/static/.

Must be run with the repo root as the working directory (same requirement as
`uv run uvicorn src.api.main:app` today), because src/api/main.py resolves
configs/, data/, bm25_index/, and graph_data/ as relative paths:

    uv run uvicorn demo.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.api.main import app  # noqa: E402

RATE_LIMITED_PATHS = frozenset({"/query", "/ingest"})
RATE_LIMIT_PER_MINUTE = int(os.environ.get("DEMO_RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_WINDOW_SECONDS = 60.0


class RateLimitMiddleware:
    """Pure-ASGI per-client-IP fixed-window rate limiter.

    Implemented as a raw ASGI middleware (not Starlette's BaseHTTPMiddleware)
    so it never buffers the response -- required for /query's SSE stream to
    keep flushing tokens incrementally instead of arriving all at once.
    """

    def __init__(self, asgi_app: object, limit: int, window_seconds: float) -> None:
        self.app = asgi_app
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        if scope["type"] != "http" or scope["path"] not in RATE_LIMITED_PATHS:
            await self.app(scope, receive, send)
            return

        client_ip = self._client_ip(scope)
        now = time.monotonic()
        cutoff = now - self.window_seconds
        hits = self._hits[client_ip]
        while hits and hits[0] < cutoff:
            hits.pop(0)

        if len(hits) >= self.limit:
            retry_after = max(1, int(self.window_seconds - (now - hits[0])) + 1)
            response = JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Please try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        hits.append(now)
        await self.app(scope, receive, send)

    @staticmethod
    def _client_ip(scope: dict) -> str:
        headers = dict(scope.get("headers") or [])
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            return forwarded.decode().split(",")[0].strip()
        client = scope.get("client")
        return client[0] if client else "unknown"


app.add_middleware(
    RateLimitMiddleware,
    limit=RATE_LIMIT_PER_MINUTE,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="demo-static")
