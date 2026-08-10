# GoFetch demo

A single-process packaging of the real RAG pipeline (`src/api/main.py`) plus
a minimal static frontend (`demo/static/`), meant to run as one deployable
service instead of the three-container `docker-compose.yml` setup.

`demo/app.py` does not reimplement retrieval or generation -- it imports
`src.api.main:app` as-is (same `/query`, `/ingest`, `/health` routes, same
lifespan-managed pipeline startup) and adds two demo-only concerns: a per-IP
rate limit on `/query` and `/ingest`, and a `StaticFiles` mount serving
`demo/static/` at `/`.

## Running locally

Same prerequisites as the root README's "Local development" section
(Postgres with pgvector, Vertex AI credentials), run from the repo root:

```bash
# 1. Start Postgres (only the DB -- the demo replaces backend + frontend)
docker compose up postgres -d

# 2. Install demo deps into the project's venv
uv pip install -r demo/requirements-demo.txt

# 3. Authenticate for Vertex AI, same as the root README
gcloud auth application-default login
cp .env.example .env   # then edit DATABASE_URL / GCP_PROJECT / GCP_REGION

# 4. Run the demo (must run from repo root -- see demo/app.py docstring)
uv run uvicorn demo.app:app --reload --port 8000

# 5. Ingest the corpus once (same as today -- this doesn't rebuild indexes,
#    it just triggers the existing /ingest endpoint against data/)
curl -X POST http://localhost:8000/ingest

# 6. Open http://localhost:8000
```

Or with Docker (build context must be the repo root):

```bash
docker build -f demo/Dockerfile -t gofetch-demo .
docker run --rm -p 8000:8000 --env-file .env gofetch-demo
```

## Fidelity gaps vs. docker-compose.yml

`docker-compose.yml` runs three containers: `postgres`, `backend`, and a
Gradio `frontend`. Folding this into one Railway service means:

- **No bundled Postgres.** pgvector storage is a stateful dependency that
  doesn't belong inside a stateless app container. `demo/` expects
  `DATABASE_URL` to point at an already-running, already-ingested Postgres
  (a managed Railway Postgres plugin, Neon, etc.) -- provisioning that is
  explicitly out of scope for this task (see `demo/railway.toml`).
- **Gradio replaced with a static page.** `ui/app.py`'s tabs (Query + Ingest
  with file upload) become one query box in `demo/static/`. File-upload
  ingestion still works (the `/ingest` endpoint is untouched and reachable),
  it's just not exposed in the demo UI -- the intent is "query the existing
  indexed corpus," not "let anonymous visitors re-ingest it."
- **No GPU dependency.** Neither the embedder (`all-MiniLM-L6-v2`) nor the
  reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) needs one; both are small
  enough to run on CPU within normal request latency. `requirements-demo.txt`
  pins the CPU-only torch wheel specifically to avoid pulling in unused CUDA
  libraries and bloating the image.
- **Rate limiting is new and demo-specific.** `src/api/main.py` has none
  (it's designed to sit behind a local Docker network). `demo/app.py` adds a
  simple per-IP fixed-window limiter (`DEMO_RATE_LIMIT_PER_MINUTE`, default
  10/min) on `/query` and `/ingest` only, since `/query` calls a paid Gemini
  backend and `/ingest` triggers embedding + optional graph-extraction LLM
  calls.

## Known follow-up (not done here)

Actually wiring this to Railway (provisioning Postgres, running the initial
`/ingest`, setting secrets) is a separate follow-up task, per the brief.
