# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## demo/ (single-process deployable)

`demo/app.py` packages the real pipeline (`src/api/main.py`) plus a minimal
static frontend as one process, for platforms like Railway that want a
single service instead of `docker-compose.yml`'s three containers. It
imports `src.api.main:app` directly rather than reimplementing anything, so
pipeline changes there apply automatically. See `demo/README.md` for run
instructions and the documented fidelity gaps (no bundled Postgres, no
Gradio, demo-only rate limiting). Must run with the repo root as the working
directory (`uv run uvicorn demo.app:app` from repo root), same constraint as
`src/api/main.py`'s own relative config paths.

`demo/Dockerfile` needs `data/` baked into the image (Railway has no
volume mount, unlike `docker-compose.yml`'s `./data:/app/data`), so
`.dockerignore` carries a `!data/` exception right after its blanket
`data/` rule. Both `Dockerfile` and `demo/Dockerfile` build from the same
repo-root context and share that one `.dockerignore`.

Railway only reads `railway.toml`/`railway.json` at the repo root, never in
a subdirectory, so the live config is the root-level `railway.toml`
(`dockerfilePath = "demo/Dockerfile"`); `demo/railway.toml` is kept only as
a stale reference copy, same pattern used by sibling portfolio repos. `/health`
(`src/api/main.py`) is NOT safe to healthcheck before `DATABASE_URL` is a
reachable Postgres: `init_dependencies()` in `src/api/dependencies.py`
does a blocking `asyncpg.connect()` during the FastAPI `lifespan` startup
(before any endpoint, including `/health`, is servable), so an unreachable
DB fails the whole process at boot rather than being reported as a
"degraded" health response.

Vertex AI credentials on Railway can't use the interactive `gcloud auth
application-default login` flow local dev relies on. `demo/app.py` reads
`GOOGLE_APPLICATION_CREDENTIALS_JSON` (a service-account key as a JSON
string) before importing `src.api.main`, writes it to a 0600 temp file, and
points `GOOGLE_APPLICATION_CREDENTIALS` at it so Google's standard ADC
discovery picks it up; unset/blank falls through to normal ADC discovery
unchanged. See `demo/README.md`'s fidelity-gaps section and
`tests/test_demo/test_app.py`.

## data/ corpus PDFs

`.gitignore` blanket-excludes `data/*.pdf` (large/generated artifacts by
default) but allowlists specific committed corpus PDFs by name right after
that rule, same pattern as `.dockerignore`'s `!data/` exception. When adding
a new PDF to the corpus, add a matching `!data/<filename>.pdf` line, verify
it has an extractable text layer via `src/ingestion/loader.py`'s
`load_pdf()` (scanned-image-only PDFs fail silently — `load_documents()`
just logs a warning and skips them), then commit normally.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
