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

`meta.confidence` in `/query`'s SSE `metadata` event (`src/api/main.py`,
top reranked chunk's score) is an unbounded cross-encoder logit, not a
0-1 probability; negative values are normal and don't mean the answer is
bad. `demo/static/app.js`'s `renderTechDetails()` (formerly
`renderLatency()`, renamed once it started covering confidence too) shows
a qualitative "Low"/"Typical" label derived from the backend's own
`low_confidence` flag, with the raw score kept alongside for technical
visitors, inside a `<details>` disclosure (`#latency-section` in
`index.html`) that stays collapsed by default so it doesn't read as an
alarm sitting next to a correct answer.

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

`bm25_index_path` (`src/config.py`, `AppConfig.__post_init__`) has its
directory overridden by the `BM25_INDEX_DIR` env var when set, keeping the
configured filename; unset falls through to the relative `bm25_index/`
local-dev default unchanged. This exists because Railway containers are
ephemeral and wipe that relative path on every redeploy, silently dropping
the sparse half of hybrid search until a manual re-ingest -- attaching an
actual Railway volume at the path `BM25_INDEX_DIR` points to is a separate
infra step, not done by this env var alone. `init_dependencies`
(`src/api/dependencies.py`) distinguishes, via clear log lines, "no index
built yet" (pickle file missing -- normal on a fresh volume before the
first `/ingest`) from "index file present but failed to load" (corrupt
pickle -- an `IndexingError` from `BM25Indexer.load_index`); neither case
crashes startup, sparse retrieval just stays unavailable until the next
successful `/ingest`.

`POST /ingest` (`src/api/main.py`) validates synchronously (saves uploads,
loads documents, 400s if none found) then queues chunking, embedding,
Postgres upsert, BM25 build, and knowledge-graph extraction as a
`BackgroundTasks` job and returns 202 with a job id immediately, instead of
blocking until graph extraction finishes -- graph extraction is sequential
per-chunk Gemini calls and took long enough to make Railway's edge 502 the
request while the service kept serving other traffic fine. Poll
`GET /ingest/status/{job_id}` for the current `IngestStage`
(`queued`/`chunking`/`embedding`/`indexing`/`graph-extraction`/`done`/`failed`)
and any error; job state lives in an in-memory dict (`_ingest_jobs`) keyed
by job id, fine at this scale but not persisted across restarts. `ui/app.py`
(the Gradio frontend for the three-container `docker-compose.yml` stack)
polls this endpoint after posting to `/ingest` rather than waiting on one
long HTTP call; `demo/static/app.js` has no `/ingest` call at all today, so
it needed no change here.

`GET /sources` (`src/api/main.py`) returns the distinct `source` values
and per-source chunk counts currently in the `chunks` table, reflecting
what's actually been ingested rather than what's on disk in `data/` (those
drift apart if `/ingest` hasn't been re-run since a corpus change).
`demo/static/app.js` fetches it on page load to show visitors what the
corpus covers before they type a query, grouped by file extension
(`.pdf` -> ML research papers, everything else -> Singapore government
schemes) with filenames rendered as readable titles via a small
acronym-aware title-case heuristic (short all-consonant-ish tokens like
`cpf`/`hdb`/`mas` get upper-cased, common short English words don't); see
`readableSourceTitle()` in that file if the corpus grows past this
two-group split and the grouping heuristic needs revisiting. Each entry
links to its actual file via a second `StaticFiles` mount at
`/corpus-files` (`demo/app.py`) serving `data/` read-only; that mount must
be registered before the catch-all `app.mount("/", ...)` for
`demo/static/`, since Starlette resolves mounts in registration order and
the catch-all would otherwise shadow it.

`demo/static/style.css` and `index.html` use the same warm portfolio
palette/font tokens (`--color-bg`, `--color-accent`, Fraunces/Public
Sans/JetBrains Mono) as the GotParking and RocketML demo frontends, no
`prefers-color-scheme` auto-switching -- see `style.css`'s `:root` for the
authoritative token values, don't hardcode colors elsewhere. Every page
section is wrapped in an element with the shared `.panel` class
(`--color-bg-raised` background, bordered, rounded) so new sections should
follow that pattern rather than sitting directly on the page background;
the query form and example-question chips live together inside
`#ask-section`, one panel, not two.

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
