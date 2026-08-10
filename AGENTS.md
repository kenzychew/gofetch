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

Railway's build-plan step only reads `railway.toml` at the **repo root**,
never inside subdirectories, so the actual config lives at `/railway.toml`
(pointing at `demo/Dockerfile`). `demo/railway.toml` is a stale pointer
comment only — don't edit it expecting Railway to see it.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
