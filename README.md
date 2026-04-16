# GoFetch

RAG pipeline with hybrid search, re-ranking, and LLM-powered answer generation with inline citations.

## What it does

Upload documents (PDF/text), ask questions in natural language, get answers grounded in your documents with source citations. The system combines three retrieval signals (keyword search, semantic vectors, knowledge graph), re-ranks candidates with a cross-encoder, and streams a cited answer via Gemini.

## Tech stack

- **Backend**: Python 3.11, FastAPI, asyncpg
- **Vector DB**: PostgreSQL with pgvector (HNSW index)
- **Sparse search**: BM25 via rank-bm25
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Re-ranker**: cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Knowledge graph**: NetworkX with LLM-based entity extraction
- **LLM**: Google Gemini (Vertex AI) with SSE streaming
- **Frontend**: Gradio
- **Deployment**: Docker Compose

## Project structure

```
src/
  api/
    main.py              # FastAPI app, /ingest /query /health endpoints
    dependencies.py      # DI container, asyncpg pool lifecycle
  ingestion/
    loader.py            # PDF/text document loading
    chunker.py           # Recursive character text splitting
    embedder.py          # Sentence-transformers embedding
    indexer.py           # pgvector upsert + BM25 index building
  retrieval/
    dense.py             # pgvector cosine similarity search
    sparse.py            # BM25 keyword search
    fusion.py            # Reciprocal Rank Fusion (RRF)
    reranker.py          # Cross-encoder re-ranking
    hyde.py              # Hypothetical document embeddings (optional)
    decomposer.py        # Multi-part query decomposition (optional)
  generation/
    prompt.py            # Token-budgeted prompt building with citations
    stream.py            # Gemini streaming with retry logic
  graph/
    builder.py           # NetworkX knowledge graph
    extractor.py         # LLM-based entity/relationship extraction
    retriever.py         # Graph traversal retrieval
  config.py              # Dataclass configs (ingestion, retrieval, generation, graph)
  schemas.py             # Data models (Document, Chunk, RetrievalResult, etc.)
  exceptions.py          # Domain exception hierarchy
  logging.py             # Structured logging setup (structlog)
configs/                 # Hydra YAML configs (swappable retrieval strategies)
eval/                    # Retrieval evaluation framework (Hit@K, MRR, keyword recall)
ui/                      # Gradio frontend
tests/                   # pytest suite
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- Google Cloud project with Vertex AI enabled

### Quick start

1. Clone the repo and create your `.env`:
   ```bash
   cp .env.example .env
   # Edit .env and set your GCP_PROJECT and GCP_REGION
   ```

2. Start everything:
   ```bash
   docker compose up
   ```

3. Open the UI at http://localhost:7860, upload documents, and ask questions.

### Local development

```bash
# Start just the database
docker compose up postgres -d

# Install dependencies
uv sync

# Run backend
uv run uvicorn src.api.main:app --reload --port 8000

# Run frontend (separate terminal)
uv run python ui/app.py

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check .
uv run ruff format .
```

### API

- `POST /ingest` -- upload and index documents
- `GET /query?q=your+question` -- SSE stream with answer, citations, and latency
- `GET /health` -- system health check

### Evaluation

```bash
uv run python eval/evaluate.py
```

Runs a 4-way retrieval ablation (dense only, BM25 only, hybrid, hybrid + rerank) and prints a metrics table.
