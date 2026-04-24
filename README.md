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

1. Authenticate with Google Cloud and create your `.env`:
   ```bash
   gcloud auth application-default login
   cp .env.example .env
   # Edit .env: set GOOGLE_ADC_PATH to your ADC credentials file
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

### Evaluation results

Retrieval ablation on 14 documents (11 Singapore gov + 3 ML papers), 24 factual questions scored:

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | KW Recall |
| --- | --- | --- | --- | --- | --- |
| Dense only | 0.958 | 0.958 | 1.000 | 0.969 | 0.805 |
| BM25 only | 0.833 | 0.958 | 1.000 | 0.897 | 0.834 |
| Hybrid (RRF) | 0.917 | 1.000 | 1.000 | 0.951 | 0.857 |
| Hybrid + Rerank | 0.917 | 1.000 | 1.000 | 0.958 | 0.878 |

Key takeaways:
- Dense retrieval achieves the highest Hit@1 (0.958) but lowest keyword recall (0.805)
- BM25 captures more keywords but misses semantic matches at rank 1
- Hybrid search (RRF) achieves perfect Hit@3 and Hit@5, combining both signals
- Cross-encoder reranking improves MRR and keyword recall over hybrid alone

### Chunk size comparison

Hybrid + Rerank results at two chunk sizes (same corpus, same questions):

| Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | KW Recall |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 0.917 | 1.000 | 1.000 | 0.958 | 0.878 |
| 256 | 1,079 | 0.917 | 1.000 | 1.000 | 0.958 | 0.788 |

Both chunk sizes hit identical rates and MRR. The 512-size chunks score higher on keyword recall (0.878 vs 0.788) because each chunk contains more text, so expected keywords are more likely to co-occur in the top results. Smaller chunks give finer-grained retrieval but need more of them to cover the same keywords.

### Generation quality (LLM-as-judge)

End-to-end eval: retrieve (Hybrid + Rerank) then generate an answer with Gemini, then score with a separate Gemini judge on 1-5 scales. 24 answerable questions scored:

| Metric | Avg score |
| --- | --- |
| Faithfulness (grounded in context) | 5.00 |
| Relevancy (addresses the question) | 4.46 |

Perfect faithfulness means the system never fabricates claims beyond retrieved context. Relevancy dips slightly on questions where the retrieved chunks partially address the question.

### Knowledge graph

During ingestion, Gemini extracts entities and relationships from each chunk to build a knowledge graph (NetworkX). The graph augments retrieval by surfacing related context that keyword and vector search might miss.

Current corpus stats (14 documents):

| Metric | Count |
| --- | --- |
| Entities | 2,825 |
| Relationships | 3,814 |

Entity types: concept (1,224), person (582), technique (301), model (185), dataset (111), organization (104), metric (71), other (47)
