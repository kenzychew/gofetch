# Production RAG System: 24-Hour Build Plan

## Project: GoFetch — Hybrid Retrieval with Re-ranking & Streaming Citations

A production-grade RAG pipeline that always fetches the right answer — with hybrid search, cross-encoder re-ranking, streaming generation, and inline citation grounding — all deployed via Docker Compose.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  Web UI      │────▶│  FastAPI Backend (streaming SSE)                 │
│  (Gradio)    │◀────│                                                  │
└─────────────┘     │  ┌────────────┐  ┌─────────────┐  ┌───────────┐ │
                    │  │ Ingestion   │  │ Retrieval    │  │ Generation│ │
                    │  │ Pipeline    │  │ Pipeline     │  │ Pipeline  │ │
                    │  │            │  │             │  │           │ │
                    │  │ • Chunking  │  │ • BM25       │  │ • Prompt  │ │
                    │  │ • Embedding │  │ • Dense Vec  │  │ • Stream  │ │
                    │  │ • Indexing  │  │ • RRF Fusion │  │ • Cite    │ │
                    │  │            │  │ • Re-rank    │  │           │ │
                    │  └─────┬──────┘  └──────┬──────┘  └───────────┘ │
                    │        │                │                        │
                    │  ┌─────▼────────────────▼──────┐                │
                    │  │     Qdrant Vector DB         │                │
                    │  │     + BM25 Index (rank_bm25) │                │
                    │  └─────────────────────────────┘                │
                    └──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer          | Tool                        | Why                                              |
|----------------|-----------------------------|--------------------------------------------------|
| Vector DB      | Qdrant (Docker)             | Fast, easy to self-host, supports payloads        |
| Sparse Search  | rank_bm25 (Python)          | Lightweight BM25 — no extra infra needed          |
| Embeddings     | sentence-transformers       | Local `all-MiniLM-L6-v2` — no API key needed     |
| Re-ranker      | cross-encoder (SBERT)       | `cross-encoder/ms-marco-MiniLM-L-6-v2`           |
| LLM            | OpenAI API (GPT-4o-mini)    | Cheap, fast, streaming support                    |
| Backend        | FastAPI + SSE               | Async, streaming, production-ready                |
| Frontend       | Gradio                      | Fast to build, looks polished, supports streaming |
| Deployment     | Docker Compose              | One command to launch entire stack                 |
| Chunking       | LangChain text splitters    | Semantic + recursive chunking out of the box       |

### Why these choices (for your presentation)

- **Local embeddings + re-ranker**: Shows you understand model tradeoffs, not just API calls
- **BM25 + Dense hybrid**: Demonstrates knowledge of IR fundamentals (sparse vs dense)
- **RRF fusion**: Simple but theoretically grounded ranking combination
- **Cross-encoder re-ranking**: Two-stage retrieval is what production systems use
- **Streaming SSE**: Shows real engineering, not just batch request/response

---

## Hour-by-Hour Build Plan

### Phase 1: Foundation (Hours 0–4)

#### Hour 0–1: Project Scaffolding
- [ ] Create repo structure (see below)
- [ ] Set up `pyproject.toml` or `requirements.txt`
- [ ] Write `docker-compose.yml` with Qdrant service
- [ ] Verify Qdrant starts and is accessible
- [ ] Create `.env` for OpenAI API key

#### Hour 1–2: Document Ingestion Pipeline
- [ ] Build `ingestion/chunker.py` — recursive character splitter (chunk_size=512, overlap=50)
- [ ] Build `ingestion/embedder.py` — sentence-transformers encoding
- [ ] Build `ingestion/loader.py` — read PDFs and text files from a `/data` folder
- [ ] Build `ingestion/indexer.py` — upsert chunks into Qdrant with payload (text, source, chunk_id)
- [ ] Also build a BM25 index from the same chunks (pickle it to disk)
- [ ] Test: ingest 2-3 sample PDFs, verify Qdrant has records

#### Hour 2–3: Retrieval Pipeline (Hybrid Search)
- [ ] Build `retrieval/dense.py` — embed query → Qdrant search → top 20
- [ ] Build `retrieval/sparse.py` — BM25 search → top 20
- [ ] Build `retrieval/fusion.py` — Reciprocal Rank Fusion combining both lists
  ```python
  # RRF formula: score(d) = Σ 1 / (k + rank(d)) across all lists
  # k = 60 is standard
  ```
- [ ] Return top 10 fused results
- [ ] Test: query your ingested docs, verify results look reasonable

#### Hour 3–4: Cross-Encoder Re-ranking
- [ ] Build `retrieval/reranker.py`
- [ ] Load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- [ ] Take top 10 from fusion → re-rank with cross-encoder → return top 5
- [ ] Test: compare fused results vs re-ranked results — re-ranked should be more relevant

**Checkpoint: You now have a working retrieval pipeline. Take a short break.**

---

### Phase 2: Generation & API (Hours 4–8)

#### Hour 4–5: Streaming Generation with Citations
- [ ] Build `generation/prompt.py` — citation-aware prompt template:
  ```
  Answer the question based on the provided context.
  For each claim, cite the source using [1], [2], etc.
  If the context doesn't contain the answer, say so.

  Context:
  [1] (source: paper.pdf, chunk 3): "..."
  [2] (source: report.pdf, chunk 7): "..."
  ...

  Question: {query}
  ```
- [ ] Build `generation/stream.py` — OpenAI streaming call, yield chunks
- [ ] Return both the streamed answer and the source chunks used

#### Hour 5–6: FastAPI Backend
- [ ] Build `api/main.py`:
  - `POST /ingest` — upload docs, run ingestion pipeline
  - `GET /query` — SSE endpoint, streams the answer
  - `GET /health` — health check
- [ ] Wire up retrieval + generation into the query endpoint
- [ ] Test with curl:
  ```bash
  curl -N "http://localhost:8000/query?q=what+is+attention"
  ```

#### Hour 6–7: Gradio Frontend
- [ ] Build `ui/app.py`:
  - Text input for query
  - Streaming text output for answer
  - Collapsible panel showing retrieved chunks with source + relevance score
  - Upload button for adding new documents
- [ ] Connect to FastAPI backend
- [ ] Verify streaming works end-to-end

#### Hour 7–8: Docker Compose
- [ ] Write `Dockerfile` for the app (Python 3.11, install deps, copy code)
- [ ] Update `docker-compose.yml`:
  ```yaml
  services:
    qdrant:
      image: qdrant/qdrant:latest
      ports: ["6333:6333"]
      volumes: [qdrant_data:/qdrant/storage]
    app:
      build: .
      ports: ["8000:8000", "7860:7860"]
      depends_on: [qdrant]
      env_file: .env
      volumes: [./data:/app/data]
  ```
- [ ] Test: `docker compose up` → full system works

**Checkpoint: You have a fully working, deployable system. Major milestone.**

---

### Phase 3: Polish & Depth (Hours 8–14)

#### Hour 8–9: Evaluation Script
- [ ] Build `eval/evaluate.py`:
  - 10 hand-written question/expected-answer pairs for your test docs
  - Run each through the pipeline
  - Score: retrieval hit rate (is correct doc in top 5?) + answer quality (LLM-as-judge)
  - Output a results table
- [ ] This is HUGE for your presentation — shows engineering rigor

#### Hour 9–10: Chunking Strategy Comparison
- [ ] Add a second chunking method: semantic chunking (split on sentence boundaries + embedding similarity)
- [ ] Run your eval with both strategies, compare results
- [ ] Save the comparison for your presentation (even a simple table is compelling)

#### Hour 10–11: Observability & Logging
- [ ] Add structured logging with timing:
  - Embedding latency
  - BM25 search latency
  - Dense search latency
  - Re-ranking latency
  - Total retrieval latency
  - Generation time-to-first-token
- [ ] Display latency breakdown in the UI (a small stats panel)
- [ ] This shows production mindset

#### Hour 11–12: README & Documentation
- [ ] Architecture diagram (can use Mermaid in the README)
- [ ] Setup instructions (one-command Docker Compose)
- [ ] Design decisions and tradeoffs section
- [ ] Example queries and expected behavior

#### Hour 12–14: Buffer / Sleep
- [ ] Fix any bugs from testing
- [ ] Handle edge cases (empty results, long documents, API errors)
- [ ] **Get some sleep** — you present better rested

---

### Phase 4: Presentation Prep (Hours 14–18)

#### Hour 14–16: Build Your Demo Script
- [ ] Prepare 2-3 demo queries that showcase different strengths:
  1. A factual question (shows retrieval accuracy + citations)
  2. A comparison question across docs (shows hybrid search value)
  3. A question the docs DON'T answer (shows the system says "I don't know")
- [ ] Pre-ingest a compelling doc set (research papers, technical docs, or something topical)
- [ ] Rehearse the demo flow twice

#### Hour 16–18: Slides (5-7 slides max)
1. **Problem**: "Most RAG demos are toy-level. Production RAG is different. GoFetch shows what a real retrieval pipeline looks like."
2. **Architecture**: The diagram from above
3. **Key Technical Decisions**: Hybrid search, why re-ranking matters, chunking strategies
4. **Live Demo**: Switch to terminal/browser
5. **Eval Results**: Show your retrieval quality numbers
6. **Latency Breakdown**: Show the timing stats
7. **What I'd Add Next**: Hypothetical embeddings, query decomposition, guardrails

---

## Repo Structure

```
gofetch/
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── README.md
├── data/                    # Drop PDFs/text files here
│   └── sample.pdf
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py        # Read files from /data
│   │   ├── chunker.py       # Recursive + semantic chunking
│   │   ├── embedder.py      # sentence-transformers
│   │   └── indexer.py       # Qdrant upsert + BM25 index
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py         # Qdrant vector search
│   │   ├── sparse.py        # BM25 search
│   │   ├── fusion.py        # Reciprocal Rank Fusion
│   │   └── reranker.py      # Cross-encoder re-ranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt.py        # Citation-aware prompt template
│   │   └── stream.py        # OpenAI streaming wrapper
│   └── api/
│       ├── __init__.py
│       └── main.py          # FastAPI app
├── ui/
│   └── app.py               # Gradio frontend
└── eval/
    ├── evaluate.py           # Retrieval + generation eval
    └── questions.json        # Test question/answer pairs
```

---

## Key Talking Points for Your Presentation

### "Why hybrid search?"
Dense retrieval misses keyword-specific matches (e.g., exact error codes, product names). BM25 misses semantic similarity. Combining them with RRF gives you the best of both — this is what Google and Bing actually do.

### "Why re-ranking?"
First-stage retrieval (BM25/dense) uses bi-encoders — fast but approximate. Cross-encoders jointly encode the query and document together, giving much more accurate relevance scores. The tradeoff is speed, which is why you do it as a second stage on a small candidate set.

### "Why streaming?"
Time-to-first-token matters for UX. Batch responses make users wait 5-10 seconds staring at a spinner. Streaming shows text appearing in ~300ms. This is a production concern, not an academic one.

### "Why citations?"
Without citations, RAG is a black box. Grounding each claim in a specific chunk lets users verify the answer and builds trust. It also lets you debug retrieval quality.

---

## Emergency Simplifications (If Running Behind)

If you're behind schedule, cut in this order (least impactful first):

1. **Drop semantic chunking comparison** → just use recursive chunking
2. **Drop the eval script** → show retrieval quality anecdotally in the demo
3. **Drop Gradio UI** → demo via curl + terminal (still impressive if Docker works)
4. **Drop Docker** → run locally (but this sacrifices your biggest differentiator)

Never cut: hybrid search, re-ranking, streaming, citations. These ARE the project.

---

## Sample Requirements

```
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
sse-starlette==2.1.0
qdrant-client==1.11.0
sentence-transformers==3.1.0
rank-bm25==0.2.2
openai==1.47.0
gradio==4.44.0
pypdf==4.3.0
python-dotenv==1.0.1
python-multipart==0.0.10
```

Good luck — you've got this. Ship it. 🐕
