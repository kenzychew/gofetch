"""FastAPI application with SSE streaming, ingestion, and health endpoints."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from google import genai
from omegaconf import OmegaConf
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import (
    close_pool,
    get_config,
    get_dense_retriever,
    get_embedder,
    get_graph_retriever,
    get_llm_client,
    get_pool,
    get_prompt_builder,
    get_request_id,
    get_reranker,
    get_sparse_retriever,
    get_vector_indexer,
    init_dependencies,
    set_graph_retriever,
    set_sparse_retriever,
)
from src.config import (
    AppConfig,
    GenerationConfig,
    GraphConfig,
    IngestionConfig,
    PromptConfig,
    RetrievalConfig,
)
from src.exceptions import GoFetchError
from src.generation.stream import stream_completion
from src.graph.builder import KnowledgeGraph
from src.graph.extractor import extract_entities_and_relationships
from src.graph.retriever import GraphRetriever
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25Indexer
from src.ingestion.loader import load_documents
from src.logging import get_logger, setup_logging
from src.retrieval.decomposer import decompose_query
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hyde import generate_hypothetical_document
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.schemas import Chunk, CitedSource, RetrievalResult

logger = get_logger(__name__)


def _load_yaml_config(path: Path) -> dict[str, object]:
    """Load and validate a YAML config file via OmegaConf.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a plain dict, or empty dict if file is missing/malformed.
    """
    if not path.exists():
        logger.warning("Config file not found, using defaults", path=str(path))
        return {}

    raw = OmegaConf.load(path)
    cfg = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(cfg, dict):
        logger.warning("Malformed config file, using defaults", path=str(path))
        return {}
    return cfg


def _extract_sub_config(cfg: dict[str, object], key: str) -> dict[str, object]:
    """Extract a sub-config dict, returning empty dict if missing or wrong type.

    Args:
        cfg: Parent config dictionary.
        key: Key to extract.

    Returns:
        Sub-config dict or empty dict.
    """
    value = cfg.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _resolve_defaults(cfg: dict[str, object]) -> dict[str, object]:
    """Resolve Hydra-style defaults list by loading referenced sub-config files.

    OmegaConf.load() does not resolve Hydra defaults -- it treats the
    defaults list as plain YAML. This function manually loads each
    referenced sub-config and merges it into the parent config.

    Args:
        cfg: The raw config dict (may contain a 'defaults' key).

    Returns:
        Merged config with sub-configs loaded from their YAML files.
    """
    defaults = cfg.pop("defaults", None)
    if not isinstance(defaults, list):
        return cfg

    configs_dir = Path("configs")
    for entry in defaults:
        if isinstance(entry, str) and entry == "_self_":
            continue
        if isinstance(entry, dict):
            for group, variant in entry.items():
                sub_path = configs_dir / group / f"{variant}.yaml"
                sub_cfg = _load_yaml_config(sub_path)
                if sub_cfg:
                    cfg[group] = sub_cfg

    return cfg


def _load_config() -> tuple[AppConfig, PromptConfig]:
    """Load application config from Hydra YAML files.

    Resolves Hydra defaults to load sub-config files for each
    pipeline component. Falls back to dataclass defaults if
    config files are not found.

    Returns:
        Tuple of (AppConfig, PromptConfig).
    """
    cfg = _load_yaml_config(Path("configs/config.yaml"))
    cfg = _resolve_defaults(cfg)

    # Only pass YAML values that are present; AppConfig dataclass
    # defaults are the single source of truth for fallback values
    top_level_keys = [
        "database_url",
        "table_name",
        "gcp_project",
        "gcp_region",
        "bm25_index_path",
        "graph_data_path",
        "data_dir",
        "log_level",
    ]
    top_level_overrides = {k: str(cfg[k]) for k in top_level_keys if k in cfg}

    app_config = AppConfig(
        ingestion=IngestionConfig(**_extract_sub_config(cfg, "ingestion")),
        retrieval=RetrievalConfig(**_extract_sub_config(cfg, "retrieval")),
        generation=GenerationConfig(**_extract_sub_config(cfg, "generation")),
        graph=GraphConfig(**_extract_sub_config(cfg, "graph")),
        **top_level_overrides,
    )

    # Load prompt config from the resolved sub-config, falling back to file
    prompts_dict = _extract_sub_config(cfg, "prompts")
    if not prompts_dict:
        prompts_dict = _load_yaml_config(Path("configs/prompts/citation.yaml"))

    if not prompts_dict:
        return app_config, PromptConfig()

    few_shot = prompts_dict.get("few_shot_examples", [])
    prompt_config = PromptConfig(
        system_prompt=str(prompts_dict.get("system_prompt", "")),
        context_template=str(prompts_dict.get("context_template", "")),
        chunk_template=str(prompts_dict.get("chunk_template", "")),
        few_shot_examples=few_shot if isinstance(few_shot, list) else [],
        low_confidence_warning=str(prompts_dict.get("low_confidence_warning", "")),
    )

    return app_config, prompt_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup/shutdown.

    Initializes all pipeline components at startup and
    closes connections at shutdown.
    """
    config, prompt_config = _load_config()
    setup_logging(level=config.log_level)
    logger.info("Starting GoFetch RAG system")
    await init_dependencies(config, prompt_config)

    try:
        # Ensure pgvector table and indexes exist
        indexer = get_vector_indexer()
        await indexer.ensure_table()

        yield
    finally:
        # Always close pool, even if ensure_table fails
        await close_pool()
        logger.info("GoFetch shutdown complete")


app = FastAPI(
    title="GoFetch RAG API",
    description="Production-grade RAG with hybrid search, re-ranking, and streaming",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Pydantic request/response models ---


class IngestResponse(BaseModel):
    """Response model for the ingestion endpoint.

    Attributes:
        documents: Number of documents ingested.
        chunks: Number of chunks created.
        message: Status message.
    """

    documents: int
    chunks: int
    message: str


class HealthResponse(BaseModel):
    """Response model for the health check endpoint.

    Attributes:
        status: Overall system status.
        postgres: Whether PostgreSQL is reachable.
        bm25_loaded: Whether the BM25 index is loaded.
    """

    status: str
    postgres: bool
    bm25_loaded: bool


# --- Endpoints ---


async def _save_uploaded_files(files: list[UploadFile], data_dir: Path) -> None:
    """Save uploaded files to the data directory.

    Args:
        files: List of uploaded files.
        data_dir: Target directory for saving files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file.filename:
            file_path = data_dir / file.filename
            content = await file.read()
            file_path.write_bytes(content)
            logger.info("Saved uploaded file", filename=file.filename)


async def _build_knowledge_graph(
    chunks: list[Chunk],
    config: AppConfig,
) -> None:
    """Build and persist the knowledge graph from ingested chunks.

    Args:
        chunks: All ingested chunks with embeddings.
        config: Application configuration.
    """
    llm_client = get_llm_client()
    graph = KnowledgeGraph(config.graph)
    batch_size = config.graph.extraction_batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            entities, relationships = await extract_entities_and_relationships(
                batch, llm_client, config.graph
            )
            graph.add_entities(entities)
            graph.add_relationships(relationships)
        except Exception as exc:
            logger.warning("Graph extraction failed for batch, skipping", batch=i, error=str(exc))

    graph.save(config.graph_data_path)
    set_graph_retriever(GraphRetriever(graph, config.graph, chunks))
    logger.info(
        "Built knowledge graph",
        nodes=graph.num_nodes,
        edges=graph.num_edges,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] | None = None,
    request_id: str = Depends(get_request_id),
) -> IngestResponse:
    """Upload and index documents into the RAG pipeline.

    Accepts file uploads or ingests all documents from the data/ directory.
    Runs chunking, embedding, pgvector upsert, and BM25 index building.

    Args:
        files: Optional list of uploaded files.
        request_id: Auto-generated request ID for tracing.

    Returns:
        IngestResponse with counts and status.

    Raises:
        HTTPException: If ingestion fails.
    """
    config = get_config()
    embedder = get_embedder()
    indexer = get_vector_indexer()

    try:
        data_dir = Path(config.data_dir)
        if files:
            await _save_uploaded_files(files, data_dir)

        documents = load_documents(data_dir)
        if not documents:
            raise HTTPException(status_code=400, detail="No documents found to ingest")

        # Chunk all documents
        chunker = RecursiveChunker(config.ingestion)
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk(doc))

        # Embed chunks (sync, but fast for small-medium corpora)
        all_chunks = await asyncio.to_thread(embedder.embed_chunks, all_chunks)

        # Upsert to PostgreSQL
        await indexer.upsert_chunks(all_chunks)

        # Build BM25 index and update sparse retriever
        bm25_indexer = BM25Indexer(config.bm25_index_path)
        bm25 = await asyncio.to_thread(bm25_indexer.build_index, all_chunks)
        set_sparse_retriever(SparseRetriever(bm25, all_chunks))

        if config.retrieval.use_graph:
            await _build_knowledge_graph(all_chunks, config)

        logger.info(
            "Ingestion complete",
            documents=len(documents),
            chunks=len(all_chunks),
        )

        return IngestResponse(
            documents=len(documents),
            chunks=len(all_chunks),
            message=f"Ingested {len(documents)} documents into {len(all_chunks)} chunks",
        )

    except HTTPException:
        raise
    except GoFetchError as exc:
        logger.error("Ingestion failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected ingestion error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal ingestion error") from exc


async def _run_retrieval(
    q: str,
    config: AppConfig,
    embedder: Embedder,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever | None,
    graph_retriever: GraphRetriever | None,
    reranker: CrossEncoderReranker,
    llm_client: genai.Client,
    latency: dict[str, float],
) -> list[RetrievalResult]:
    """Run the full retrieval pipeline: embed, retrieve, fuse, rerank.

    Handles optional HyDE, query decomposition, and graph retrieval
    based on configuration flags.

    Args:
        q: The user's search query.
        config: Application configuration.
        embedder: Embedder instance for query embedding.
        dense_retriever: Dense vector retriever.
        sparse_retriever: BM25 retriever (may be None).
        graph_retriever: Graph retriever (may be None).
        reranker: Cross-encoder reranker.
        llm_client: GenAI client (for HyDE/decomposition).
        latency: Mutable dict to record timing per stage.

    Returns:
        Re-ranked retrieval results.
    """
    # Query decomposition (optional)
    queries = [q]
    if config.retrieval.use_decomposition:
        t0 = time.perf_counter()
        queries = await decompose_query(q, llm_client, config.generation)
        latency["decompose_ms"] = (time.perf_counter() - t0) * 1000

    # Embed query (optionally via HyDE)
    t0 = time.perf_counter()
    if config.retrieval.use_hyde:
        hyde_doc = await generate_hypothetical_document(q, llm_client, config.generation)
        query_embedding = await asyncio.to_thread(embedder.embed_query, hyde_doc)
        latency["hyde_ms"] = (time.perf_counter() - t0) * 1000
    else:
        query_embedding = await asyncio.to_thread(embedder.embed_query, q)
        latency["embed_ms"] = (time.perf_counter() - t0) * 1000

    # Parallel retrieval across all sub-queries
    t0 = time.perf_counter()
    all_ranked_lists: list[list[RetrievalResult]] = []

    for sub_query in queries:
        dense_retriever.set_query_embedding(query_embedding)

        tasks: list[asyncio.Task[list[RetrievalResult]]] = []
        tasks.append(
            asyncio.create_task(dense_retriever.retrieve(sub_query, config.retrieval.dense_top_k))
        )
        if sparse_retriever and config.retrieval.sparse_top_k > 0:
            tasks.append(
                asyncio.create_task(
                    sparse_retriever.retrieve(sub_query, config.retrieval.sparse_top_k)
                )
            )
        if graph_retriever and config.retrieval.use_graph:
            tasks.append(
                asyncio.create_task(
                    graph_retriever.retrieve(sub_query, config.retrieval.fusion_top_k)
                )
            )

        ranked_lists = await asyncio.gather(*tasks)
        all_ranked_lists.extend(ranked_lists)

    latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # RRF Fusion
    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion(
        all_ranked_lists,
        k=config.retrieval.rrf_k,
        top_k=config.retrieval.fusion_top_k,
    )
    latency["fusion_ms"] = (time.perf_counter() - t0) * 1000

    # Re-rank
    t0 = time.perf_counter()
    reranked = await reranker.rerank(q, fused, config.retrieval.rerank_top_k)
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

    return reranked


@app.get("/query")
async def query_rag(
    q: str = Query(..., description="The search query"),
    request_id: str = Depends(get_request_id),
) -> EventSourceResponse:
    """Query the RAG pipeline with SSE streaming response.

    Runs the full pipeline: embed query -> parallel retrieval (dense +
    sparse + graph) -> RRF fusion -> cross-encoder re-rank -> prompt
    build -> stream LLM response.

    Returns an SSE stream with event types: "token", "metadata", "error".

    Args:
        q: The user's search query.
        request_id: Auto-generated request ID for tracing.

    Returns:
        EventSourceResponse with streaming tokens and metadata.
    """
    config = get_config()
    embedder = get_embedder()
    dense_retriever = get_dense_retriever()
    sparse_retriever = get_sparse_retriever()
    graph_retriever = get_graph_retriever()
    reranker = get_reranker()
    llm_client = get_llm_client()
    prompt_builder = get_prompt_builder()

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        latency: dict[str, float] = {}

        try:
            reranked = await _run_retrieval(
                q,
                config,
                embedder,
                dense_retriever,
                sparse_retriever,
                graph_retriever,
                reranker,
                llm_client,
                latency,
            )

            # Check confidence
            confidence = reranked[0].score if reranked else 0.0
            low_confidence = confidence < config.retrieval.confidence_threshold
            if low_confidence:
                logger.warning(
                    "Low confidence retrieval",
                    confidence=confidence,
                    threshold=config.retrieval.confidence_threshold,
                )

            # Handle empty results
            if not reranked:
                yield {
                    "event": "token",
                    "data": "I don't have enough context to answer this question.",
                }
                yield {
                    "event": "metadata",
                    "data": json.dumps({"citations": [], "latency_ms": latency, "confidence": 0.0}),
                }
                return

            # Build prompt
            t0 = time.perf_counter()
            messages = prompt_builder.build_messages(q, reranked, low_confidence)
            latency["prompt_ms"] = (time.perf_counter() - t0) * 1000

            # Stream LLM response
            t0 = time.perf_counter()
            async for token in stream_completion(messages, llm_client, config.generation):
                yield {"event": "token", "data": token}
            latency["generation_ms"] = (time.perf_counter() - t0) * 1000

            # Send metadata as final event
            citations = [
                CitedSource(
                    index=i + 1,
                    source=r.chunk.source,
                    text=r.chunk.text[:200],
                    score=r.score,
                )
                for i, r in enumerate(reranked)
            ]

            metadata = {
                "citations": [
                    {
                        "index": c.index,
                        "source": c.source,
                        "text": c.text,
                        "score": round(c.score, 4),
                    }
                    for c in citations
                ],
                "latency_ms": {k: round(v, 1) for k, v in latency.items()},
                "confidence": round(confidence, 4),
                "low_confidence": low_confidence,
            }
            yield {"event": "metadata", "data": json.dumps(metadata)}

        except GoFetchError as exc:
            logger.error("Query pipeline failed", error=str(exc))
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        except Exception as exc:
            logger.error("Unexpected query error", error=str(exc), exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": "Internal error"})}

    return EventSourceResponse(event_generator())


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of the RAG system.

    Verifies PostgreSQL connectivity and BM25 index availability.

    Returns:
        HealthResponse with component status.
    """
    sparse = get_sparse_retriever()

    pg_ok = False
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        pg_ok = True
    except (OSError, TimeoutError) as exc:
        logger.warning("PostgreSQL health check failed", error=str(exc))
    except RuntimeError as exc:
        logger.warning("PostgreSQL pool not initialized", error=str(exc))

    return HealthResponse(
        status="healthy" if pg_ok else "degraded",
        postgres=pg_ok,
        bm25_loaded=sparse is not None,
    )
