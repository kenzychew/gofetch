"""FastAPI dependency injection factories for pipeline components."""

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import structlog
from google import genai
from pgvector.asyncpg import register_vector

from src.config import AppConfig, PromptConfig
from src.generation.prompt import PromptBuilder
from src.graph.builder import KnowledgeGraph
from src.graph.retriever import GraphRetriever
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import BM25Indexer, VectorIndexer
from src.logging import get_logger
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever

logger = get_logger(__name__)


class DependencyContainer:
    """Singleton container for all pipeline components.

    Holds references to initialized components and provides typed
    accessors. Eliminates the need for module-level global variables.

    Attributes:
        config: Application configuration.
        pool: asyncpg connection pool for PostgreSQL.
        embedder: Sentence-transformer embedder.
        dense_retriever: pgvector dense vector retriever.
        sparse_retriever: BM25 sparse retriever (None until ingestion).
        graph_retriever: Graph retriever (None until ingestion with graph enabled).
        reranker: Cross-encoder reranker.
        llm_client: Google GenAI client for Gemini.
        prompt_builder: Citation-aware prompt builder.
        vector_indexer: pgvector vector indexer.
    """

    def __init__(self) -> None:
        """Initialize the container with all fields set to None."""
        self.config: AppConfig | None = None
        self.pool: asyncpg.Pool | None = None
        self.embedder: Embedder | None = None
        self.dense_retriever: DenseRetriever | None = None
        self.sparse_retriever: SparseRetriever | None = None
        self.graph_retriever: GraphRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.llm_client: genai.Client | None = None
        self.prompt_builder: PromptBuilder | None = None
        self.vector_indexer: VectorIndexer | None = None


_container = DependencyContainer()


def _load_env(env_path: str = ".env") -> None:
    """Load environment variables from a .env file if it exists.

    Reads key=value pairs from the file and sets them in os.environ.
    Skips blank lines, comments, and already-set variables.

    Args:
        env_path: Path to the .env file. Defaults to ".env" in the working directory.
    """
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


async def _init_pg_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector type codec on each new connection.

    Called automatically by asyncpg when a new connection is created
    in the pool, enabling native vector type encoding/decoding.

    Args:
        conn: The new asyncpg connection.
    """
    await register_vector(conn)


async def init_dependencies(config: AppConfig, prompt_config: PromptConfig) -> None:
    """Initialize all pipeline components as singletons.

    Called once at application startup. Components are stored in the
    module-level container and served via FastAPI Depends().

    Args:
        config: Application configuration.
        prompt_config: Prompt template configuration (loaded from Hydra YAML).
    """
    _load_env()

    # Env var takes precedence over config file for deployment flexibility
    database_url = os.environ.get("DATABASE_URL", config.database_url)

    # Create the pgvector extension before opening the pool, because
    # register_vector (the pool init callback) needs the type to exist
    bootstrap_conn = await asyncpg.connect(dsn=database_url)
    try:
        await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await bootstrap_conn.close()

    _container.pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=2,
        max_size=10,
        init=_init_pg_connection,
    )

    _container.config = config
    _container.embedder = Embedder(config.ingestion)
    _container.dense_retriever = DenseRetriever(_container.pool, config)
    _container.reranker = CrossEncoderReranker(config.retrieval)
    gcp_project = os.environ.get("GCP_PROJECT", config.gcp_project)
    gcp_region = os.environ.get("GCP_REGION", config.gcp_region)
    _container.llm_client = genai.Client(
        vertexai=True,
        project=gcp_project,
        location=gcp_region,
    )
    _container.vector_indexer = VectorIndexer(_container.pool, config)

    config.prompts = prompt_config
    _container.prompt_builder = PromptBuilder(prompt_config, config.generation)

    # Load BM25 index if it exists
    bm25_path = Path(config.bm25_index_path)
    if bm25_path.exists():
        bm25_indexer = BM25Indexer(config.bm25_index_path)
        bm25, chunks = bm25_indexer.load_index()
        _container.sparse_retriever = SparseRetriever(bm25, chunks)
        logger.info("Loaded existing BM25 index")

        # Load graph if it exists and graph retrieval is enabled
        graph_path = Path(config.graph_data_path)
        if graph_path.exists() and config.retrieval.use_graph:
            graph = KnowledgeGraph(config.graph)
            graph.load(config.graph_data_path)
            _container.graph_retriever = GraphRetriever(graph, config.graph, chunks)
            logger.info(
                "Loaded knowledge graph",
                nodes=graph.num_nodes,
                edges=graph.num_edges,
            )
        elif config.retrieval.use_graph:
            logger.warning("Graph retrieval enabled but no graph data found")
    else:
        logger.warning("No BM25 index found, sparse retrieval unavailable until ingestion")

    logger.info("All dependencies initialized")


def get_pool() -> asyncpg.Pool:
    """Get the asyncpg connection pool.

    Returns:
        The asyncpg connection pool.
    """
    if _container.pool is None:
        msg = "Connection pool not initialized"
        raise RuntimeError(msg)
    return _container.pool


async def close_pool() -> None:
    """Close the asyncpg connection pool."""
    if _container.pool is not None:
        try:
            await _container.pool.close()
        finally:
            _container.pool = None


def get_config() -> AppConfig:
    """Get the application configuration.

    Returns:
        The application configuration instance.
    """
    if _container.config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return _container.config


def get_embedder() -> Embedder:
    """Get the embedder instance.

    Returns:
        The embedder singleton.
    """
    if _container.embedder is None:
        msg = "Embedder not initialized"
        raise RuntimeError(msg)
    return _container.embedder


def get_dense_retriever() -> DenseRetriever:
    """Get the dense retriever instance.

    Returns:
        The dense retriever singleton.
    """
    if _container.dense_retriever is None:
        msg = "Dense retriever not initialized"
        raise RuntimeError(msg)
    return _container.dense_retriever


def get_sparse_retriever() -> SparseRetriever | None:
    """Get the sparse retriever instance, or None if not yet built.

    Returns:
        The sparse retriever singleton or None.
    """
    return _container.sparse_retriever


def set_sparse_retriever(retriever: SparseRetriever) -> None:
    """Set the sparse retriever after BM25 index is built.

    Args:
        retriever: The newly built sparse retriever.
    """
    _container.sparse_retriever = retriever


def get_graph_retriever() -> GraphRetriever | None:
    """Get the graph retriever instance, or None if not available.

    Returns:
        The graph retriever singleton or None.
    """
    return _container.graph_retriever


def set_graph_retriever(retriever: GraphRetriever) -> None:
    """Set the graph retriever after graph is built during ingestion.

    Args:
        retriever: The newly built graph retriever.
    """
    _container.graph_retriever = retriever


def get_reranker() -> CrossEncoderReranker:
    """Get the reranker instance.

    Returns:
        The cross-encoder reranker singleton.
    """
    if _container.reranker is None:
        msg = "Reranker not initialized"
        raise RuntimeError(msg)
    return _container.reranker


def get_llm_client() -> genai.Client:
    """Get the Google GenAI client instance.

    Returns:
        The GenAI client singleton.
    """
    if _container.llm_client is None:
        msg = "LLM client not initialized"
        raise RuntimeError(msg)
    return _container.llm_client


def get_prompt_builder() -> PromptBuilder:
    """Get the prompt builder instance.

    Returns:
        The prompt builder singleton.
    """
    if _container.prompt_builder is None:
        msg = "Prompt builder not initialized"
        raise RuntimeError(msg)
    return _container.prompt_builder


def get_vector_indexer() -> VectorIndexer:
    """Get the vector indexer instance.

    Returns:
        The vector indexer singleton.
    """
    if _container.vector_indexer is None:
        msg = "Vector indexer not initialized"
        raise RuntimeError(msg)
    return _container.vector_indexer


async def get_request_id() -> AsyncIterator[str]:
    """Generate a unique request ID for tracing.

    Binds the request_id to structlog context vars so all log
    entries within this request include the ID.

    Yields:
        A unique request ID string.
    """
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        yield request_id
    finally:
        structlog.contextvars.unbind_contextvars("request_id")
