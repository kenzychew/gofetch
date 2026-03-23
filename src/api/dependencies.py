"""FastAPI dependency injection factories for pipeline components."""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import structlog
from anthropic import AsyncAnthropic

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

# Module-level singletons initialized at startup
_config: AppConfig | None = None
_embedder: Embedder | None = None
_dense_retriever: DenseRetriever | None = None
_sparse_retriever: SparseRetriever | None = None
_graph_retriever: GraphRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_anthropic_client: AsyncAnthropic | None = None
_prompt_builder: PromptBuilder | None = None
_vector_indexer: VectorIndexer | None = None


def init_dependencies(config: AppConfig, prompt_config: PromptConfig) -> None:
    """Initialize all pipeline components as singletons.

    Called once at application startup. Components are loaded into
    module-level variables and served via FastAPI Depends().

    Args:
        config: Application configuration.
        prompt_config: Prompt template configuration (loaded from Hydra YAML).
    """
    global _config, _embedder, _dense_retriever, _sparse_retriever  # noqa: PLW0603
    global _graph_retriever  # noqa: PLW0603
    global _reranker, _anthropic_client, _prompt_builder, _vector_indexer  # noqa: PLW0603

    _config = config
    _embedder = Embedder(config.ingestion)
    _dense_retriever = DenseRetriever(config)
    _reranker = CrossEncoderReranker(config.retrieval)
    _anthropic_client = AsyncAnthropic()
    _vector_indexer = VectorIndexer(config)

    config.prompts = prompt_config
    _prompt_builder = PromptBuilder(prompt_config, config.generation)

    # Load BM25 index if it exists
    bm25_path = Path(config.bm25_index_path)
    if bm25_path.exists():
        bm25_indexer = BM25Indexer(config.bm25_index_path)
        bm25, chunks = bm25_indexer.load_index()
        _sparse_retriever = SparseRetriever(bm25, chunks)
        logger.info("Loaded existing BM25 index")

        # Load graph if it exists and graph retrieval is enabled
        graph_path = Path(config.graph_data_path)
        if graph_path.exists() and config.retrieval.use_graph:
            graph = KnowledgeGraph(config.graph)
            graph.load(config.graph_data_path)
            _graph_retriever = GraphRetriever(graph, config.graph, chunks)
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


def get_config() -> AppConfig:
    """Get the application configuration.

    Returns:
        The application configuration instance.
    """
    if _config is None:
        msg = "Config not initialized"
        raise RuntimeError(msg)
    return _config


def get_embedder() -> Embedder:
    """Get the embedder instance.

    Returns:
        The embedder singleton.
    """
    if _embedder is None:
        msg = "Embedder not initialized"
        raise RuntimeError(msg)
    return _embedder


def get_dense_retriever() -> DenseRetriever:
    """Get the dense retriever instance.

    Returns:
        The dense retriever singleton.
    """
    if _dense_retriever is None:
        msg = "Dense retriever not initialized"
        raise RuntimeError(msg)
    return _dense_retriever


def get_sparse_retriever() -> SparseRetriever | None:
    """Get the sparse retriever instance, or None if not yet built.

    Returns:
        The sparse retriever singleton or None.
    """
    return _sparse_retriever


def set_sparse_retriever(retriever: SparseRetriever) -> None:
    """Set the sparse retriever after BM25 index is built.

    Args:
        retriever: The newly built sparse retriever.
    """
    global _sparse_retriever  # noqa: PLW0603
    _sparse_retriever = retriever


def get_graph_retriever() -> GraphRetriever | None:
    """Get the graph retriever instance, or None if not available.

    Returns:
        The graph retriever singleton or None.
    """
    return _graph_retriever


def set_graph_retriever(retriever: GraphRetriever) -> None:
    """Set the graph retriever after graph is built during ingestion.

    Args:
        retriever: The newly built graph retriever.
    """
    global _graph_retriever  # noqa: PLW0603
    _graph_retriever = retriever


def get_reranker() -> CrossEncoderReranker:
    """Get the reranker instance.

    Returns:
        The cross-encoder reranker singleton.
    """
    if _reranker is None:
        msg = "Reranker not initialized"
        raise RuntimeError(msg)
    return _reranker


def get_anthropic_client() -> AsyncAnthropic:
    """Get the async Anthropic client instance.

    Returns:
        The Anthropic client singleton.
    """
    if _anthropic_client is None:
        msg = "Anthropic client not initialized"
        raise RuntimeError(msg)
    return _anthropic_client


def get_prompt_builder() -> PromptBuilder:
    """Get the prompt builder instance.

    Returns:
        The prompt builder singleton.
    """
    if _prompt_builder is None:
        msg = "Prompt builder not initialized"
        raise RuntimeError(msg)
    return _prompt_builder


def get_vector_indexer() -> VectorIndexer:
    """Get the vector indexer instance.

    Returns:
        The vector indexer singleton.
    """
    if _vector_indexer is None:
        msg = "Vector indexer not initialized"
        raise RuntimeError(msg)
    return _vector_indexer


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
