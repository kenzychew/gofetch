"""Pydantic structured configs validated by Hydra at startup."""

import re
from dataclasses import dataclass, field

from src.exceptions import ConfigError

_VALID_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        embedding_model: Name of the sentence-transformers model.
        embedding_dim: Dimensionality of the embedding vectors.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline.

    Attributes:
        dense_top_k: Number of results from dense vector search.
        sparse_top_k: Number of results from BM25 search.
        fusion_top_k: Number of results after RRF fusion.
        rerank_top_k: Number of results after cross-encoder re-ranking.
        rrf_k: RRF smoothing parameter (default 60).
        reranker_model: Name of the cross-encoder model.
        use_graph: Whether to include graph retrieval in fusion.
        use_hyde: Whether to use HyDE for query expansion.
        use_decomposition: Whether to decompose multi-part queries.
        confidence_threshold: Minimum reranker score for confident answers.
    """

    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_graph: bool = True
    use_hyde: bool = False
    use_decomposition: bool = False
    confidence_threshold: float = 0.3


@dataclass
class GenerationConfig:
    """Configuration for the generation pipeline.

    Attributes:
        model: Anthropic model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the generated response.
        max_context_tokens: Maximum tokens for context chunks.
        stream: Whether to stream the response via SSE.
    """

    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.1
    max_tokens: int = 1024
    max_context_tokens: int = 3000
    stream: bool = True


@dataclass
class GraphConfig:
    """Configuration for the knowledge graph.

    Attributes:
        extraction_model: Anthropic model for entity extraction.
        extraction_batch_size: Number of chunks per extraction call.
        max_entities_per_chunk: Max entities to extract per chunk.
        max_relationships_per_chunk: Max relationships per chunk.
        entity_similarity_threshold: Threshold for entity name matching.
        traversal_hops: Number of hops for graph traversal.
        graph_top_k: Number of chunks from graph retrieval.
    """

    extraction_model: str = "claude-haiku-4-5-20251001"
    extraction_batch_size: int = 3
    max_entities_per_chunk: int = 10
    max_relationships_per_chunk: int = 15
    entity_similarity_threshold: float = 0.85
    traversal_hops: int = 1
    graph_top_k: int = 20


@dataclass
class PromptConfig:
    """Configuration for prompt templates.

    Attributes:
        system_prompt: System prompt template for the LLM.
        context_template: Template for formatting context chunks.
        chunk_template: Template for formatting individual chunks.
        few_shot_examples: List of few-shot example dicts.
        low_confidence_warning: Warning text for low confidence answers.
    """

    system_prompt: str = ""
    context_template: str = ""
    chunk_template: str = ""
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    low_confidence_warning: str = ""


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        ingestion: Ingestion pipeline config.
        retrieval: Retrieval pipeline config.
        generation: Generation pipeline config.
        graph: Knowledge graph config.
        prompts: Prompt template config.
        database_url: PostgreSQL connection string.
        table_name: Name of the chunks table in PostgreSQL.
        bm25_index_path: File path for pickled BM25 index.
        graph_data_path: File path for serialized graph data.
        data_dir: Directory containing source documents.
        log_level: Logging level.
    """

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    database_url: str = "postgresql://gofetch:gofetch@localhost:5432/gofetch"
    table_name: str = "chunks"
    bm25_index_path: str = "bm25_index/bm25.pkl"
    graph_data_path: str = "graph_data/graph.json"
    data_dir: str = "data"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate config values that are used in SQL identifiers.

        Raises:
            ConfigError: If table_name is not a valid SQL identifier.
        """
        if not _VALID_SQL_IDENTIFIER.match(self.table_name):
            raise ConfigError(
                f"Invalid table_name: '{self.table_name}'. Must match [a-z_][a-z0-9_]{{{{0,62}}}}."
            )
