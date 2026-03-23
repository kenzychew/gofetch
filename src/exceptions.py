"""Custom exception hierarchy for GoFetch RAG system."""


class GoFetchError(Exception):
    """Base exception for all GoFetch errors."""


class IngestionError(GoFetchError):
    """Raised when document ingestion fails."""


class ChunkingError(IngestionError):
    """Raised when text chunking fails."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


class IndexingError(IngestionError):
    """Raised when vector/BM25 indexing fails."""


class RetrievalError(GoFetchError):
    """Raised when retrieval pipeline fails."""


class VectorSearchError(RetrievalError):
    """Raised when Qdrant vector search fails."""


class SparseSearchError(RetrievalError):
    """Raised when BM25 search fails."""


class RerankError(RetrievalError):
    """Raised when cross-encoder re-ranking fails."""


class GenerationError(GoFetchError):
    """Raised when LLM generation fails."""


class PromptError(GenerationError):
    """Raised when prompt construction fails."""


class StreamError(GenerationError):
    """Raised when SSE streaming fails."""


class GraphError(GoFetchError):
    """Raised when knowledge graph operations fail."""


class ExtractionError(GraphError):
    """Raised when entity/relationship extraction fails."""


class GraphBuildError(GraphError):
    """Raised when graph construction fails."""


class ConfigError(GoFetchError):
    """Raised when configuration is invalid."""
