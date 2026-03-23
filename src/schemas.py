"""Shared data models for the GoFetch RAG system."""

from dataclasses import dataclass, field


@dataclass
class Document:
    """A loaded document before chunking.

    Attributes:
        content: Full text content of the document.
        source: File path or identifier of the source.
        metadata: Additional metadata key-value pairs.
    """

    content: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk with its embedding and metadata.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The chunk text content.
        source: Source document identifier.
        index: Position index within the source document.
        embedding: Dense vector embedding (None before embedding step).
        metadata: Additional metadata key-value pairs.
    """

    chunk_id: str
    text: str
    source: str
    index: int
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval result with score and provenance.

    Attributes:
        chunk: The retrieved chunk.
        score: Relevance score (higher is better).
        rank: Position in the result list (1-indexed).
        source_stage: Which retrieval stage produced this result.
    """

    chunk: Chunk
    score: float
    rank: int
    source_stage: str


@dataclass
class CitedSource:
    """A source used in the generated answer with citation index.

    Attributes:
        index: Citation number (1-indexed) used in the answer text.
        source: Source document identifier.
        text: Chunk text used as context.
        score: Relevance score from retrieval.
    """

    index: int
    source: str
    text: str
    score: float


@dataclass
class QueryResponse:
    """Complete response to a user query.

    Attributes:
        answer: Generated answer text with inline citations.
        citations: List of cited sources referenced in the answer.
        latency_ms: Total latency breakdown by pipeline stage.
        confidence: Confidence score from re-ranker (top result score).
        low_confidence: Whether the confidence is below threshold.
    """

    answer: str
    citations: list[CitedSource]
    latency_ms: dict[str, float]
    confidence: float
    low_confidence: bool = False


@dataclass
class GraphEntity:
    """An entity extracted from text for the knowledge graph.

    Attributes:
        name: Normalized entity name.
        entity_type: Category of the entity (eg concept, model, technique).
        chunk_ids: IDs of chunks where this entity appears.
    """

    name: str
    entity_type: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GraphRelationship:
    """A relationship between two entities in the knowledge graph.

    Attributes:
        source: Source entity name.
        target: Target entity name.
        relation: Type of relationship (eg introduced_in, used_by).
        chunk_ids: IDs of chunks where this relationship is mentioned.
    """

    source: str
    target: str
    relation: str
    chunk_ids: list[str] = field(default_factory=list)
