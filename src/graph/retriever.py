"""Graph-based chunk retrieval implementing BaseRetriever ABC."""

from src.config import GraphConfig
from src.graph.builder import KnowledgeGraph
from src.logging import get_logger
from src.retrieval.base import BaseRetriever
from src.schemas import Chunk, RetrievalResult

logger = get_logger(__name__)


class GraphRetriever(BaseRetriever):
    """Retrieves chunks by traversing the knowledge graph.

    Matches query terms to graph entities, traverses relationships
    to find related entities, and returns chunks associated with
    those entities. Acts as a third retrieval signal for RRF fusion.

    Attributes:
        graph: The knowledge graph instance.
        config: Graph configuration.
        chunk_lookup: Mapping from chunk_id to Chunk for result construction.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        config: GraphConfig,
        chunks: list[Chunk],
    ) -> None:
        """Initialize the graph retriever.

        Args:
            graph: A populated knowledge graph.
            config: Graph configuration with traversal settings.
            chunks: All indexed chunks for lookup by ID.
        """
        self.graph = graph
        self.config = config
        self.chunk_lookup: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve chunks via knowledge graph traversal.

        Tokenizes the query, matches against graph entities, traverses
        relationships, and returns chunks ranked by connection count.

        Args:
            query: The user's search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results from graph traversal.
        """
        query_terms = query.lower().split()
        chunk_ids = self.graph.get_related_chunk_ids(query_terms, hops=self.config.traversal_hops)

        results: list[RetrievalResult] = []
        for rank, chunk_id in enumerate(chunk_ids[:top_k], start=1):
            chunk = self.chunk_lookup.get(chunk_id)
            if chunk is None:
                logger.warning("Chunk not found in lookup", chunk_id=chunk_id)
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=1.0 / rank,
                    rank=rank,
                    source_stage="graph",
                )
            )

        logger.info("Graph retrieval", query_preview=query[:50], results=len(results))
        return results
