"""Dense vector retrieval using Qdrant."""

from qdrant_client import AsyncQdrantClient

from src.config import AppConfig
from src.exceptions import VectorSearchError
from src.logging import get_logger
from src.retrieval.base import BaseRetriever
from src.schemas import Chunk, RetrievalResult

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Retrieves chunks via cosine similarity search in Qdrant.

    Uses the async Qdrant client for non-blocking vector search.
    Embeddings are generated externally and passed as query vectors.

    Attributes:
        client: Async Qdrant client instance.
        collection_name: Name of the Qdrant collection to search.
        query_embedding: Cached query embedding for the current query.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the dense retriever.

        Args:
            config: Application configuration with Qdrant URL and collection name.
        """
        self.client = AsyncQdrantClient(url=config.qdrant_url)
        self.collection_name = config.collection_name
        self._query_embedding: list[float] = []

    def set_query_embedding(self, embedding: list[float]) -> None:
        """Set the query embedding for the next retrieval call.

        The embedding is generated externally (by the Embedder) and passed
        here to decouple retrieval from embedding logic.

        Args:
            embedding: The query's dense vector embedding.
        """
        self._query_embedding = embedding

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve chunks by cosine similarity to the query embedding.

        Args:
            query: The user's search query (used for logging only;
                the actual search uses the pre-set embedding).
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results sorted by similarity score.

        Raises:
            VectorSearchError: If the Qdrant search fails.
        """
        if not self._query_embedding:
            raise VectorSearchError("Query embedding not set. Call set_query_embedding() first.")

        try:
            hits = await self.client.query_points(
                collection_name=self.collection_name,
                query=self._query_embedding,
                limit=top_k,
                with_payload=True,
            )

            results: list[RetrievalResult] = []
            for rank, hit in enumerate(hits.points, start=1):
                payload = hit.payload or {}
                chunk = Chunk(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    text=str(payload.get("text", "")),
                    source=str(payload.get("source", "")),
                    index=int(payload.get("index", 0)),
                    metadata=payload.get("metadata", {}),
                )
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=hit.score,
                        rank=rank,
                        source_stage="dense",
                    )
                )

            logger.info("Dense retrieval", query_preview=query[:50], results=len(results))
            return results
        except Exception as exc:
            raise VectorSearchError(f"Qdrant search failed: {exc}") from exc

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self.client.close()
