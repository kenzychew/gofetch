"""Dense vector retrieval using PostgreSQL with pgvector."""

import asyncpg
import numpy as np

from src.config import AppConfig
from src.exceptions import VectorSearchError
from src.logging import get_logger
from src.retrieval.base import BaseRetriever
from src.schemas import Chunk, RetrievalResult

logger = get_logger(__name__)

# Column "chunk_index" maps to Chunk.index (avoids SQL reserved word "index")
SEARCH_SQL = """
SELECT chunk_id, text, source, chunk_index, metadata,
       1 - (embedding <=> $1::vector) AS score
FROM {table}
ORDER BY embedding <=> $1::vector
LIMIT $2
"""


class DenseRetriever(BaseRetriever):
    """Retrieves chunks via cosine similarity search in PostgreSQL with pgvector.

    Uses asyncpg for non-blocking vector search against an HNSW index.
    Embeddings are generated externally and passed as query vectors.

    Attributes:
        pool: asyncpg connection pool.
        table_name: Name of the chunks table to search.
        query_embedding: Cached query embedding for the current query.
    """

    def __init__(self, pool: asyncpg.Pool, config: AppConfig) -> None:
        """Initialize the dense retriever.

        Args:
            pool: asyncpg connection pool.
            config: Application configuration with table name.
        """
        self.pool = pool
        self.table_name = config.table_name
        self._query_embedding: list[float] | None = None

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
            VectorSearchError: If the vector search fails.
        """
        # Snapshot the embedding to avoid race conditions with concurrent
        # requests overwriting _query_embedding on this singleton
        embedding = self._query_embedding
        if embedding is None:
            raise VectorSearchError("Query embedding not set. Call set_query_embedding() first.")

        try:
            sql = SEARCH_SQL.format(table=self.table_name)
            query_vector = np.array(embedding, dtype=np.float32)

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, query_vector, top_k)

            results: list[RetrievalResult] = []
            for rank, row in enumerate(rows, start=1):
                metadata = dict(row["metadata"]) if row["metadata"] else {}
                chunk = Chunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    source=row["source"],
                    index=row["chunk_index"],
                    metadata=metadata,
                )
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=float(row["score"]),
                        rank=rank,
                        source_stage="dense",
                    )
                )

            logger.info("Dense retrieval", query_preview=query[:50], results=len(results))
            return results
        except Exception as exc:
            raise VectorSearchError(f"pgvector search failed: {exc}") from exc
