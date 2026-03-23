"""BM25 sparse retrieval."""

import numpy as np
from rank_bm25 import BM25Okapi

from src.logging import get_logger
from src.retrieval.base import BaseRetriever
from src.schemas import Chunk, RetrievalResult

logger = get_logger(__name__)


class SparseRetriever(BaseRetriever):
    """Retrieves chunks using BM25 keyword matching.

    The BM25 index is loaded from disk (built during ingestion)
    and kept in memory for fast retrieval.

    Attributes:
        bm25: The BM25Okapi index.
        chunks: The indexed chunks corresponding to the BM25 corpus.
    """

    def __init__(self, bm25: BM25Okapi, chunks: list[Chunk]) -> None:
        """Initialize the sparse retriever with a pre-built BM25 index.

        Args:
            bm25: A pre-built BM25Okapi index.
            chunks: The chunks that were indexed (same order as the BM25 corpus).
        """
        self.bm25 = bm25
        self.chunks = chunks

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve chunks by BM25 keyword relevance.

        Args:
            query: The user's search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results sorted by BM25 score.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0:
                break
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    source_stage="sparse",
                )
            )

        logger.info("Sparse retrieval", query_preview=query[:50], results=len(results))
        return results
