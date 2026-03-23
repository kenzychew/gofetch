"""Abstract base classes for retrieval and re-ranking strategies."""

from abc import ABC, abstractmethod

from src.schemas import RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies.

    All retriever implementations (dense, sparse, graph) must
    inherit from this class. This enables pluggable retrieval
    signals that feed into RRF fusion.
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user's search query.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieval results ranked by relevance.
        """


class BaseReranker(ABC):
    """Abstract base class for re-ranking strategies.

    Re-rankers take a list of retrieval results and re-score
    them using a more expensive but accurate model.
    """

    @abstractmethod
    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Re-rank retrieval results using a cross-encoder or similar model.

        Args:
            query: The original user query.
            results: Candidate results from the fusion stage.
            top_k: Maximum number of results to return after re-ranking.

        Returns:
            Re-ranked results with updated scores.
        """
