"""Cross-encoder re-ranking for second-stage retrieval."""

import asyncio

from sentence_transformers import CrossEncoder

from src.config import RetrievalConfig
from src.exceptions import RerankError
from src.logging import get_logger
from src.retrieval.base import BaseReranker
from src.schemas import RetrievalResult

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Re-ranks results using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs for more
    accurate relevance scoring than bi-encoder similarity. The model
    is run via asyncio.to_thread to avoid blocking the event loop.

    Attributes:
        model: The loaded cross-encoder model.
        model_name: Name of the model for logging.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            config: Retrieval configuration with the reranker model name.
        """
        self.model_name = config.reranker_model
        logger.info("Loading cross-encoder model", model=self.model_name)
        self.model = CrossEncoder(self.model_name)
        logger.info("Cross-encoder model loaded", model=self.model_name)

    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """Score query-document pairs synchronously.

        Args:
            pairs: List of [query, document] pairs.

        Returns:
            List of relevance scores.
        """
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    async def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Re-rank results using the cross-encoder model.

        Runs the model in a thread pool to avoid blocking the async event loop.

        Args:
            query: The original user query.
            results: Candidate results from the fusion stage.
            top_k: Maximum number of results to return after re-ranking.

        Returns:
            Re-ranked results with updated scores, sorted by cross-encoder score.

        Raises:
            RerankError: If re-ranking fails.
        """
        if not results:
            return []

        pairs = [[query, result.chunk.text] for result in results]

        try:
            scores = await asyncio.to_thread(self._score_pairs, pairs)
        except Exception as exc:
            raise RerankError(f"Cross-encoder scoring failed: {exc}") from exc

        # Pair scores with results and sort
        scored = list(zip(scores, results, strict=True))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked: list[RetrievalResult] = []
        for rank, (score, result) in enumerate(scored[:top_k], start=1):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=score,
                    rank=rank,
                    source_stage="reranked",
                )
            )

        logger.info(
            "Reranked results",
            input_count=len(results),
            output_count=len(reranked),
            top_score=reranked[0].score if reranked else 0.0,
        )
        return reranked
