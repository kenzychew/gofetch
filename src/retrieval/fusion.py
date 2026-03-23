"""Reciprocal Rank Fusion (RRF) for combining multiple retrieval signals."""

from src.logging import get_logger
from src.schemas import RetrievalResult

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF assigns each document a score of 1/(k + rank) for each list
    it appears in, then sums across lists. This is parameter-free
    (given k) and robust across different score scales.

    Args:
        ranked_lists: List of ranked result lists from different retrievers.
        k: Smoothing constant (default 60, from the original RRF paper).
        top_k: Number of results to return after fusion.

    Returns:
        Fused results sorted by RRF score.
    """
    # Accumulate RRF scores by chunk_id
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        for result in ranked_list:
            cid = result.chunk.chunk_id
            rrf_score = 1.0 / (k + result.rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

            # Keep the result with the highest original score
            if cid not in chunk_map or result.score > chunk_map[cid].score:
                chunk_map[cid] = result

    # Sort by RRF score and take top_k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    fused_results: list[RetrievalResult] = []
    for rank, cid in enumerate(sorted_ids, start=1):
        original = chunk_map[cid]
        fused_results.append(
            RetrievalResult(
                chunk=original.chunk,
                score=rrf_scores[cid],
                rank=rank,
                source_stage="rrf_fusion",
            )
        )

    sources = [rl[0].source_stage if rl else "empty" for rl in ranked_lists]
    logger.info(
        "RRF fusion",
        input_lists=len(ranked_lists),
        sources=sources,
        total_candidates=len(rrf_scores),
        output_count=len(fused_results),
    )
    return fused_results
