"""Tests for RRF fusion logic."""

from src.retrieval.fusion import reciprocal_rank_fusion
from src.schemas import Chunk, RetrievalResult


def _make_result(chunk_id: str, rank: int, stage: str, score: float = 1.0) -> RetrievalResult:
    """Helper to create a RetrievalResult for testing.

    Args:
        chunk_id: Unique chunk identifier.
        rank: Rank position.
        stage: Source retrieval stage name.
        score: Relevance score.

    Returns:
        A RetrievalResult instance.
    """
    chunk = Chunk(chunk_id=chunk_id, text=f"text for {chunk_id}", source="test.pdf", index=0)
    return RetrievalResult(chunk=chunk, score=score, rank=rank, source_stage=stage)


def test_rrf_single_list() -> None:
    """RRF with a single list should preserve ranking order."""
    results = [
        _make_result("a", 1, "dense", 0.9),
        _make_result("b", 2, "dense", 0.8),
        _make_result("c", 3, "dense", 0.7),
    ]
    fused = reciprocal_rank_fusion([results], k=60, top_k=3)

    assert len(fused) == 3
    assert fused[0].chunk.chunk_id == "a"
    assert fused[1].chunk.chunk_id == "b"
    assert fused[2].chunk.chunk_id == "c"


def test_rrf_two_lists_overlapping() -> None:
    """RRF should boost chunks that appear in multiple lists."""
    dense_results = [
        _make_result("a", 1, "dense", 0.9),
        _make_result("b", 2, "dense", 0.8),
        _make_result("c", 3, "dense", 0.7),
    ]
    sparse_results = [
        _make_result("b", 1, "sparse", 0.95),
        _make_result("d", 2, "sparse", 0.85),
        _make_result("a", 3, "sparse", 0.75),
    ]

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60, top_k=5)

    # Both "a" and "b" appear in both lists, so they should rank highest
    top_ids = [r.chunk.chunk_id for r in fused[:2]]
    assert "a" in top_ids
    assert "b" in top_ids


def test_rrf_non_overlapping() -> None:
    """RRF should interleave results from non-overlapping lists."""
    list1 = [_make_result("a", 1, "dense"), _make_result("b", 2, "dense")]
    list2 = [_make_result("c", 1, "sparse"), _make_result("d", 2, "sparse")]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=4)

    assert len(fused) == 4
    # Rank 1 items from each list should tie and both appear in top 2
    top_ids = {r.chunk.chunk_id for r in fused[:2]}
    assert "a" in top_ids
    assert "c" in top_ids


def test_rrf_top_k_truncation() -> None:
    """RRF should respect top_k limit."""
    results = [_make_result(f"chunk_{i}", i + 1, "dense") for i in range(20)]
    fused = reciprocal_rank_fusion([results], k=60, top_k=5)

    assert len(fused) == 5


def test_rrf_empty_lists() -> None:
    """RRF should handle empty input lists gracefully."""
    fused = reciprocal_rank_fusion([], k=60, top_k=10)
    assert len(fused) == 0


def test_rrf_one_empty_list() -> None:
    """RRF should handle a mix of empty and non-empty lists."""
    results = [_make_result("a", 1, "dense")]
    fused = reciprocal_rank_fusion([results, []], k=60, top_k=5)

    assert len(fused) == 1
    assert fused[0].chunk.chunk_id == "a"


def test_rrf_scores_are_correct() -> None:
    """RRF scores should equal sum of 1/(k+rank) across lists."""
    k = 60
    dense = [_make_result("a", 1, "dense")]
    sparse = [_make_result("a", 2, "sparse")]

    fused = reciprocal_rank_fusion([dense, sparse], k=k, top_k=1)

    expected_score = 1.0 / (k + 1) + 1.0 / (k + 2)
    assert abs(fused[0].score - expected_score) < 1e-10


def test_rrf_three_lists() -> None:
    """RRF should work with three input lists (for graph signal)."""
    dense = [_make_result("a", 1, "dense"), _make_result("b", 2, "dense")]
    sparse = [_make_result("b", 1, "sparse"), _make_result("c", 2, "sparse")]
    graph = [_make_result("c", 1, "graph"), _make_result("a", 2, "graph")]

    fused = reciprocal_rank_fusion([dense, sparse, graph], k=60, top_k=3)

    # All three should be present -- each appears in 2 out of 3 lists
    ids = {r.chunk.chunk_id for r in fused}
    assert ids == {"a", "b", "c"}


def test_rrf_source_stage_is_updated() -> None:
    """Fused results should have source_stage set to rrf_fusion."""
    results = [_make_result("a", 1, "dense")]
    fused = reciprocal_rank_fusion([results], k=60, top_k=1)

    assert fused[0].source_stage == "rrf_fusion"


def test_rrf_ranks_are_sequential() -> None:
    """Fused results should have sequential 1-indexed ranks."""
    results = [_make_result(f"chunk_{i}", i + 1, "dense") for i in range(5)]
    fused = reciprocal_rank_fusion([results], k=60, top_k=5)

    for i, result in enumerate(fused, start=1):
        assert result.rank == i
