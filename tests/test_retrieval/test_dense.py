"""Tests for dense vector retrieval."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import AppConfig
from src.retrieval.dense import DenseRetriever


def _row(chunk_id: str) -> dict[str, object]:
    """Build a fake pgvector result row for the given chunk id.

    Args:
        chunk_id: Identifier to embed in the fake row.

    Returns:
        A dict shaped like an asyncpg Record for that chunk.
    """
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "source": "doc.pdf",
        "chunk_index": 0,
        "metadata": {},
        "score": 0.9,
    }


def _make_pool(rows_by_embedding: dict[tuple[float, ...], list[dict[str, object]]]) -> MagicMock:
    """Build a mock asyncpg pool whose fetch() result depends on the query embedding.

    Each fetch call awaits briefly before returning, so overlapping retrieve()
    calls genuinely interleave rather than running one after another. This is
    what would have exposed the old set_query_embedding()/retrieve() race,
    where a second call's embedding could overwrite the first call's shared
    state before the first call read it.

    Args:
        rows_by_embedding: Maps a query embedding (as a tuple) to the rows
            that call should get back.

    Returns:
        A mock pool supporting `async with pool.acquire() as conn`.
    """

    async def fetch(sql: str, query_vector: object, top_k: int) -> list[dict[str, object]]:
        key = tuple(round(float(v), 4) for v in query_vector)
        await asyncio.sleep(0.01)
        return rows_by_embedding[key]

    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=fetch)

    @asynccontextmanager
    async def acquire() -> object:
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool


@pytest.mark.asyncio
async def test_concurrent_retrieve_calls_use_their_own_embedding() -> None:
    """Overlapping retrieve() calls with different embeddings must not cross-contaminate."""
    embedding_a = [1.0, 0.0, 0.0]
    embedding_b = [0.0, 1.0, 0.0]

    pool = _make_pool(
        {
            tuple(embedding_a): [_row("chunk-a")],
            tuple(embedding_b): [_row("chunk-b")],
        }
    )
    retriever = DenseRetriever(pool, AppConfig())

    results_a, results_b = await asyncio.gather(
        retriever.retrieve("query a", 5, embedding_a),
        retriever.retrieve("query b", 5, embedding_b),
    )

    assert [r.chunk.chunk_id for r in results_a] == ["chunk-a"]
    assert [r.chunk.chunk_id for r in results_b] == ["chunk-b"]


@pytest.mark.asyncio
async def test_many_concurrent_retrieve_calls_each_get_their_own_results() -> None:
    """A larger fan-out of concurrent calls should still resolve per-call, not last-write-wins."""
    embeddings = [[float(i == j) for j in range(10)] for i in range(10)]
    rows_by_embedding = {tuple(emb): [_row(f"chunk-{i}")] for i, emb in enumerate(embeddings)}
    pool = _make_pool(rows_by_embedding)
    retriever = DenseRetriever(pool, AppConfig())

    results = await asyncio.gather(
        *(retriever.retrieve(f"query {i}", 5, emb) for i, emb in enumerate(embeddings))
    )

    for i, result in enumerate(results):
        assert [r.chunk.chunk_id for r in result] == [f"chunk-{i}"]
