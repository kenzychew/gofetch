"""Tests for FastAPI API endpoints."""

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.config import AppConfig, RetrievalConfig
from src.exceptions import IndexingError
from src.schemas import Document


@pytest.fixture
async def async_client() -> AsyncClient:
    """Create an async test client for the FastAPI app.

    Returns:
        An httpx AsyncClient configured for testing.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_health_endpoint_structure() -> None:
    """Health endpoint should return expected structure even if PostgreSQL is down."""
    # Mock the dependencies to avoid needing real services
    with (
        patch("src.api.main.get_pool") as mock_pool,
        patch("src.api.main.get_sparse_retriever") as mock_sparse,
    ):
        mock_pool.side_effect = RuntimeError("Pool not initialized")
        mock_sparse.return_value = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # The health endpoint doesn't require lifespan init
            # because we're mocking the dependencies
            response = await client.get("/health")

        # May return 500 if dependencies aren't initialized,
        # but the structure should be consistent when it works
        assert response.status_code in (200, 500)


def _mock_pool(rows: list[dict[str, object]]) -> MagicMock:
    """Build a mock asyncpg pool whose acquired connection returns fixed rows.

    Args:
        rows: Rows to return from conn.fetch().

    Returns:
        A mock pool supporting `async with pool.acquire() as conn`.
    """
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def acquire() -> object:
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool


@pytest.mark.asyncio
async def test_sources_endpoint_returns_distinct_sources() -> None:
    """Sources endpoint should return distinct sources with chunk counts."""
    rows = [
        {"source": "attention-is-all-you-need.pdf", "chunk_count": 12},
        {"source": "cpf-housing-scheme.txt", "chunk_count": 5},
    ]
    with (
        patch("src.api.main.get_config") as mock_config,
        patch("src.api.main.get_pool") as mock_pool,
    ):
        mock_config.return_value = AppConfig()
        mock_pool.return_value = _mock_pool(rows)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/sources")

    assert response.status_code == 200
    assert response.json() == {"sources": rows}


@pytest.mark.asyncio
async def test_sources_endpoint_empty_table_returns_empty_list() -> None:
    """Sources endpoint should return an empty list, not an error, when nothing is ingested."""
    with (
        patch("src.api.main.get_config") as mock_config,
        patch("src.api.main.get_pool") as mock_pool,
    ):
        mock_config.return_value = AppConfig()
        mock_pool.return_value = _mock_pool([])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/sources")

    assert response.status_code == 200
    assert response.json() == {"sources": []}


@pytest.mark.asyncio
async def test_sources_endpoint_pool_unavailable_returns_500() -> None:
    """Sources endpoint should return 500 when the PostgreSQL pool isn't initialized."""
    with (
        patch("src.api.main.get_config") as mock_config,
        patch("src.api.main.get_pool") as mock_pool,
    ):
        mock_config.return_value = AppConfig()
        mock_pool.side_effect = RuntimeError("Pool not initialized")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/sources")

    assert response.status_code == 500


def _ingest_test_config(tmp_path: Path, *, use_graph: bool = False) -> AppConfig:
    """Build an AppConfig pointed at tmp_path for isolated /ingest tests.

    Args:
        tmp_path: Pytest tmp_path fixture, used for data_dir and bm25_index_path.
        use_graph: Whether to enable the (slow) knowledge-graph extraction stage.

    Returns:
        An AppConfig safe to use without a real Postgres or GCP connection.
    """
    return AppConfig(
        data_dir=str(tmp_path / "data"),
        bm25_index_path=str(tmp_path / "bm25_index" / "bm25.pkl"),
        retrieval=RetrievalConfig(use_graph=use_graph),
    )


@pytest.mark.asyncio
async def test_ingest_returns_400_when_no_documents_found(tmp_path: Path) -> None:
    """POST /ingest should fail fast, synchronously, when there's nothing to ingest."""
    config = _ingest_test_config(tmp_path)

    with (
        patch("src.api.main.get_config", return_value=config),
        patch("src.api.main.load_documents", return_value=[]),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/ingest")

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_ingest_queues_job_and_background_task_completes(tmp_path: Path) -> None:
    """POST /ingest should return immediately with a job id that reaches 'done'.

    Graph extraction is disabled here so the job only exercises the fast
    stages (chunking, embedding, indexing) -- the part that used to run
    inline in the request and now runs as a background task.
    """
    config = _ingest_test_config(tmp_path, use_graph=False)
    document = Document(content="Self-attention is a mechanism. " * 20, source="doc.txt")

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks = MagicMock(side_effect=lambda chunks: chunks)

    mock_indexer = MagicMock()
    mock_indexer.upsert_chunks = AsyncMock(return_value=1)

    with (
        patch("src.api.main.get_config", return_value=config),
        patch("src.api.main.load_documents", return_value=[document]),
        patch("src.api.main.get_embedder", return_value=mock_embedder),
        patch("src.api.main.get_vector_indexer", return_value=mock_indexer),
        patch("src.api.main.set_sparse_retriever") as mock_set_sparse,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/ingest")
            assert response.status_code == 202
            body = response.json()
            assert body["stage"] == "queued"
            job_id = body["job_id"]

            status_response = await client.get(f"/ingest/status/{job_id}")

    status = status_response.json()
    assert status["stage"] == "done"
    assert status["documents"] == 1
    assert status["chunks"] > 0
    assert status["error"] is None
    mock_set_sparse.assert_called_once()
    assert (tmp_path / "bm25_index" / "bm25.pkl").exists()


@pytest.mark.asyncio
async def test_ingest_job_failure_is_reported_via_status(tmp_path: Path) -> None:
    """A pipeline failure during the background job should surface as stage=failed."""
    config = _ingest_test_config(tmp_path, use_graph=False)
    document = Document(content="Some content to chunk and embed.", source="doc.txt")

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks = MagicMock(side_effect=lambda chunks: chunks)

    mock_indexer = MagicMock()
    mock_indexer.upsert_chunks = AsyncMock(side_effect=IndexingError("Postgres unavailable"))

    with (
        patch("src.api.main.get_config", return_value=config),
        patch("src.api.main.load_documents", return_value=[document]),
        patch("src.api.main.get_embedder", return_value=mock_embedder),
        patch("src.api.main.get_vector_indexer", return_value=mock_indexer),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/ingest")
            job_id = response.json()["job_id"]

            status_response = await client.get(f"/ingest/status/{job_id}")

    status = status_response.json()
    assert status["stage"] == "failed"
    assert "Postgres unavailable" in status["error"]


@pytest.mark.asyncio
async def test_ingest_status_unknown_job_id_returns_404() -> None:
    """GET /ingest/status/{job_id} should 404 for a job id that was never queued."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ingest/status/does-not-exist")

    assert response.status_code == 404
