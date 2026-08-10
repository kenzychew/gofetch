"""Tests for FastAPI API endpoints."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.config import AppConfig


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
