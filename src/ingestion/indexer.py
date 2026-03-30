"""PostgreSQL pgvector indexing and BM25 index building."""

import json
import pickle
from pathlib import Path

import asyncpg
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import AppConfig
from src.exceptions import IndexingError
from src.logging import get_logger
from src.schemas import Chunk

logger = get_logger(__name__)

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector({dim}) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
ON {table} USING hnsw (embedding vector_cosine_ops)
"""

UPSERT_SQL = """
INSERT INTO {table} (chunk_id, text, source, chunk_index, embedding, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (chunk_id) DO UPDATE SET
    text = EXCLUDED.text,
    source = EXCLUDED.source,
    chunk_index = EXCLUDED.chunk_index,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata
"""


class VectorIndexer:
    """Manages PostgreSQL pgvector table and upserts chunk embeddings.

    Handles table creation, HNSW index setup, and batch upsert of
    chunk vectors with associated metadata.

    Attributes:
        pool: asyncpg connection pool.
        table_name: Name of the chunks table.
        embedding_dim: Dimensionality of the embedding vectors.
    """

    def __init__(self, pool: asyncpg.Pool, config: AppConfig) -> None:
        """Initialize the vector indexer.

        Args:
            pool: asyncpg connection pool.
            config: Application configuration with table and embedding settings.
        """
        self.pool = pool
        self.table_name = config.table_name
        self.embedding_dim = config.ingestion.embedding_dim

    async def ensure_table(self) -> None:
        """Create the pgvector extension, chunks table, and HNSW index if needed.

        Raises:
            IndexingError: If table creation fails.
        """
        try:
            async with self.pool.acquire() as conn:
                # Wrap DDL in a transaction so partial failures
                # (eg table created but index fails) get rolled back
                async with conn.transaction():
                    await conn.execute(CREATE_EXTENSION_SQL)
                    await conn.execute(
                        CREATE_TABLE_SQL.format(table=self.table_name, dim=self.embedding_dim)
                    )
                    await conn.execute(CREATE_INDEX_SQL.format(table=self.table_name))

            logger.info(
                "Ensured pgvector table",
                table=self.table_name,
                dim=self.embedding_dim,
            )
        except Exception as exc:
            raise IndexingError(f"Failed to ensure pgvector table: {exc}") from exc

    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert chunk embeddings into PostgreSQL.

        Args:
            chunks: List of chunks with populated embeddings.

        Returns:
            Number of chunks upserted.

        Raises:
            IndexingError: If upsert fails.
        """
        if not chunks:
            return 0

        try:
            sql = UPSERT_SQL.format(table=self.table_name)
            rows = [
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.source,
                    chunk.index,
                    np.array(chunk.embedding, dtype=np.float32),
                    json.dumps(chunk.metadata),
                )
                for chunk in chunks
            ]

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(sql, rows)

            logger.info("Upserted chunks to PostgreSQL", count=len(chunks))
            return len(chunks)
        except Exception as exc:
            raise IndexingError(f"Failed to upsert chunks to PostgreSQL: {exc}") from exc


class BM25Indexer:
    """Builds and persists a BM25 index from chunks.

    The BM25 index is stored in memory for fast retrieval and
    pickled to disk for persistence across restarts.

    Attributes:
        index_path: File path for the pickled BM25 index.
    """

    def __init__(self, index_path: str) -> None:
        """Initialize the BM25 indexer.

        Args:
            index_path: File path to save/load the BM25 index.
        """
        self.index_path = Path(index_path)

    def build_index(self, chunks: list[Chunk]) -> BM25Okapi:
        """Build a BM25 index from chunks.

        Tokenizes chunk texts and builds the BM25Okapi index.
        Persists the index and chunk mapping to disk.

        Args:
            chunks: List of chunks to index.

        Returns:
            The built BM25Okapi index.

        Raises:
            IndexingError: If index building fails.
        """
        if not chunks:
            raise IndexingError("Cannot build BM25 index from empty chunk list")

        try:
            tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
            bm25 = BM25Okapi(tokenized_corpus)

            # Save index and chunk mapping
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            index_data = {
                "bm25": bm25,
                "chunks": chunks,
                "tokenized_corpus": tokenized_corpus,
            }
            with open(self.index_path, "wb") as f:
                pickle.dump(index_data, f)

            logger.info(
                "Built BM25 index",
                num_chunks=len(chunks),
                index_path=str(self.index_path),
            )
            return bm25
        except Exception as exc:
            raise IndexingError(f"Failed to build BM25 index: {exc}") from exc

    def load_index(self) -> tuple[BM25Okapi, list[Chunk]]:
        """Load a previously built BM25 index from disk.

        Returns:
            Tuple of (BM25 index, list of indexed chunks).

        Raises:
            IndexingError: If the index file cannot be loaded.
        """
        if not self.index_path.exists():
            raise IndexingError(f"BM25 index not found at: {self.index_path}")

        try:
            with open(self.index_path, "rb") as f:
                # Pickle is safe: file is generated internally during ingestion,
                # never from untrusted external input
                index_data = pickle.load(f)

            bm25: BM25Okapi = index_data["bm25"]
            chunks: list[Chunk] = index_data["chunks"]
            logger.info("Loaded BM25 index", num_chunks=len(chunks))
            return bm25, chunks
        except Exception as exc:
            raise IndexingError(f"Failed to load BM25 index: {exc}") from exc
