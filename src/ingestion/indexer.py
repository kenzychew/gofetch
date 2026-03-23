"""Qdrant vector indexing and BM25 index building."""

import pickle
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from src.config import AppConfig
from src.exceptions import IndexingError
from src.logging import get_logger
from src.schemas import Chunk

logger = get_logger(__name__)


class VectorIndexer:
    """Manages Qdrant collection and upserts chunk embeddings.

    Handles collection creation and batch upsert of chunk vectors
    with associated payloads.

    Attributes:
        client: Async Qdrant client instance.
        collection_name: Name of the Qdrant collection.
        embedding_dim: Dimensionality of the embedding vectors.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the vector indexer.

        Args:
            config: Application configuration with Qdrant URL and collection settings.
        """
        self.client = AsyncQdrantClient(url=config.qdrant_url)
        self.collection_name = config.collection_name
        self.embedding_dim = config.ingestion.embedding_dim

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist.

        Raises:
            IndexingError: If collection creation fails.
        """
        try:
            collections = await self.client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.collection_name not in existing_names:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection",
                    collection=self.collection_name,
                    dim=self.embedding_dim,
                )
            else:
                logger.info("Qdrant collection already exists", collection=self.collection_name)
        except Exception as exc:
            raise IndexingError(f"Failed to ensure Qdrant collection: {exc}") from exc

    async def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert chunk embeddings into Qdrant.

        Args:
            chunks: List of chunks with populated embeddings.

        Returns:
            Number of chunks upserted.

        Raises:
            IndexingError: If upsert fails.
        """
        if not chunks:
            return 0

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        try:
            # Qdrant supports batch upsert, process in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )

            logger.info("Upserted chunks to Qdrant", count=len(chunks))
            return len(chunks)
        except Exception as exc:
            raise IndexingError(f"Failed to upsert chunks to Qdrant: {exc}") from exc

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self.client.close()


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
