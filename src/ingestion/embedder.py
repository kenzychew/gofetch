"""Sentence-transformers embedding wrapper."""

from sentence_transformers import SentenceTransformer

from src.config import IngestionConfig
from src.exceptions import EmbeddingError
from src.logging import get_logger
from src.schemas import Chunk

logger = get_logger(__name__)


class Embedder:
    """Generates dense vector embeddings using sentence-transformers.

    Runs locally with no API calls. The model is loaded once at
    initialization and reused for all embedding operations.

    Attributes:
        model: The loaded sentence-transformers model.
        model_name: Name of the model for logging.
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the embedder with a sentence-transformers model.

        Args:
            config: Ingestion configuration with the embedding model name.
        """
        self.model_name = config.embedding_model
        logger.info("Loading embedding model", model=self.model_name)
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded", model=self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors as float lists.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add embeddings to a list of chunks in place.

        Args:
            chunks: List of chunks to embed.

        Returns:
            The same chunks with embeddings populated.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding

        logger.info("Embedded chunks", count=len(chunks), model=self.model_name)
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector as a float list.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        result = self.embed_texts([query])
        return result[0]
