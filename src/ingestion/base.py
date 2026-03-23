"""Abstract base class for text chunking strategies."""

from abc import ABC, abstractmethod

from src.schemas import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies.

    All chunking implementations must inherit from this class
    and implement the chunk method. This enables strategy-pattern
    swapping of chunking algorithms via configuration.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of chunks with unique IDs and metadata.
        """
