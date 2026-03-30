"""Text chunking implementations using LangChain text splitters."""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import IngestionConfig
from src.exceptions import ChunkingError
from src.ingestion.base import BaseChunker
from src.logging import get_logger
from src.schemas import Chunk, Document

logger = get_logger(__name__)


def _generate_chunk_id(source: str, index: int) -> str:
    """Generate a deterministic chunk ID from source and index.

    Uses the first 32 hex chars of a SHA-256 hash formatted as a UUID
    string, ensuring deterministic IDs across re-ingestion runs.

    Args:
        source: Source document identifier.
        index: Chunk index within the document.

    Returns:
        A UUID-formatted string used as chunk ID.
    """
    raw = f"{source}::{index}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class RecursiveChunker(BaseChunker):
    """Chunker using LangChain's RecursiveCharacterTextSplitter.

    Splits text using a hierarchy of separators (paragraphs, sentences,
    words) to keep semantically coherent units together.

    Attributes:
        splitter: The underlying LangChain text splitter.
    """

    def __init__(self, config: IngestionConfig) -> None:
        """Initialize the recursive chunker.

        Args:
            config: Ingestion configuration with chunk_size and overlap.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            "Initialized RecursiveChunker",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks using recursive character splitting.

        Args:
            document: The document to chunk.

        Returns:
            List of chunks with unique IDs and metadata.

        Raises:
            ChunkingError: If chunking produces no results.
        """
        if not document.content.strip():
            raise ChunkingError(f"Empty document content for source: {document.source}")

        texts = self.splitter.split_text(document.content)

        if not texts:
            raise ChunkingError(f"Chunking produced no results for source: {document.source}")

        chunks: list[Chunk] = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                chunk_id=_generate_chunk_id(document.source, i),
                text=text,
                source=document.source,
                index=i,
                metadata=document.metadata.copy(),
            )
            chunks.append(chunk)

        logger.info(
            "Chunked document",
            source=document.source,
            num_chunks=len(chunks),
            avg_chunk_len=sum(len(c.text) for c in chunks) // len(chunks),
        )
        return chunks
