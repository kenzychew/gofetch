"""Tests for text chunking."""

import pytest

from src.config import IngestionConfig
from src.exceptions import ChunkingError
from src.ingestion.chunker import RecursiveChunker, _generate_chunk_id
from src.schemas import Document


def test_generate_chunk_id_deterministic() -> None:
    """Chunk IDs should be deterministic for the same input."""
    id1 = _generate_chunk_id("test.pdf", 0)
    id2 = _generate_chunk_id("test.pdf", 0)
    assert id1 == id2


def test_generate_chunk_id_unique_per_index() -> None:
    """Different indices should produce different chunk IDs."""
    id1 = _generate_chunk_id("test.pdf", 0)
    id2 = _generate_chunk_id("test.pdf", 1)
    assert id1 != id2


def test_generate_chunk_id_unique_per_source() -> None:
    """Different sources should produce different chunk IDs."""
    id1 = _generate_chunk_id("a.pdf", 0)
    id2 = _generate_chunk_id("b.pdf", 0)
    assert id1 != id2


def test_recursive_chunker_basic(sample_document: Document) -> None:
    """RecursiveChunker should produce multiple chunks from a document."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text
        assert chunk.source == "test-paper.pdf"
        assert chunk.chunk_id


def test_recursive_chunker_preserves_source(sample_document: Document) -> None:
    """All chunks should carry the source document name."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)
    chunks = chunker.chunk(sample_document)

    for chunk in chunks:
        assert chunk.source == sample_document.source


def test_recursive_chunker_sequential_indices(sample_document: Document) -> None:
    """Chunks should have sequential 0-indexed positions."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)
    chunks = chunker.chunk(sample_document)

    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_recursive_chunker_empty_content_raises() -> None:
    """Chunking an empty document should raise ChunkingError."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)
    empty_doc = Document(content="   ", source="empty.pdf")

    with pytest.raises(ChunkingError):
        chunker.chunk(empty_doc)


def test_recursive_chunker_respects_chunk_size() -> None:
    """Chunks should not significantly exceed the configured chunk_size."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)

    long_text = "word " * 500
    doc = Document(content=long_text, source="long.txt")
    chunks = chunker.chunk(doc)

    for chunk in chunks:
        # Allow some tolerance for the text splitter's behavior
        assert len(chunk.text) <= config.chunk_size + 50


def test_recursive_chunker_preserves_metadata() -> None:
    """Chunks should inherit metadata from the source document."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)

    doc = Document(
        content="Some content that is long enough to chunk into pieces " * 10,
        source="meta.pdf",
        metadata={"author": "Test", "year": "2024"},
    )
    chunks = chunker.chunk(doc)

    for chunk in chunks:
        assert chunk.metadata == {"author": "Test", "year": "2024"}


def test_recursive_chunker_unique_ids() -> None:
    """All chunks from a document should have unique IDs."""
    config = IngestionConfig(chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(config)

    doc = Document(
        content="Some content that is long enough to produce multiple chunks " * 20,
        source="unique.pdf",
    )
    chunks = chunker.chunk(doc)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
