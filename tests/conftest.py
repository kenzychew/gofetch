"""Shared test fixtures for GoFetch tests."""

import pytest

from src.config import (
    AppConfig,
    GenerationConfig,
    GraphConfig,
    IngestionConfig,
    PromptConfig,
    RetrievalConfig,
)
from src.schemas import Chunk, Document, RetrievalResult


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing.

    Returns:
        A Document with test content.
    """
    return Document(
        content=(
            "The Transformer model was introduced in 2017. "
            "It uses self-attention mechanisms to process sequences in parallel. "
            "Multi-head attention allows the model to attend to information "
            "from different representation subspaces. "
            "The architecture consists of an encoder and decoder, "
            "each with multiple layers of self-attention and feed-forward networks. "
            "Positional encodings are added to give the model information about "
            "the position of tokens in the sequence."
        ),
        source="test-paper.pdf",
        metadata={"author": "Test Author"},
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing.

    Returns:
        List of test chunks with IDs and metadata.
    """
    return [
        Chunk(
            chunk_id="chunk_001",
            text="Self-attention computes a weighted sum of values based on query-key similarity.",
            source="paper_a.pdf",
            index=0,
        ),
        Chunk(
            chunk_id="chunk_002",
            text="The Transformer architecture uses multi-head attention for parallel processing.",
            source="paper_a.pdf",
            index=1,
        ),
        Chunk(
            chunk_id="chunk_003",
            text="BERT uses masked language modeling as a pre-training objective.",
            source="paper_b.pdf",
            index=0,
        ),
        Chunk(
            chunk_id="chunk_004",
            text="Retrieval-Augmented Generation combines a retriever with a generator model.",
            source="paper_c.pdf",
            index=0,
        ),
        Chunk(
            chunk_id="chunk_005",
            text="Dense passage retrieval uses BERT to encode queries and passages.",
            source="paper_c.pdf",
            index=1,
        ),
    ]


@pytest.fixture
def sample_retrieval_results(sample_chunks: list[Chunk]) -> list[RetrievalResult]:
    """Create sample retrieval results for testing.

    Args:
        sample_chunks: Sample chunks fixture.

    Returns:
        List of RetrievalResult objects.
    """
    return [
        RetrievalResult(chunk=sample_chunks[0], score=0.95, rank=1, source_stage="dense"),
        RetrievalResult(chunk=sample_chunks[1], score=0.88, rank=2, source_stage="dense"),
        RetrievalResult(chunk=sample_chunks[2], score=0.72, rank=3, source_stage="sparse"),
        RetrievalResult(chunk=sample_chunks[3], score=0.65, rank=4, source_stage="sparse"),
        RetrievalResult(chunk=sample_chunks[4], score=0.55, rank=5, source_stage="dense"),
    ]


@pytest.fixture
def ingestion_config() -> IngestionConfig:
    """Create test ingestion config.

    Returns:
        IngestionConfig with test values.
    """
    return IngestionConfig(chunk_size=200, chunk_overlap=20)


@pytest.fixture
def retrieval_config() -> RetrievalConfig:
    """Create test retrieval config.

    Returns:
        RetrievalConfig with test values.
    """
    return RetrievalConfig()


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Create test generation config.

    Returns:
        GenerationConfig with test values.
    """
    return GenerationConfig()


@pytest.fixture
def prompt_config() -> PromptConfig:
    """Create test prompt config.

    Returns:
        PromptConfig with test templates.
    """
    return PromptConfig(
        system_prompt="You are a helpful assistant. Answer using only the provided context.",
        context_template="Context:\n{chunks}",
        chunk_template="[{index}] (Source: {source}, Score: {score:.3f})\n{text}",
        few_shot_examples=[],
        low_confidence_warning="Warning: low confidence answer.",
    )


@pytest.fixture
def app_config() -> AppConfig:
    """Create test application config.

    Returns:
        AppConfig with test values.
    """
    return AppConfig(
        ingestion=IngestionConfig(chunk_size=200, chunk_overlap=20),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(),
        graph=GraphConfig(),
    )
