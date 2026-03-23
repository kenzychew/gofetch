"""Tests for prompt building."""

from src.config import GenerationConfig, PromptConfig
from src.generation.prompt import PromptBuilder
from src.schemas import Chunk, RetrievalResult


def _make_prompt_config() -> PromptConfig:
    """Create a test prompt config.

    Returns:
        PromptConfig with test templates.
    """
    return PromptConfig(
        system_prompt="You are a helpful assistant.",
        context_template="Context:\n{chunks}",
        chunk_template="[{index}] (Source: {source}, Score: {score:.3f})\n{text}",
        few_shot_examples=[],
        low_confidence_warning="Warning: low confidence.",
    )


def _make_gen_config() -> GenerationConfig:
    """Create a test generation config.

    Returns:
        GenerationConfig with test values.
    """
    return GenerationConfig(max_context_tokens=500)


def _make_result(chunk_id: str, text: str, score: float, rank: int) -> RetrievalResult:
    """Create a retrieval result for testing.

    Args:
        chunk_id: Chunk identifier.
        text: Chunk text content.
        score: Relevance score.
        rank: Rank position.

    Returns:
        A RetrievalResult instance.
    """
    chunk = Chunk(chunk_id=chunk_id, text=text, source="test.pdf", index=0)
    return RetrievalResult(chunk=chunk, score=score, rank=rank, source_stage="test")


def test_build_messages_basic() -> None:
    """Build messages should produce system and user messages."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())
    results = [_make_result("a", "Test chunk text", 0.9, 1)]

    messages = builder.build_messages("What is attention?", results)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is attention?"


def test_build_messages_includes_context() -> None:
    """System message should contain formatted chunk text."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())
    results = [_make_result("a", "Attention is a mechanism", 0.9, 1)]

    messages = builder.build_messages("What is attention?", results)

    assert "Attention is a mechanism" in messages[0]["content"]
    assert "[1]" in messages[0]["content"]


def test_build_messages_low_confidence_warning() -> None:
    """Low confidence flag should include warning in system message."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())
    results = [_make_result("a", "Test text", 0.1, 1)]

    messages = builder.build_messages("Test?", results, low_confidence=True)

    assert "Warning: low confidence" in messages[0]["content"]


def test_build_messages_no_warning_when_confident() -> None:
    """Warning should not appear when low_confidence is False."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())
    results = [_make_result("a", "Test text", 0.9, 1)]

    messages = builder.build_messages("Test?", results, low_confidence=False)

    assert "Warning: low confidence" not in messages[0]["content"]


def test_format_chunks_respects_token_budget() -> None:
    """Format chunks should stop adding chunks when token budget is reached."""
    config = _make_gen_config()
    config.max_context_tokens = 50  # Very small budget
    builder = PromptBuilder(_make_prompt_config(), config)

    # Create many results with substantial text
    results = [
        _make_result(f"chunk_{i}", f"This is a moderately long text chunk number {i} " * 5, 0.9, i)
        for i in range(10)
    ]

    formatted = builder.format_chunks(results)
    # Should include fewer than all 10 chunks
    assert formatted.count("[") < 10


def test_count_tokens() -> None:
    """Token count should be a positive integer for non-empty text."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())

    count = builder.count_tokens("Hello world")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty() -> None:
    """Empty string should have zero tokens."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())

    count = builder.count_tokens("")
    assert count == 0


def test_format_chunks_empty_results() -> None:
    """Formatting empty results should produce empty string."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())

    formatted = builder.format_chunks([])
    assert formatted == ""


def test_build_messages_multiple_chunks() -> None:
    """Messages should include all chunks that fit in the token budget."""
    builder = PromptBuilder(_make_prompt_config(), _make_gen_config())
    results = [
        _make_result("a", "First chunk about transformers", 0.9, 1),
        _make_result("b", "Second chunk about attention", 0.8, 2),
    ]

    messages = builder.build_messages("What is attention?", results)

    system_content = messages[0]["content"]
    assert "[1]" in system_content
    assert "[2]" in system_content
    assert "First chunk" in system_content
    assert "Second chunk" in system_content
