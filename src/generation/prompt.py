"""Prompt builder with citation formatting and token budget management."""

import tiktoken

from src.config import GenerationConfig, PromptConfig
from src.logging import get_logger
from src.schemas import RetrievalResult

logger = get_logger(__name__)


class PromptBuilder:
    """Builds citation-aware prompts with token budget management.

    Formats retrieval results into numbered context chunks and constructs
    the full prompt for the LLM. Manages token budget to prevent exceeding
    context window limits by truncating lowest-scored chunks.

    Attributes:
        prompt_config: Prompt template configuration.
        gen_config: Generation configuration with token limits.
        encoding: Tiktoken encoding for token counting.
    """

    def __init__(self, prompt_config: PromptConfig, gen_config: GenerationConfig) -> None:
        """Initialize the prompt builder.

        Args:
            prompt_config: Prompt template configuration.
            gen_config: Generation configuration with token limits.
        """
        self.prompt_config = prompt_config
        self.gen_config = gen_config
        # Use cl100k_base encoding for approximate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self.encoding.encode(text))

    def format_chunks(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as numbered context chunks.

        Respects the token budget by including chunks in score order
        until the budget is exhausted.

        Args:
            results: Retrieval results to format as context.

        Returns:
            Formatted context string with numbered chunks.
        """
        formatted_chunks: list[str] = []
        total_tokens = 0
        max_tokens = self.gen_config.max_context_tokens

        for i, result in enumerate(results, start=1):
            chunk_text = self.prompt_config.chunk_template.format(
                index=i,
                source=result.chunk.source,
                score=result.score,
                text=result.chunk.text,
            )
            chunk_tokens = self.count_tokens(chunk_text)

            if total_tokens + chunk_tokens > max_tokens:
                logger.info(
                    "Token budget reached, truncating context",
                    included=i - 1,
                    total_available=len(results),
                    tokens_used=total_tokens,
                )
                break

            formatted_chunks.append(chunk_text)
            total_tokens += chunk_tokens

        return "\n\n".join(formatted_chunks)

    def build_messages(
        self,
        query: str,
        results: list[RetrievalResult],
        low_confidence: bool = False,
    ) -> list[dict[str, str]]:
        """Build the full message list for the LLM API call.

        Constructs system message with context chunks and user message
        with the query. Optionally prepends a low-confidence warning.

        Args:
            query: The user's question.
            results: Retrieval results to include as context.
            low_confidence: Whether to include a confidence warning.

        Returns:
            List of message dicts for the LLM API.
        """
        context = self.format_chunks(results)
        context_block = self.prompt_config.context_template.format(chunks=context)

        system_parts = [self.prompt_config.system_prompt.strip()]

        # Add few-shot examples
        few_shot_lines: list[str] = []
        for example in self.prompt_config.few_shot_examples:
            few_shot_lines.append(
                f"Example:\nQ: {example['question']}\n"
                f"Context: {example['context']}\n"
                f"A: {example['answer']}"
            )
        if few_shot_lines:
            system_parts.append("\n\n".join(few_shot_lines))

        system_parts.append(context_block)

        if low_confidence:
            system_parts.append(self.prompt_config.low_confidence_warning.strip())

        system_message = "\n\n".join(system_parts)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

        total_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        logger.info(
            "Built prompt",
            total_tokens=total_tokens,
            context_chunks=len(results),
            low_confidence=low_confidence,
        )
        return messages
