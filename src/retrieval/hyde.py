"""Hypothetical Document Embeddings (HyDE) for query expansion."""

from anthropic import AsyncAnthropic

from src.config import GenerationConfig
from src.logging import get_logger

logger = get_logger(__name__)

HYDE_PROMPT = (
    "Write a short, factual paragraph that would directly answer the following question. "
    "Write as if this paragraph appears in a research paper. Do not include preamble.\n\n"
    "Question: {query}\n\n"
    "Paragraph:"
)


async def generate_hypothetical_document(
    query: str,
    client: AsyncAnthropic,
    config: GenerationConfig,
) -> str:
    """Generate a hypothetical document that answers the query.

    Uses the LLM to generate a plausible answer paragraph, which is
    then embedded instead of (or alongside) the raw query. This bridges
    the vocabulary gap between questions and document text.

    Args:
        query: The user's search query.
        client: Async Anthropic client instance.
        config: Generation configuration with model settings.

    Returns:
        A hypothetical document text.
    """
    prompt = HYDE_PROMPT.format(query=query)

    response = await client.messages.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
    )

    content = response.content[0].text if response.content else ""
    logger.info("Generated HyDE document", query_preview=query[:50], hyde_len=len(content))
    return content
