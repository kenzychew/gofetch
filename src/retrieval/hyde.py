"""Hypothetical Document Embeddings (HyDE) for query expansion."""

from google import genai
from google.genai import types

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
    client: genai.Client,
    config: GenerationConfig,
) -> str:
    """Generate a hypothetical document that answers the query.

    Uses the LLM to generate a plausible answer paragraph, which is
    then embedded instead of (or alongside) the raw query. This bridges
    the vocabulary gap between questions and document text.

    Args:
        query: The user's search query.
        client: Google GenAI client instance.
        config: Generation configuration with model settings.

    Returns:
        A hypothetical document text.
    """
    prompt = HYDE_PROMPT.format(query=query)

    response = client.models.generate_content(
        model=config.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=256,
        ),
    )

    content = response.text or ""
    logger.info("Generated HyDE document", query_preview=query[:50], hyde_len=len(content))
    return content
