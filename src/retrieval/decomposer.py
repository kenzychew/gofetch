"""Query decomposition for multi-part questions."""

import json

from google import genai
from google.genai import types

from src.config import GenerationConfig
from src.logging import get_logger

logger = get_logger(__name__)

DECOMPOSITION_PROMPT = (
    "You are a search query optimizer. Analyze the following question and determine "
    "if it contains multiple distinct sub-questions that should be searched separately.\n\n"
    "If the question is simple and single-topic, return a JSON array with just the "
    "original question.\n"
    "If the question is complex or multi-part, break it into 2-4 focused sub-queries "
    "that can each be searched independently.\n\n"
    "Return ONLY a JSON array of strings. No explanation.\n\n"
    "Question: {query}\n\n"
    "Sub-queries:"
)


async def decompose_query(
    query: str,
    client: genai.Client,
    config: GenerationConfig,
) -> list[str]:
    """Decompose a complex query into simpler sub-queries.

    Detects multi-part questions (eg comparisons, multi-hop) and splits
    them into focused sub-queries for separate retrieval passes. Results
    from all sub-queries are merged before fusion.

    Args:
        query: The user's search query.
        client: Google GenAI client instance.
        config: Generation configuration with model settings.

    Returns:
        List of sub-queries (may be a single-element list for simple queries).
    """
    prompt = DECOMPOSITION_PROMPT.format(query=query)

    response = client.models.generate_content(
        model=config.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
        ),
    )

    content = response.text or "[]"

    try:
        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list) or not sub_queries:
            logger.warning("Decomposition returned invalid format, using original query")
            return [query]
        sub_queries = [str(q) for q in sub_queries]
    except json.JSONDecodeError:
        logger.warning("Failed to parse decomposition response, using original query")
        return [query]

    logger.info(
        "Decomposed query",
        original=query[:50],
        sub_queries=len(sub_queries),
    )
    return sub_queries
