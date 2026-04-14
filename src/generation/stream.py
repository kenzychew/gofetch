"""Google Gemini streaming wrapper for SSE generation with retry logic."""

from collections.abc import AsyncIterator

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import GenerationConfig
from src.exceptions import StreamError
from src.logging import get_logger

logger = get_logger(__name__)

RETRY_DECORATOR = retry(
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)


def _build_contents(
    messages: list[dict[str, str]],
) -> tuple[str, list[types.Content]]:
    """Convert message list into Gemini system instruction and contents.

    Args:
        messages: Message list with optional system message first.

    Returns:
        Tuple of (system instruction string, list of Content objects).
    """
    system = ""
    contents: list[types.Content] = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    return system, contents


@RETRY_DECORATOR
async def stream_completion(
    messages: list[dict[str, str]],
    client: genai.Client,
    config: GenerationConfig,
) -> AsyncIterator[str]:
    """Stream a chat completion response from Gemini.

    Yields text chunks as they arrive from the API. The caller
    can forward these chunks as SSE events to the client.
    Retries up to 3 times with exponential backoff.

    Args:
        messages: The message list for the API call.
        client: Google GenAI client instance.
        config: Generation configuration with model settings.

    Yields:
        Text chunks from the streaming response.

    Raises:
        StreamError: If the streaming request fails after retries.
    """
    system, contents = _build_contents(messages)

    try:
        response = client.models.generate_content_stream(
            model=config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system if system else None,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as exc:
        raise StreamError(f"Gemini streaming failed: {exc}") from exc


@RETRY_DECORATOR
async def generate_completion(
    messages: list[dict[str, str]],
    client: genai.Client,
    config: GenerationConfig,
) -> str:
    """Generate a non-streaming chat completion response.

    Used for HyDE, query decomposition, and evaluation where
    streaming is not needed. Retries up to 3 times.

    Args:
        messages: The message list for the API call.
        client: Google GenAI client instance.
        config: Generation configuration with model settings.

    Returns:
        The complete response text.

    Raises:
        StreamError: If the API request fails after retries.
    """
    system, contents = _build_contents(messages)

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system if system else None,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            ),
        )
        return response.text or ""
    except Exception as exc:
        raise StreamError(f"Gemini completion failed: {exc}") from exc
