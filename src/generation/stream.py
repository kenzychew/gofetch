"""Anthropic streaming wrapper for SSE generation with retry logic."""

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import GenerationConfig
from src.exceptions import StreamError
from src.logging import get_logger

logger = get_logger(__name__)

RETRY_DECORATOR = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)


def _split_system_and_user(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Split a message list into system prompt and user messages.

    The Anthropic API takes the system prompt as a separate parameter,
    not as a message with role 'system'.

    Args:
        messages: Message list with optional system message first.

    Returns:
        Tuple of (system prompt string, remaining messages).
    """
    system = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            user_messages.append(msg)
    return system, user_messages


@RETRY_DECORATOR
async def stream_completion(
    messages: list[dict[str, str]],
    client: AsyncAnthropic,
    config: GenerationConfig,
) -> AsyncIterator[str]:
    """Stream a chat completion response from Anthropic.

    Yields text chunks as they arrive from the API. The caller
    can forward these chunks as SSE events to the client.
    Retries up to 3 times on rate limit errors with exponential backoff.

    Args:
        messages: The message list for the API call.
        client: Async Anthropic client instance.
        config: Generation configuration with model settings.

    Yields:
        Text chunks from the streaming response.

    Raises:
        StreamError: If the streaming request fails after retries.
    """
    system, user_messages = _split_system_and_user(messages)

    try:
        async with client.messages.stream(
            model=config.model,
            system=system,
            messages=user_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    except RateLimitError:
        raise
    except Exception as exc:
        raise StreamError(f"Anthropic streaming failed: {exc}") from exc


@RETRY_DECORATOR
async def generate_completion(
    messages: list[dict[str, str]],
    client: AsyncAnthropic,
    config: GenerationConfig,
) -> str:
    """Generate a non-streaming chat completion response.

    Used for HyDE, query decomposition, and evaluation where
    streaming is not needed. Retries up to 3 times on rate limit errors.

    Args:
        messages: The message list for the API call.
        client: Async Anthropic client instance.
        config: Generation configuration with model settings.

    Returns:
        The complete response text.

    Raises:
        StreamError: If the API request fails after retries.
    """
    system, user_messages = _split_system_and_user(messages)

    try:
        response = await client.messages.create(
            model=config.model,
            system=system,
            messages=user_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return response.content[0].text if response.content else ""
    except RateLimitError:
        raise
    except Exception as exc:
        raise StreamError(f"Anthropic completion failed: {exc}") from exc
