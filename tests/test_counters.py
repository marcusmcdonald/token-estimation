"""Tests for UnifiedTokenCounter."""

from types import SimpleNamespace

from token_estimation.counters import UnifiedTokenCounter


def test_count_empty_text():
    """Empty string returns 0 tokens."""
    counter = UnifiedTokenCounter()
    result = counter.count("", "gpt-4o-mini")
    assert result == 0


def test_count_openai_model_tiktoken():
    """Count tokens for OpenAI model using tiktoken."""
    counter = UnifiedTokenCounter()
    text = "Hello, world! This is a test."
    result = counter.count(text, "gpt-4o-mini")
    assert result > 0


def test_count_claude_model():
    """Count tokens for a Claude model through the configured client."""
    messages = SimpleNamespace(
        count_tokens=lambda **kwargs: SimpleNamespace(input_tokens=4)
    )
    client = SimpleNamespace(messages=messages)
    counter = UnifiedTokenCounter(anthropic_client=client)

    result = counter.count("Hello, world!", "claude-3-5-sonnet-20240620")

    assert result == 4
