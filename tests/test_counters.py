"""Tests for UnifiedTokenCounter."""

import pytest

from src.token_estimation.counters import UnifiedTokenCounter


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


def test_count_claude_model_skip_if_missing():
    """Count tokens for Claude model - skipped if anthropic not installed."""
    try:
        counter = UnifiedTokenCounter()
        text = "Hello, world!"
        result = counter.count(text, "claude-3-5-sonnet-20240620")
        assert result > 0
    except ImportError:
        pytest.skip("anthropic package not installed")
