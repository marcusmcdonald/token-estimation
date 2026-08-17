"""Tests for TokenManager - unified token estimation manager."""

import pytest

from token_estimation import TokenManager, UnifiedTokenCounter


def test_manager_estimate_default_model():
    """TokenManager.estimate uses default rate when no calibrated estimator exists."""
    manager = TokenManager()
    text = "Hello, world!"
    result = manager.estimate(text, "gpt-4o")
    assert result > 0


def test_manager_estimate_calibrated_model():
    """TokenManager.estimate uses calibrated estimator after calibrate_model."""
    manager = TokenManager()
    corpus = [
        "This is a short sentence.",
        "This is a much longer sentence that will definitely have more tokens.",
        "Mid length text here.",
    ]
    manager.calibrate_model("gpt-4o", corpus)
    text = "Hello, world!"
    result = manager.estimate(text, "gpt-4o")
    assert result > 0


def test_manager_calibrate_model_registers_estimator():
    """TokenManager.calibrate_model registers estimator by normalized key."""
    manager = TokenManager()
    corpus = ["short text", "longer text here"]
    estimator = manager.calibrate_model("GPT-4o", corpus)
    assert "gpt-4o" in manager.estimators
    assert estimator.rate > 0


def test_manager_default_rates_fallback():
    """TokenManager uses default rates when no estimator is registered."""
    manager = TokenManager()
    # Should not raise - falls back to default rate
    result = manager.estimate("test", "unknown-model")
    assert result > 0


def test_manager_count_exact():
    """TokenManager.count_exact delegates to UnifiedTokenCounter."""
    manager = TokenManager()
    text = "Hello, world!"
    result = manager.count_exact(text, "gpt-4o-mini")
    assert result > 0


def test_manager_check_context_window_fits():
    """TokenManager.check_context_window returns fits=True when under limit."""
    manager = TokenManager()
    text = "Hello, world!"
    result = manager.check_context_window(text, "gpt-4o", max_context_limit=1000)
    assert result["fits"] is True
    assert result["method"] == "fast_estimate"


def test_manager_check_context_window_exact_fallback():
    """TokenManager.check_context_window falls back to exact count near limits."""
    manager = TokenManager()
    # Use a very long text to trigger exact count fallback
    text = "A " * 500  # ~1000 chars
    result = manager.check_context_window(text, "gpt-4o-mini", max_context_limit=1000)
    assert "fits" in result
    assert "method" in result
