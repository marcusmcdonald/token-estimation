"""Tests for token estimators: FastTokenEstimator and RobustFastTokenEstimator."""

import pytest

from token_estimation.estimators import FastTokenEstimator, RobustFastTokenEstimator


def test_fast_estimator_default_rate():
    """FastTokenEstimator with default rate estimates positive token count."""
    estimator = FastTokenEstimator()
    text = "Hello, world!"
    result = estimator.estimate(text)
    assert result > 0


def test_fast_estimator_custom_rate():
    """FastTokenEstimator with custom rate."""
    estimator = FastTokenEstimator(rate=0.5, intercept=2.0)
    text = "Hello"
    result = estimator.estimate(text)
    assert result > 0


def test_fast_estimator_safe_upper_bound():
    """FastTokenEstimator with safe_upper_bound flag."""
    estimator = FastTokenEstimator(rate=0.25, intercept=1.0, safety_factor=1.1)
    text = "Hello, world!"
    result = estimator.estimate(text, safe_upper_bound=True)
    assert result >= estimator.estimate(text, safe_upper_bound=False)


def test_fast_estimator_empty_text():
    """FastTokenEstimator returns 0 for empty text."""
    estimator = FastTokenEstimator()
    result = estimator.estimate("")
    assert result == 0


def test_fast_estimator_calibrate():
    """FastTokenEstimator.calibrate creates estimator from corpus."""
    corpus = [
        "This is a short sentence.",
        "This is a much longer sentence that will definitely have more tokens.",
        "Mid length text here.",
    ]
    estimator = FastTokenEstimator.calibrate(corpus)
    assert estimator.rate > 0
    assert estimator.intercept >= 0


def test_robust_estimator_default():
    """RobustFastTokenEstimator with default parameters estimates positive tokens."""
    estimator = RobustFastTokenEstimator()
    text = "Hello, world!"
    result = estimator.estimate(text)
    assert result > 0


def test_robust_estimator_safe_upper_bound():
    """RobustFastTokenEstimator with safe_upper_bound flag."""
    estimator = RobustFastTokenEstimator(rate=0.25, intercept=1.0)
    text = "Hello, world!"
    result = estimator.estimate(text, safe_upper_bound=True)
    assert result >= estimator.estimate(text, safe_upper_bound=False)


def test_robust_estimator_empty_text():
    """RobustFastTokenEstimator returns 0 for empty text."""
    estimator = RobustFastTokenEstimator()
    result = estimator.estimate("")
    assert result == 0


def test_robust_estimator_calibrate():
    """RobustFastTokenEstimator.calibrate creates estimator from corpus and counter."""
    corpus = [
        "This is a short sentence.",
        "This is a much longer sentence that will definitely have more tokens.",
        "Mid length text here.",
    ]
    estimator = RobustFastTokenEstimator.calibrate(corpus, lambda t: len(t.split()))
    assert estimator.rate > 0
    assert estimator.intercept >= 0
    assert estimator.fixed_buffer >= 1
