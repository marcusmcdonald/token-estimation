"""Unified token estimation manager with hybrid verification."""

from typing import Any

from .protocols import TokenEstimator
from .estimators import FastTokenEstimator
from .counters import UnifiedTokenCounter


class TokenManager:
    """Unified engine supporting multi-provider estimation and hybrid verification."""

    # Baseline default rates for major models if uncalibrated
    DEFAULT_RATES: dict[str, float] = {
        "gpt-4o": 0.22,
        "gpt-4o-mini": 0.22,
        "claude-3-5-sonnet": 0.23,
        "claude-3-haiku": 0.23,
        "gemini-1.5-pro": 0.24,
        "gemini-1.5-flash": 0.24,
        "llama-3-8b": 0.25,
    }

    def __init__(self, exact_counter: UnifiedTokenCounter | None = None) -> None:
        self.exact_counter = exact_counter or UnifiedTokenCounter()
        self.estimators: dict[str, TokenEstimator] = {}

    def _normalize_key(self, model_name: str) -> str:
        return model_name.lower().strip()

    def calibrate_model(
        self,
        model_name: str,
        corpus: list[str],
        estimator_cls: type[TokenEstimator] = FastTokenEstimator,
        **calibrate_kwargs: Any,
    ) -> TokenEstimator:
        """Calibrates and registers an estimator specifically for a target model."""
        key = self._normalize_key(model_name)
        estimator = estimator_cls.calibrate(corpus=corpus, **calibrate_kwargs)
        self.estimators[key] = estimator
        return estimator

    def estimate(
        self, text: str, model_name: str, *, safe_upper_bound: bool = False
    ) -> int:
        """Fast O(1) multi-model estimate."""
        key = self._normalize_key(model_name)
        estimator = self.estimators.get(key)

        if not estimator:
            matched_rate = next(
                (r for k, r in self.DEFAULT_RATES.items() if k in key), 0.25
            )
            estimator = FastTokenEstimator(rate=matched_rate, intercept=1.0)

        return estimator.estimate(text, safe_upper_bound=safe_upper_bound)

    def count_exact(self, text: str, model_name: str, **kwargs: Any) -> int:
        """Direct exact token count via API or local tokenizer."""
        return self.exact_counter.count(text, model_name=model_name, **kwargs)

    def check_context_window(
        self,
        text: str,
        model_name: str,
        max_context_limit: int,
        threshold_ratio: float = 0.85,
    ) -> dict[str, Any]:
        """Hybrid check: Uses fast upper-bound first; falls back to exact count near limits."""
        fast_point = self.estimate(text, model_name, safe_upper_bound=False)
        fast_upper = self.estimate(text, model_name, safe_upper_bound=True)
        threshold = int(max_context_limit * threshold_ratio)

        if fast_upper < threshold:
            return {
                "fits": True,
                "token_count": fast_point,
                "upper_bound_estimate": fast_upper,
                "method": "fast_estimate",
                "margin_remaining": max_context_limit - fast_upper,
            }

        exact_tokens = self.count_exact(text, model_name)
        return {
            "fits": exact_tokens <= max_context_limit,
            "token_count": exact_tokens,
            "method": "exact_count",
            "margin_remaining": max_context_limit - exact_tokens,
        }
