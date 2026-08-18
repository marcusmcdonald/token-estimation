"""Token Estimators: Fast OLS and Robust Theil-Sen implementations."""

import math
import statistics
from typing import Any, Callable

from .protocols import TokenEstimator

class CharCountEncoding:
    """Encoding that returns one token per four characters."""

    def encode(self, text: str, **kwargs) -> list[int]:
        return list(map(ord, text[::4]))


class FastTokenEstimator:
    """Fast O(1) character-length linear estimator using OLS regression."""

    def __init__(
        self,
        rate: float = 0.25,
        intercept: float = 1.0,
        safety_factor: float = 1.05,
    ):
        self.rate = rate
        self.intercept = intercept
        self.safety_factor = safety_factor

    @classmethod
    def calibrate(
        cls,
        corpus: list[str],
        counter_fn: Callable[[str], int] | None = None,
        safety_factor: float = 1.10,
        **kwargs: Any,
    ) -> "FastTokenEstimator":
        """Calculates rate (slope) and intercept via linear regression.

        Args:
            corpus: List of text samples for calibration.
            counter_fn: Optional custom token counting function. If not provided,
                uses CharCountEncoding as the default.
            safety_factor: Multiplicative safety factor for upper bound estimates.
        """

        if counter_fn is None:
            enc = CharCountEncoding()
            counter_fn = lambda text: len(enc.encode(text))

        x = [len(text) for text in corpus if len(text) > 0]
        y = [counter_fn(text) for text in corpus if len(text) > 0]

        if not x:
            raise ValueError("Sample corpus must contain non-empty text samples.")
        if len(x) < 2:
            raise ValueError(
                "At least 2 non-empty text samples are required for calibration."
            )

        # Ordinary Linear Regression (OLS): y = slope * x + intercept
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else (mean_y / mean_x)
        intercept = mean_y - (slope * mean_x)

        return cls(
            rate=slope,
            intercept=max(0.0, intercept),
            safety_factor=safety_factor,
        )

    def estimate(self, text: str, *, safe_upper_bound: bool = False) -> int:
        """Fast O(1) estimate using only len(text)."""
        char_len = len(text)
        if char_len == 0:
            return 0

        raw_estimate = (char_len * self.rate) + self.intercept

        if safe_upper_bound:
            raw_estimate *= self.safety_factor

        return math.ceil(raw_estimate)


class RobustFastTokenEstimator:
    """Robust O(1) character-length linear estimator using Theil-Sen regression."""

    def __init__(
        self,
        rate: float = 0.25,
        intercept: float = 1.0,
        safety_buffer_rate: float = 0.0,
        fixed_buffer: int = 1,
    ) -> None:
        self.rate = rate
        self.intercept = intercept
        self.safety_buffer_rate = safety_buffer_rate
        self.fixed_buffer = fixed_buffer

    @classmethod
    def calibrate(
        cls,
        corpus: list[str],
        counter_fn: Callable[[str], int],
        target_percentile: float = 95.0,
        **kwargs: Any,
    ) -> "RobustFastTokenEstimator":
        """Fits rate and intercept against a custom counting function using Theil-Sen.

        Args:
            corpus: List of text samples for calibration.
            counter_fn: Token counting function (API, local tokenizer, etc.).
            target_percentile: Percentile for residual-based fixed buffer (default 95).
        """
        data = [(len(t), counter_fn(t)) for t in corpus if len(t) > 0]

        if len(data) < 2:
            raise ValueError(
                "At least 2 non-empty text samples are required for calibration."
            )

        # Compute pairwise slopes
        pairwise_slopes = []
        n = len(data)
        for i in range(n):
            x1, y1 = data[i]
            for j in range(i + 1, n):
                x2, y2 = data[j]
                if x1 != x2:
                    pairwise_slopes.append((y2 - y1) / (x2 - x1))

        if not pairwise_slopes:
            rate = statistics.median(y / x for x, y in data)
        else:
            rate = statistics.median(pairwise_slopes)

        intercept_candidates = [y - (rate * x) for x, y in data]
        intercept = statistics.median(intercept_candidates)

        # Residuals for conservative upper-bound buffer
        residuals = [y - (rate * x + intercept) for x, y in data if x < 500]
        if not residuals:
            residuals = [y - (rate * x + intercept) for x, y in data]

        residuals.sort()
        percentile_residual = statistics.quantiles(residuals, n=100)[
            int(target_percentile) - 1
        ]
        percentile_residual = max(0.0, percentile_residual)

        return cls(
            rate=rate,
            intercept=max(0.0, intercept),
            safety_buffer_rate=0.05 * rate,
            fixed_buffer=math.ceil(percentile_residual) + 1,
        )

    def estimate(self, text: str, *, safe_upper_bound: bool = False) -> int:
        """Instant O(1) estimate based purely on len(text)."""
        char_len = len(text)
        if char_len == 0:
            return 0

        if not safe_upper_bound:
            return max(1, math.ceil((char_len * self.rate) + self.intercept))

        effective_rate = self.rate + self.safety_buffer_rate
        return max(
            1,
            math.ceil((char_len * effective_rate) + self.intercept + self.fixed_buffer),
        )
