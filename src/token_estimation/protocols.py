"""Token Estimator Protocol - Public interface contract."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenEstimator(Protocol):
    """Protocol for token estimators supporting fast O(1) estimation."""

    rate: float
    intercept: float

    def estimate(self, text: str, *, safe_upper_bound: bool = False) -> int: ...

    @classmethod
    def calibrate(cls, corpus: list[str], **kwargs: Any) -> "TokenEstimator": ...
