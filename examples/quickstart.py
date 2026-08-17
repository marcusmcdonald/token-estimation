"""Progressive, runnable introduction to the token-estimation package.

Run from the repository root with:

    uv run python examples/quickstart.py

The TokenManager API is the best starting point for most applications. The
lower-level estimator API is useful when you want to manage one estimator
directly or place it in a performance-sensitive loop.
"""

import tiktoken

from token_estimation import FastTokenEstimator, TokenManager, UnifiedTokenCounter


TEXT = "Hello, world! This is a test of the token estimation system."
ENCODING_NAME = "o200k_base"

# Calibration works best with text representative of the real prompts,
# documents, or code that the application will process. At least two non-empty
# samples are required; varied lengths produce a more useful fit.
CALIBRATION_SAMPLES = [
    "Short user query.",
    "This is a standard paragraph with typical English sentence structure, punctuation, and wording.",
    "Please summarize the following document and provide three actionable bullet points.",
    "Large language models process text by converting words and subwords into integer tokens.",
    "The quick brown fox jumps over the lazy dog repeatedly across multiple lines of text.",
    "Functions should do one thing, do it well, and do it only.",
    "API endpoints require authentication headers including bearer tokens and request signatures.",
]


def manager_workflow() -> None:
    """Use one manager to estimate and verify text for multiple models."""
    print("1. TokenManager: the recommended high-level API")
    manager = TokenManager()

    # These estimates use the manager's built-in baseline rates. Estimation is
    # O(1): it depends on character length rather than tokenizing the full text.
    for model_name in ("gpt-4o", "claude-3-5-sonnet", "llama-3-8b"):
        estimate = manager.estimate(TEXT, model_name)
        print(f"   {model_name}: {estimate} estimated tokens")

    print("\n2. Calibrate a model for your own text distribution")
    # calibrate_model both creates the estimator and registers it under the
    # model name. Future manager.estimate calls automatically use it.
    estimator = manager.calibrate_model(
        "custom-model",
        CALIBRATION_SAMPLES,
        encoding_name=ENCODING_NAME,
        safety_factor=1.10,
    )
    print(f"   Rate: {estimator.rate:.4f} tokens/character")
    print(f"   Intercept: {estimator.intercept:.4f} tokens")
    print(f"   Estimate: {manager.estimate(TEXT, 'custom-model')} tokens")
    print(
        "   Conservative upper bound: "
        f"{manager.estimate(TEXT, 'custom-model', safe_upper_bound=True)} tokens"
    )

    print("\n3. Check a context window")
    # The manager first uses the fast conservative estimate. Near the supplied
    # threshold, it falls back to the exact counter configured on the manager.
    result = manager.check_context_window(
        TEXT,
        model_name="gpt-4o-mini",
        max_context_limit=128_000,
        threshold_ratio=0.85,
    )
    print(f"   Fits: {result['fits']}")
    print(f"   Count method: {result['method']}")
    print(f"   Tokens used: {result['token_count']}")
    print(f"   Conservative margin remaining: {result['margin_remaining']}")


def exact_counting() -> None:
    """Count tokens exactly when speed is less important than precision."""
    print("\n4. Exact counting")
    counter = UnifiedTokenCounter()
    exact = counter.count(TEXT, model_name="gpt-4o-mini")
    print(f"   gpt-4o-mini: {exact} exact tokens")

    # UnifiedTokenCounter also supports Claude, Gemini, and LLaMA model names.
    # Those providers require their optional dependency and, where applicable,
    # API credentials or a locally downloaded tokenizer. For example:
    #
    # counter.count(TEXT, model_name="claude-3-5-sonnet-latest")
    # counter.count(TEXT, model_name="gemini-1.5-pro")
    # counter.count(TEXT, model_name="llama-3-8b")


def direct_estimator_workflow() -> None:
    """Calibrate and use a standalone estimator without TokenManager."""
    print("\n5. FastTokenEstimator: the lower-level API")
    estimator = FastTokenEstimator.calibrate(
        corpus=CALIBRATION_SAMPLES,
        encoding_name=ENCODING_NAME,
        safety_factor=1.10,
    )

    incoming_text = (
        "Calculate the estimate for this incoming request without running tiktoken."
    )
    point_estimate = estimator.estimate(incoming_text)
    upper_bound = estimator.estimate(incoming_text, safe_upper_bound=True)

    print(f"   Text length: {len(incoming_text)} characters")
    print(f"   Point estimate: {point_estimate} tokens")
    print(f"   Conservative upper bound: {upper_bound} tokens")

    # Tokenization is shown only to verify the fast estimate. In production,
    # keep estimator.estimate() in the hot path and tokenize only when needed.
    encoder = tiktoken.get_encoding(ENCODING_NAME)
    actual_count = len(encoder.encode(incoming_text))
    print(f"   Actual tiktoken count: {actual_count} tokens")

    # Supplying counter_fn is equivalent to encoding_name and also lets the
    # estimator calibrate against another provider or custom tokenizer.
    custom_counter_estimator = FastTokenEstimator.calibrate(
        corpus=CALIBRATION_SAMPLES,
        counter_fn=lambda text: len(encoder.encode(text)),
        safety_factor=1.10,
    )
    print(
        "   Estimate using a custom counter: "
        f"{custom_counter_estimator.estimate(incoming_text)} tokens"
    )


def main() -> None:
    manager_workflow()
    exact_counting()
    direct_estimator_workflow()


if __name__ == "__main__":
    main()
