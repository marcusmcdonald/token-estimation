# Token Estimation

Fast and robust token estimation for LLMs. Provides two estimators:

- **FastTokenEstimator** - OLS linear regression with tiktoken
- **RobustFastTokenEstimator** - Theil-Sen robust regression with custom counter functions

## Installation

```bash
pip install -e .
# With optional dependencies:
pip install -e .[anthropic,gemini,llama]
```

## Quick Start

```python
from token_estimation import FastTokenEstimator, RobustFastTokenEstimator, TokenManager

# Fast OLS estimator with tiktoken
estimator = FastTokenEstimator.calibrate(
    corpus=["sample text", "more samples"],
    encoding_name="o200k_base"
)
print(estimator.estimate("Hello world"))

# Robust Theil-Sen estimator with custom counter
estimator = RobustFastTokenEstimator.calibrate(
    corpus=["sample text", "more samples"],
    counter_fn=lambda t: len(t) // 4,
    target_percentile=95.0
)

# Unified TokenManager for multi-model support
tm = TokenManager()
tm.calibrate_model("gpt-4o", corpus, encoding_name="o200k_base")
print(tm.estimate("Hello world", "gpt-4o"))
```

## Estimators

### FastTokenEstimator
- Uses Ordinary Least Squares (OLS) regression
- Requires `tiktoken` for calibration
- Multiplicative safety factor for upper bounds
- Best for: OpenAI models with known encodings

### RobustFastTokenEstimator
- Uses Theil-Sen median-of-slopes regression
- Accepts any token counting function (API, local tokenizer, etc.)
- Additive safety buffer + fixed buffer for upper bounds
- Best for: Anthropic, Gemini, LLaMA, or custom tokenizers

## TokenManager
Unified interface supporting:
- Multi-provider estimation (OpenAI, Anthropic, Gemini, LLaMA)
- Hybrid verification (fast estimate → exact count near limits)
- Calibration per model with custom corpora

## License

MIT