# Token Estimation

Fast token estimates for LLM applications, with exact counting available when
precision matters. The package supports a practical hybrid workflow:

1. Estimate from character length in O(1) time.
2. Use a conservative upper bound when checking limits.
3. Fall back to exact tokenization only near the context-window boundary.

`TokenManager` is the recommended high-level API. Standalone estimators and
exact counters are also available when you need more control.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Set up the project

This repository is managed with uv and includes a committed `uv.lock` file.
From the repository root, install the locked base dependencies with:

```bash
uv sync
```

Install test tooling as well:

```bash
uv sync --extra dev
```

Provider integrations are optional. Install one provider extra or all of them:

```bash
uv sync --extra anthropic
uv sync --extra gemini
uv sync --extra llama
uv sync --extra all
```

To install every provider integration and the development tools:

```bash
uv sync --extra all --extra dev
```

## Quick start

Create a manager and estimate immediately using its built-in baseline rates:

```python
from token_estimation import TokenManager

manager = TokenManager()
text = "Hello, world! This is a test of the token estimation system."

for model_name in ("gpt-4o", "claude-3-5-sonnet", "llama-3-8b"):
    tokens = manager.estimate(text, model_name)
    print(f"{model_name}: {tokens} estimated tokens")
```

These estimates use only the length of the input, so they are suitable for hot
paths where full tokenization would be unnecessarily expensive.

The complete, runnable walkthrough is in
[`examples/quickstart.py`](examples/quickstart.py). Run it with:

```bash
uv run python examples/quickstart.py
```

## Calibrate for your workload

Built-in rates are convenient defaults. For better estimates, calibrate against
text representative of the prompts, documents, or code your application will
process. Use at least two non-empty samples with varied lengths.

```python
from token_estimation import TokenManager

samples = [
    "Short user query.",
    "A typical paragraph from the application with representative wording.",
    "A longer document sample containing the kind of content users normally submit.",
]

manager = TokenManager()
estimator = manager.calibrate_model(
    "custom-model",
    samples,
    safety_factor=1.10,
)

point_estimate = manager.estimate("New incoming text", "custom-model")
upper_bound = manager.estimate(
    "New incoming text",
    "custom-model",
    safe_upper_bound=True,
)
```

`calibrate_model` creates the estimator and registers it on that manager
instance. Later calls to `manager.estimate` automatically use the calibrated
estimator for the normalized model name.

## Check a context window

`check_context_window` starts with a fast conservative estimate. If the text is
near the configured threshold, the manager switches to its exact counter.

```python
result = manager.check_context_window(
    text,
    model_name="gpt-4o-mini",
    max_context_limit=128_000,
    threshold_ratio=0.85,
)

print(result["fits"])
print(result["token_count"])
print(result["method"])  # "fast_estimate" or "exact_count"
print(result["margin_remaining"])
```

The context limit is supplied by your application; the package does not keep a
model context-window registry.

## Count tokens exactly

Use `UnifiedTokenCounter` directly when an exact count is required:

```python
from token_estimation import UnifiedTokenCounter

counter = UnifiedTokenCounter()
tokens = counter.count("Hello, world!", model_name="gpt-4o-mini")
```

OpenAI-compatible names use local `tiktoken` counting. Other model families
require their optional dependencies and, where applicable, credentials or a
locally downloaded tokenizer:

| Model family | Extra | Counting method |
| --- | --- | --- |
| OpenAI | Included | Local `tiktoken` encoding |
| Anthropic / Claude | `anthropic` | Anthropic API |
| Google Gemini | `gemini` | Gemini API or local tokenizer |
| Meta LLaMA | `llama` | Local Transformers tokenizer |

## Use an estimator directly

Use the lower-level API when you want to own one estimator directly or place it
in a performance-sensitive loop:

```python
from token_estimation import FastTokenEstimator

estimator = FastTokenEstimator.calibrate(
    corpus=samples,
    safety_factor=1.10,
)

tokens = estimator.estimate("New incoming text")
upper_bound = estimator.estimate("New incoming text", safe_upper_bound=True)
```

`FastTokenEstimator` uses ordinary least-squares regression. By default it
calibrates against a lightweight character-based counter. Supply a custom
counting function to calibrate against a tokenizer or another token-counting
service:

```python
estimator = FastTokenEstimator.calibrate(
    corpus=samples,
    counter_fn=my_token_counter,
    safety_factor=1.10,
)
```

For calibration data containing outliers, use
`RobustFastTokenEstimator`. It uses Theil-Sen regression and requires a custom
counting function:

```python
from token_estimation import RobustFastTokenEstimator

estimator = RobustFastTokenEstimator.calibrate(
    corpus=samples,
    counter_fn=my_token_counter,
    target_percentile=95.0,
)
```

## Development

Run the test suite through the managed environment:

```bash
uv run pytest
```

When dependency metadata changes, update and commit both `pyproject.toml` and
`uv.lock` so development and CI resolve the same environment.

## License

MIT
