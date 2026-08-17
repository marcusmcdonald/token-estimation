"""Exact token counting across major model providers."""

import logging
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore


logger = logging.getLogger(__name__)


class UnifiedTokenCounter:
    """Exact token counting across major model providers."""

    def __init__(
        self,
        anthropic_client: Any | None = None,
        gemini_client: Any | None = None,
        llama_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ) -> None:
        self._anthropic_client = anthropic_client
        self._gemini_client = gemini_client
        self._llama_model_id = llama_model_id

        # Lazy-loaded tokenizer instances
        self._llama_tokenizer: Any | None = None
        self._gemini_local_tokenizer: Any | None = None

    @property
    def anthropic_client(self) -> Any:
        """Get or initialize Anthropic client."""
        if self._anthropic_client is None:
            if anthropic is None:
                raise ImportError("anthropic package is required for Claude counting.")
            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    @property
    def gemini_client(self) -> Any:
        """Get or initialize Gemini client."""
        if self._gemini_client is None:
            if genai is None:
                raise ImportError(
                    "google-genai package is required for Gemini API counting."
                )
            self._gemini_client = genai.Client()
        return self._gemini_client

    @property
    def llama_tokenizer(self) -> Any:
        """Get or initialize LLaMA tokenizer."""
        if self._llama_tokenizer is None:
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is required for local LLaMA tokenization."
                )
            self._llama_tokenizer = AutoTokenizer.from_pretrained(self._llama_model_id)
        return self._llama_tokenizer

    def count(
        self, text: str, model_name: str, *, use_local_gemini: bool = False
    ) -> int:
        """Count exact tokens for a given text and model identifier."""
        if not text:
            return 0

        model_lower = model_name.lower()

        # Google Gemini Models
        if "gemini" in model_lower:
            if use_local_gemini:
                if genai is None:
                    raise ImportError("google-genai package is required.")
                if self._gemini_local_tokenizer is None:
                    self._gemini_local_tokenizer = genai.LocalTokenizer(
                        model_name=model_name
                    )
                return self._gemini_local_tokenizer.count_tokens(text).total_tokens  # type: ignore
            res = self.gemini_client.models.count_tokens(
                model=model_name, contents=text
            )
            return res.total_tokens

        # Anthropic / Claude Models
        if "claude" in model_lower:
            res = self.anthropic_client.messages.count_tokens(
                model=model_name,
                messages=[{"role": "user", "content": text}],
            )
            return res.input_tokens

        # Meta LLaMA Models
        if "llama" in model_lower:
            return len(self.llama_tokenizer.encode(text))

        # OpenAI / Tiktoken Fallback
        if tiktoken is None:
            raise ImportError("tiktoken package is required for OpenAI models.")

        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(
                "Model '%s' not found in tiktoken. Falling back to o200k_base.",
                model_name,
            )
            enc = tiktoken.get_encoding("o200k_base")

        return len(enc.encode(text))
