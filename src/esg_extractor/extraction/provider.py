"""Picks which LLM provider client to instantiate.

Every call site in this project (``MetricExtractor``, ``ReportQA``,
``pipeline.process_report``, the CLI, the Streamlit app) goes through
``get_llm_client()`` instead of importing a concrete provider class, so
switching providers is a single env var (``LLM_PROVIDER``) rather than a
code change.
"""

from __future__ import annotations

import os
from typing import Optional

from esg_extractor.extraction.base import BaseLLMClient

_PROVIDERS = ("anthropic", "gemini")


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).strip().lower()
    if provider == "anthropic":
        from esg_extractor.extraction.llm_client import AnthropicClient

        return AnthropicClient()
    if provider == "gemini":
        from esg_extractor.extraction.gemini_client import GeminiClient

        return GeminiClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Supported: {', '.join(_PROVIDERS)}")
