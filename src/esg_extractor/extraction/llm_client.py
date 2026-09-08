"""Anthropic (Claude) provider client.

Uses tool-use (forced tool_choice) rather than "ask for JSON in a prompt" so
the model's output is validated against a JSON schema by the API itself
instead of being parsed out of free text.
"""

from __future__ import annotations

import os
from typing import Optional

from esg_extractor.extraction.tool_schema import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_TOOL_NAME,
    QA_SYSTEM_PROMPT,
    build_metric_properties,
    build_user_message,
)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


class AnthropicClient:
    """Wraps the Anthropic client; instantiate once and reuse across chunks."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def _tool(self, metric_keys: Optional[list[str]]) -> dict:
        return {
            "name": EXTRACTION_TOOL_NAME,
            "description": "Record whichever of the requested ESG KPIs are explicitly present in the given text.",
            "input_schema": {
                "type": "object",
                "properties": build_metric_properties(metric_keys),
                "additionalProperties": False,
            },
        }

    def extract_metrics(self, text: str, metric_keys: Optional[list[str]] = None) -> dict[str, dict]:
        """Run one extraction call over a chunk of report text.

        ``metric_keys`` restricts which KPIs are requested (used when a
        chunk was only pulled in as a candidate for a subset of metrics),
        keeping the prompt/tool schema focused and cutting cost.
        Returns ``{metric_key: {value, unit, quote, confidence}}`` for
        whichever of the requested metrics the model actually found.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=EXTRACTION_SYSTEM_PROMPT,
            tools=[self._tool(metric_keys)],
            tool_choice={"type": "tool", "name": EXTRACTION_TOOL_NAME},
            messages=[{"role": "user", "content": build_user_message(text, metric_keys)}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == EXTRACTION_TOOL_NAME:
                return dict(block.input)
        return {}

    def answer_question(self, question: str, context: str) -> str:
        """RAG answer-generation call: answer strictly from the given context."""
        user_msg = f"Excerpts:\n{context}\n\nQuestion: {question}"
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=QA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


# Backwards-compatible alias; prefer esg_extractor.extraction.provider.get_llm_client()
LLMClient = AnthropicClient
