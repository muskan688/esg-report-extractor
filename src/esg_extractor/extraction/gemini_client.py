"""Google Gemini provider client.

Uses forced function-calling (``FunctionCallingConfigMode.ANY`` restricted to
one allowed function) as the equivalent of Anthropic's forced tool_choice,
and ``FunctionDeclaration.parameters_json_schema`` to pass the shared plain
JSON Schema straight through - no schema-format translation needed, unlike
providers that only accept the OpenAPI-subset ``Schema`` object.
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

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


class GeminiClient:
    """Wraps the Gemini client; instantiate once and reuse across chunks."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        from google import genai

        self.model = model
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=resolved_key)

    def _tool(self, metric_keys: Optional[list[str]]):
        from google.genai import types

        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=EXTRACTION_TOOL_NAME,
                    description="Record whichever of the requested ESG KPIs are explicitly present in the given text.",
                    parameters_json_schema={
                        "type": "object",
                        "properties": build_metric_properties(metric_keys),
                        "additionalProperties": False,
                    },
                )
            ]
        )

    def extract_metrics(self, text: str, metric_keys: Optional[list[str]] = None) -> dict[str, dict]:
        """Run one extraction call over a chunk of report text.

        Mirrors ``AnthropicClient.extract_metrics``: ``metric_keys``
        restricts which KPIs are requested, and only KPIs explicitly found
        in ``text`` come back populated.
        """
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=build_user_message(text, metric_keys),
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                tools=[self._tool(metric_keys)],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[EXTRACTION_TOOL_NAME],
                    )
                ),
            ),
        )
        candidates = response.candidates or []
        if not candidates or not candidates[0].content or not candidates[0].content.parts:
            return {}
        for part in candidates[0].content.parts:
            if part.function_call and part.function_call.name == EXTRACTION_TOOL_NAME:
                return dict(part.function_call.args or {})
        return {}

    def answer_question(self, question: str, context: str) -> str:
        """RAG answer-generation call: answer strictly from the given context."""
        from google.genai import types

        user_msg = f"Excerpts:\n{context}\n\nQuestion: {question}"
        response = self._client.models.generate_content(
            model=self.model,
            contents=user_msg,
            config=types.GenerateContentConfig(system_instruction=QA_SYSTEM_PROMPT),
        )
        return response.text or ""
