"""Thin wrapper around the Anthropic API for schema-constrained extraction.

Uses tool-use (forced tool_choice) rather than "ask for JSON in a prompt" so
the model's output is validated against a JSON schema by the API itself
instead of being parsed out of free text.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from esg_extractor.schema.metrics import METRIC_FIELDS

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

_EXTRACTION_TOOL_NAME = "record_esg_metrics"

_SYSTEM_PROMPT = """You are an information-extraction assistant for corporate sustainability \
(ESG/CSRD) reports. You will be given a slice of a company's sustainability report (page text \
and any tables, in markdown). Extract only the specific KPIs you are asked to record, and ONLY \
if they are explicitly stated in this text with a number - never estimate, infer, or carry over \
a figure from general knowledge. For each KPI you find, give the number and unit exactly as \
printed, a short verbatim quote (<= 25 words) containing that number so a human can verify it, \
and your confidence (0-1) that this is the correct, most current figure for the requested KPI \
(e.g. prefer the current reporting year's total over a prior-year comparison column, and prefer \
group-wide totals over one region/segment unless no total is given). If a KPI is not present in \
this text, omit it entirely - do not guess or fill in a placeholder."""


def _build_extraction_schema() -> dict[str, Any]:
    metric_props = {}
    for m in METRIC_FIELDS:
        metric_props[m.key] = {
            "type": "object",
            "description": f"{m.description} (report as printed; canonical unit is {m.canonical_unit})",
            "properties": {
                "value": {"type": "number", "description": "The numeric value, exactly as printed."},
                "unit": {"type": "string", "description": "The unit exactly as printed, e.g. 'kt CO2e', 'GWh', '%'."},
                "quote": {"type": "string", "description": "Short verbatim source quote containing this figure."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["value", "unit", "quote", "confidence"],
        }
    return {
        "name": _EXTRACTION_TOOL_NAME,
        "description": "Record whichever of the requested ESG KPIs are explicitly present in the given text.",
        "input_schema": {
            "type": "object",
            "properties": metric_props,
            "additionalProperties": False,
        },
    }


class LLMClient:
    """Wraps the Anthropic client; instantiate once and reuse across chunks."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)
        self._tool_schema = _build_extraction_schema()

    def extract_metrics(self, text: str, metric_keys: Optional[list[str]] = None) -> dict[str, dict]:
        """Run one extraction call over a chunk of report text.

        ``metric_keys`` restricts which KPIs are requested (used when a
        chunk was only pulled in as a candidate for a subset of metrics),
        keeping the prompt/tool schema focused and cutting cost.
        Returns ``{metric_key: {value, unit, quote, confidence}}`` for
        whichever of the requested metrics the model actually found.
        """
        tool = self._tool_schema
        if metric_keys is not None:
            tool = dict(tool)
            tool["input_schema"] = dict(tool["input_schema"])
            props = tool["input_schema"]["properties"]
            tool["input_schema"]["properties"] = {k: v for k, v in props.items() if k in metric_keys}

        wanted = ", ".join(metric_keys) if metric_keys else "all"
        user_msg = (
            f"Requested KPIs: {wanted}\n\n"
            f"--- REPORT TEXT ---\n{text}\n--- END REPORT TEXT ---\n\n"
            f"Call {_EXTRACTION_TOOL_NAME} with whichever requested KPIs are explicitly present."
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": _EXTRACTION_TOOL_NAME},
            messages=[{"role": "user", "content": user_msg}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == _EXTRACTION_TOOL_NAME:
                return dict(block.input)
        return {}

    def answer_question(self, question: str, context: str) -> str:
        """RAG answer-generation call: answer strictly from the given context."""
        system = (
            "You answer questions about a company's sustainability report using ONLY the "
            "provided excerpts. Cite the page number(s) you used, e.g. '(p. 42)'. If the "
            "excerpts don't contain the answer, say so plainly instead of guessing."
        )
        user_msg = f"Excerpts:\n{context}\n\nQuestion: {question}"
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in response.content if b.type == "text")
