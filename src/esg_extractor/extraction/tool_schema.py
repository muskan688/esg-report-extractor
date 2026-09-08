"""Provider-agnostic prompt text and JSON-schema-shaped tool definition.

Both the Anthropic and Gemini clients extract the same KPIs with the same
instructions and the same field-level schema; only the wire format for
"here's a function you can call" differs between the two APIs. Keeping the
shared parts here means adding a third provider is a matter of translating
this one schema/prompt pair into its tool-calling format, not re-deriving
what to ask for.
"""

from __future__ import annotations

from typing import Any

from esg_extractor.schema.metrics import METRIC_FIELDS

EXTRACTION_TOOL_NAME = "record_esg_metrics"

EXTRACTION_SYSTEM_PROMPT = """You are an information-extraction assistant for corporate sustainability \
(ESG/CSRD) reports. You will be given a slice of a company's sustainability report (page text \
and any tables, in markdown). Extract only the specific KPIs you are asked to record, and ONLY \
if they are explicitly stated in this text with a number - never estimate, infer, or carry over \
a figure from general knowledge. For each KPI you find, give the number and unit exactly as \
printed, a short verbatim quote (<= 25 words) containing that number so a human can verify it, \
and your confidence (0-1) that this is the correct, most current figure for the requested KPI \
(e.g. prefer the current reporting year's total over a prior-year comparison column, and prefer \
group-wide totals over one region/segment unless no total is given). If a KPI is not present in \
this text, omit it entirely - do not guess or fill in a placeholder."""

QA_SYSTEM_PROMPT = (
    "You answer questions about a company's sustainability report using ONLY the "
    "provided excerpts. Cite the page number(s) you used, e.g. '(p. 42)'. If the "
    "excerpts don't contain the answer, say so plainly instead of guessing."
)


def build_metric_properties(metric_keys: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """JSON-schema ``properties`` dict, one entry per requested KPI.

    Plain JSON Schema (lowercase ``type`` values); each provider client
    translates this into its own tool/function-declaration format.
    """
    props = {}
    for m in METRIC_FIELDS:
        if metric_keys is not None and m.key not in metric_keys:
            continue
        props[m.key] = {
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
    return props


def build_user_message(text: str, metric_keys: list[str] | None) -> str:
    wanted = ", ".join(metric_keys) if metric_keys else "all"
    return (
        f"Requested KPIs: {wanted}\n\n"
        f"--- REPORT TEXT ---\n{text}\n--- END REPORT TEXT ---\n\n"
        f"Call {EXTRACTION_TOOL_NAME} with whichever requested KPIs are explicitly present."
    )
