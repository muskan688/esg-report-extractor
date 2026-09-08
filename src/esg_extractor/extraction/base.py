"""Common interface every provider client (Anthropic, Gemini, ...) implements.

A ``typing.Protocol`` rather than an ABC: it lets call sites type-hint
against "something with these two methods" without importing a concrete
provider module (and its SDK) just for the type.
"""

from __future__ import annotations

from typing import Optional, Protocol


class BaseLLMClient(Protocol):
    model: str

    def extract_metrics(self, text: str, metric_keys: Optional[list[str]] = None) -> dict[str, dict]: ...

    def answer_question(self, question: str, context: str) -> str: ...
