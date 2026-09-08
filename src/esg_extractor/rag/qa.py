"""Retrieval-augmented QA over ingested sustainability reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from esg_extractor.extraction.base import BaseLLMClient
from esg_extractor.extraction.provider import get_llm_client
from esg_extractor.rag.vectorstore import ReportVectorStore

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    error: bool = False


class ReportQA:
    def __init__(self, vectorstore: ReportVectorStore, client: Optional[BaseLLMClient] = None, top_k: int = 5):
        self.vectorstore = vectorstore
        self.client = client or get_llm_client()
        self.top_k = top_k

    def ask(self, question: str, source_file: Optional[str] = None) -> QAResult:
        retrieved = self.vectorstore.query(question, top_k=self.top_k, source_file=source_file)
        if not retrieved:
            return QAResult(answer="No indexed report content to answer from yet.", sources=[])

        context = "\n\n".join(
            f"[Source: {r.source_file}, p.{r.page_start}-{r.page_end}]\n{r.text}" for r in retrieved
        )
        sources = [
            {"source_file": r.source_file, "page_start": r.page_start, "page_end": r.page_end, "distance": r.distance}
            for r in retrieved
        ]
        try:
            answer = self.client.answer_question(question, context)
        except Exception as exc:
            # Retrieval succeeded (sources above are real); only the LLM call
            # failed, often a transient provider-side error (rate limit,
            # temporary overload). Surface that plainly instead of crashing
            # the CLI/UI - the caller can just retry.
            logger.exception("answer_question failed")
            return QAResult(
                answer=f"The model call failed ({exc}). This is usually transient (e.g. provider overload/rate "
                "limit) - try again in a moment.",
                sources=sources,
                error=True,
            )
        return QAResult(answer=answer, sources=sources)
