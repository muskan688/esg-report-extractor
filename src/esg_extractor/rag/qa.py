"""Retrieval-augmented QA over ingested sustainability reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from esg_extractor.extraction.llm_client import LLMClient
from esg_extractor.rag.vectorstore import ReportVectorStore


@dataclass
class QAResult:
    answer: str
    sources: list[dict]


class ReportQA:
    def __init__(self, vectorstore: ReportVectorStore, client: Optional[LLMClient] = None, top_k: int = 5):
        self.vectorstore = vectorstore
        self.client = client or LLMClient()
        self.top_k = top_k

    def ask(self, question: str, source_file: Optional[str] = None) -> QAResult:
        retrieved = self.vectorstore.query(question, top_k=self.top_k, source_file=source_file)
        if not retrieved:
            return QAResult(answer="No indexed report content to answer from yet.", sources=[])

        context = "\n\n".join(
            f"[Source: {r.source_file}, p.{r.page_start}-{r.page_end}]\n{r.text}" for r in retrieved
        )
        answer = self.client.answer_question(question, context)
        sources = [
            {"source_file": r.source_file, "page_start": r.page_start, "page_end": r.page_end, "distance": r.distance}
            for r in retrieved
        ]
        return QAResult(answer=answer, sources=sources)
