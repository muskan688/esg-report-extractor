"""End-to-end pipeline: PDF -> parsed pages -> structured metrics + RAG index."""

from __future__ import annotations

import json
import os
from typing import Optional

from esg_extractor.extraction.base import BaseLLMClient
from esg_extractor.extraction.extractor import MetricExtractor
from esg_extractor.extraction.provider import get_llm_client
from esg_extractor.ingestion.pdf_parser import chunk_report, parse_pdf
from esg_extractor.rag.vectorstore import ReportVectorStore
from esg_extractor.schema.metrics import ESGReportMetrics


def process_report(
    pdf_path: str,
    company: str,
    report_year: Optional[int] = None,
    vectorstore: Optional[ReportVectorStore] = None,
    client: Optional[BaseLLMClient] = None,
    pages_per_chunk: int = 3,
) -> ESGReportMetrics:
    """Parse one PDF, run structured extraction, and index it for QA."""
    report = parse_pdf(pdf_path)
    shared_client = client or get_llm_client()

    extractor = MetricExtractor(client=shared_client)
    metrics = extractor.extract(report, company=company, report_year=report_year)

    if vectorstore is not None:
        chunks = chunk_report(report, pages_per_chunk=pages_per_chunk)
        vectorstore.add_chunks(chunks)

    return metrics


def save_metrics_json(metrics: ESGReportMetrics, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(metrics.source_file))[0]
    out_path = os.path.join(out_dir, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2, ensure_ascii=False)
    return out_path


def process_corpus(
    pdf_dir: str,
    manifest: dict[str, dict],
    out_dir: str,
    vectorstore: Optional[ReportVectorStore] = None,
) -> list[ESGReportMetrics]:
    """Process every PDF listed in ``manifest`` ({filename: {company, report_year}}).

    Sharing one LLM client/vectorstore across the corpus avoids re-instantiating
    the embedding model per file.
    """
    client = get_llm_client()
    vs = vectorstore or ReportVectorStore(persist_dir=os.path.join(out_dir, "chroma"))
    results = []
    for filename, meta in manifest.items():
        pdf_path = os.path.join(pdf_dir, filename)
        metrics = process_report(
            pdf_path,
            company=meta["company"],
            report_year=meta.get("report_year"),
            vectorstore=vs,
            client=client,
        )
        save_metrics_json(metrics, out_dir)
        results.append(metrics)
    return results
