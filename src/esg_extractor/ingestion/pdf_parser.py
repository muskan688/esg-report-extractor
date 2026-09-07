"""PDF layout parsing: per-page text + tables, and chunking for retrieval.

Sustainability reports mix narrative text, charts-as-images (unrecoverable
without OCR/vision, so we don't try) and data tables. Tables carry most of
the KPI figures (emissions by scope, energy mix, diversity ratios) so we
extract them separately with pdfplumber and render them back to a markdown-ish
text block that reads well for both the LLM extractor and the RAG index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str
    tables: list[list[list[Optional[str]]]] = field(default_factory=list)

    def as_markdown(self) -> str:
        parts = [self.text.strip()]
        for i, table in enumerate(self.tables):
            parts.append(f"\n[Table {i + 1} on page {self.page_number}]")
            parts.append(_table_to_markdown(table))
        return "\n".join(p for p in parts if p.strip())


@dataclass
class ParsedReport:
    source_file: str
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.as_markdown() for p in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def _table_to_markdown(table: list[list]) -> str:
    rows = [[("" if c is None else str(c).strip()) for c in row] for row in table]
    if not rows:
        return ""
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    if len(rows) > 1:
        header_sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
        lines.insert(1, header_sep)
    return "\n".join(lines)


def parse_pdf(path: str, extract_tables: bool = True, max_pages: int | None = None) -> ParsedReport:
    """Parse a PDF into per-page text + tables.

    Text extraction uses PyMuPDF (fast, good text layout fidelity). Table
    extraction uses pdfplumber's line-detection algorithm, which handles the
    ruled tables common in annual/sustainability reports well; it's slower
    so pages are only opened once via a shared pdfplumber document when
    ``extract_tables`` is on.
    """
    import pymupdf as fitz

    pages: list[ParsedPage] = []
    doc = fitz.open(path)
    try:
        n_pages = len(doc) if max_pages is None else min(max_pages, len(doc))
        texts = [doc[i].get_text("text") for i in range(n_pages)]
    finally:
        doc.close()

    tables_by_page: dict[int, list] = {}
    if extract_tables:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            for i in range(n):
                try:
                    found = pdf.pages[i].extract_tables()
                except Exception:
                    found = []
                if found:
                    tables_by_page[i] = found

    for i, text in enumerate(texts):
        pages.append(
            ParsedPage(
                page_number=i + 1,
                text=_clean_text(text),
                tables=tables_by_page.get(i, []),
            )
        )
    return ParsedReport(source_file=path, pages=pages)


def _clean_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen from justified PDF text
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class TextChunk:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str


def chunk_report(report: ParsedReport, pages_per_chunk: int = 2, overlap_pages: int = 0) -> list[TextChunk]:
    """Group consecutive pages into overlap-able chunks for embedding/RAG.

    Page-aligned (rather than token-window) chunking keeps each chunk's
    provenance ("page 42-43") meaningful for citations in QA answers.
    """
    if pages_per_chunk < 1:
        raise ValueError("pages_per_chunk must be >= 1")
    step = max(1, pages_per_chunk - overlap_pages)
    chunks: list[TextChunk] = []
    pages = report.pages
    i = 0
    while i < len(pages):
        window = pages[i : i + pages_per_chunk]
        if not window:
            break
        text = "\n\n".join(p.as_markdown() for p in window).strip()
        if text:
            chunks.append(
                TextChunk(
                    chunk_id=f"{report.source_file}::p{window[0].page_number}-{window[-1].page_number}",
                    source_file=report.source_file,
                    page_start=window[0].page_number,
                    page_end=window[-1].page_number,
                    text=text,
                )
            )
        i += step
    return chunks
