"""Orchestrates PDF -> candidate pages -> LLM extraction -> normalized metrics."""

from __future__ import annotations

import logging
from typing import Optional

from esg_extractor.extraction.base import BaseLLMClient
from esg_extractor.extraction.provider import get_llm_client
from esg_extractor.extraction.units import normalize_unit
from esg_extractor.ingestion.pdf_parser import ParsedReport
from esg_extractor.schema.metrics import METRIC_FIELDS, ESGReportMetrics, ExtractedMetric

logger = logging.getLogger(__name__)


def find_candidate_pages(report: ParsedReport) -> dict[str, list[int]]:
    """Keyword pre-filter: which pages plausibly mention each metric.

    Sustainability reports run 80-300 pages; sending the whole document to
    the LLM is slow and expensive. Most KPIs cluster in a "Facts & Figures"
    / ESG data appendix, so a cheap alias match narrows each metric down to
    a handful of candidate pages before the LLM ever runs.
    """
    candidates: dict[str, list[int]] = {m.key: [] for m in METRIC_FIELDS}
    for page in report.pages:
        haystack = page.as_markdown().lower()
        for m in METRIC_FIELDS:
            if any(alias in haystack for alias in m.aliases):
                candidates[m.key].append(page.page_number)
    return candidates


def _merge_pages_to_windows(
    pages: list[int], gap: int = 1, pad: int = 1, max_page: int = 10**9
) -> list[tuple[int, int]]:
    """Merge a sorted-ish set of page numbers into (start, end) windows,
    padding by ``pad`` pages on each side so a KPI split across a page
    break isn't cut off, and merging windows that end up within ``gap``
    pages of each other."""
    if not pages:
        return []
    uniq = sorted(set(pages))
    windows: list[list[int]] = [[uniq[0], uniq[0]]]
    for p in uniq[1:]:
        if p - windows[-1][1] <= gap + 1:
            windows[-1][1] = p
        else:
            windows.append([p, p])
    return [(max(1, s - pad), min(max_page, e + pad)) for s, e in windows]


class MetricExtractor:
    def __init__(self, client: Optional[BaseLLMClient] = None, page_pad: int = 1, page_gap: int = 2):
        self.client = client or get_llm_client()
        self.page_pad = page_pad
        self.page_gap = page_gap

    def extract(self, report: ParsedReport, company: str, report_year: Optional[int] = None) -> ESGReportMetrics:
        candidates = find_candidate_pages(report)
        pages_by_num = {p.page_number: p for p in report.pages}

        # invert: window -> metric keys it should be queried for, so a page
        # range covering several KPIs' aliases is only sent to the LLM once.
        windows_for_metric: dict[str, list[tuple[int, int]]] = {
            key: _merge_pages_to_windows(pages, gap=self.page_gap, pad=self.page_pad, max_page=report.num_pages)
            for key, pages in candidates.items()
        }

        window_to_keys: dict[tuple[int, int], set[str]] = {}
        for key, windows in windows_for_metric.items():
            for w in windows:
                window_to_keys.setdefault(w, set()).add(key)

        found: dict[str, list[ExtractedMetric]] = {m.key: [] for m in METRIC_FIELDS}

        for (start, end), keys in window_to_keys.items():
            chunk_text = "\n\n".join(pages_by_num[p].as_markdown() for p in range(start, end + 1) if p in pages_by_num)
            if not chunk_text.strip():
                continue
            try:
                raw_result = self.client.extract_metrics(chunk_text, metric_keys=sorted(keys))
            except Exception:
                logger.exception("extraction call failed for pages %s-%s", start, end)
                continue
            for key, payload in raw_result.items():
                metric_def = next((m for m in METRIC_FIELDS if m.key == key), None)
                if metric_def is None or not payload:
                    continue
                norm = normalize_unit(payload.get("value"), payload.get("unit"), metric_def.canonical_unit)
                found[key].append(
                    ExtractedMetric(
                        value=norm.value if norm.matched else None,
                        raw_value=payload.get("value"),
                        raw_unit=payload.get("unit"),
                        source_page=start,
                        quote=payload.get("quote"),
                        confidence=payload.get("confidence"),
                    )
                )

        metrics: dict[str, ExtractedMetric] = {}
        for key, candidates_list in found.items():
            metrics[key] = _pick_best(candidates_list)

        return ESGReportMetrics(
            company=company,
            report_year=report_year,
            source_file=report.source_file,
            metrics=metrics,
        )


def _pick_best(candidates: list[ExtractedMetric]) -> ExtractedMetric:
    """Among duplicate mentions of a KPI (e.g. also appears in a 5-year
    trend table), prefer the highest-confidence, most-complete one."""
    present = [c for c in candidates if c.is_present]
    if not present:
        return ExtractedMetric()
    return max(present, key=lambda c: (c.confidence or 0.0))
