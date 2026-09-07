from esg_extractor.extraction.extractor import MetricExtractor, _merge_pages_to_windows, find_candidate_pages
from esg_extractor.ingestion.pdf_parser import ParsedPage, ParsedReport


def _report(pages_text: list[str]) -> ParsedReport:
    return ParsedReport(
        source_file="test.pdf",
        pages=[ParsedPage(page_number=i + 1, text=t) for i, t in enumerate(pages_text)],
    )


def test_merge_pages_to_windows_merges_nearby_and_pads():
    windows = _merge_pages_to_windows([5, 6, 20], gap=2, pad=1, max_page=100)
    assert windows == [(4, 7), (19, 21)]


def test_merge_pages_to_windows_empty():
    assert _merge_pages_to_windows([]) == []


def test_find_candidate_pages_matches_aliases():
    report = _report(
        [
            "General company overview, no numbers here.",
            "Scope 1 emissions totalled 12,000 t CO2e this year.",
            "We had 5,000 employees across our sites.",
        ]
    )
    candidates = find_candidate_pages(report)
    assert candidates["ghg_scope1_tco2e"] == [2]
    assert candidates["total_workforce_headcount"] == [3]
    assert candidates["renewable_energy_share_pct"] == []


class FakeLLMClient:
    """Stands in for LLMClient.extract_metrics without any network call."""

    def extract_metrics(self, text: str, metric_keys=None):
        result = {}
        if metric_keys and "ghg_scope1_tco2e" in metric_keys and "Scope 1" in text:
            result["ghg_scope1_tco2e"] = {
                "value": 12.0,
                "unit": "kt CO2e",
                "quote": "Scope 1 emissions totalled 12 kt CO2e",
                "confidence": 0.95,
            }
        if metric_keys and "total_workforce_headcount" in metric_keys and "employ" in text:
            result["total_workforce_headcount"] = {
                "value": 5000,
                "unit": "employees",
                "quote": "We employ 5,000 people",
                "confidence": 0.8,
            }
        return result


def test_metric_extractor_end_to_end_with_fake_client():
    report = _report(
        [
            "General company overview, no numbers here.",
            "Scope 1 emissions totalled 12 kt CO2e this year.",
            "We had 5,000 employees across our sites.",
        ]
    )
    extractor = MetricExtractor(client=FakeLLMClient())
    metrics = extractor.extract(report, company="Acme AG", report_year=2024)

    scope1 = metrics.get("ghg_scope1_tco2e")
    assert scope1.value == 12_000.0  # kt -> t normalization
    assert scope1.confidence == 0.95

    headcount = metrics.get("total_workforce_headcount")
    assert headcount.value == 5000.0

    # never mentioned -> stays absent, not hallucinated
    assert not metrics.get("renewable_energy_share_pct").is_present
