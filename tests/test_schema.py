from esg_extractor.schema.metrics import (
    METRIC_FIELDS,
    METRIC_KEYS,
    ESGReportMetrics,
    ExtractedMetric,
)


def test_metric_keys_are_unique():
    assert len(METRIC_KEYS) == len(set(METRIC_KEYS))
    assert len(METRIC_FIELDS) == len(METRIC_KEYS)


def test_extracted_metric_is_present():
    assert ExtractedMetric(value=1.0).is_present
    assert not ExtractedMetric().is_present


def test_to_flat_dict_includes_all_metric_columns():
    report = ESGReportMetrics(
        company="Acme AG",
        report_year=2024,
        source_file="acme_2024.pdf",
        metrics={"ghg_scope1_tco2e": ExtractedMetric(value=123.0, source_page=5, confidence=0.9)},
    )
    row = report.to_flat_dict()
    assert row["company"] == "Acme AG"
    assert row["ghg_scope1_tco2e"] == 123.0
    assert row["ghg_scope1_tco2e__page"] == 5
    assert row["ghg_scope1_tco2e__confidence"] == 0.9
    # metrics never mentioned still show up as None, not missing keys
    assert row["ghg_scope2_tco2e"] is None
    for key in METRIC_KEYS:
        assert key in row
