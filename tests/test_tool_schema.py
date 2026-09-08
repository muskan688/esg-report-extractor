from esg_extractor.extraction.tool_schema import build_metric_properties, build_user_message
from esg_extractor.schema.metrics import METRIC_KEYS


def test_build_metric_properties_all_by_default():
    props = build_metric_properties()
    assert set(props.keys()) == set(METRIC_KEYS)
    for spec in props.values():
        assert spec["required"] == ["value", "unit", "quote", "confidence"]


def test_build_metric_properties_filters_to_requested_keys():
    props = build_metric_properties(["ghg_scope1_tco2e", "total_workforce_headcount"])
    assert set(props.keys()) == {"ghg_scope1_tco2e", "total_workforce_headcount"}


def test_build_user_message_lists_requested_kpis_and_wraps_text():
    msg = build_user_message("Some report text.", ["ghg_scope1_tco2e"])
    assert "ghg_scope1_tco2e" in msg
    assert "Some report text." in msg
    assert "record_esg_metrics" in msg


def test_build_user_message_all_when_keys_none():
    msg = build_user_message("text", None)
    assert "Requested KPIs: all" in msg
