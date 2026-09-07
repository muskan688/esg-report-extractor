import pandas as pd

from esg_extractor.eval.evaluate import evaluate_extractions
from esg_extractor.schema.metrics import METRIC_KEYS


def _row(source_file, **overrides):
    row = {"source_file": source_file}
    for key in METRIC_KEYS:
        row[key] = overrides.get(key)
    return row


def test_evaluate_extractions_scores_presence_and_tolerance():
    ground_truth = pd.DataFrame(
        [
            _row("a.pdf", ghg_scope1_tco2e=1000.0, total_workforce_headcount=500.0),
            _row("b.pdf", ghg_scope1_tco2e=2000.0),
        ]
    )
    predictions = pd.DataFrame(
        [
            # correct within tolerance
            _row("a.pdf", ghg_scope1_tco2e=1005.0, total_workforce_headcount=None),
            # wrong value (way outside 2% tolerance)
            _row("b.pdf", ghg_scope1_tco2e=500.0),
        ]
    )

    table, overall = evaluate_extractions(predictions, ground_truth, rel_tol=0.02)

    scope1 = table.loc["ghg_scope1_tco2e"]
    assert scope1["true_positive"] == 1
    assert scope1["wrong_value"] == 1
    assert scope1["false_negative"] == 0

    headcount = table.loc["total_workforce_headcount"]
    assert headcount["false_negative"] == 1  # ground truth had it, prediction missed it

    assert overall["n_reports"] == 2
    assert 0.0 <= overall["macro_precision"] <= 1.0


def test_evaluate_extractions_flags_hallucinated_value_as_false_positive():
    ground_truth = pd.DataFrame([_row("a.pdf")])  # nothing disclosed
    predictions = pd.DataFrame([_row("a.pdf", ghg_scope1_tco2e=999.0)])

    table, _overall = evaluate_extractions(predictions, ground_truth)
    assert table.loc["ghg_scope1_tco2e", "false_positive"] == 1


def test_evaluate_extractions_raises_on_missing_ground_truth_columns():
    import pytest

    ground_truth = pd.DataFrame([{"source_file": "a.pdf"}])
    predictions = pd.DataFrame([{"source_file": "a.pdf"}])
    with pytest.raises(ValueError):
        evaluate_extractions(predictions, ground_truth)
