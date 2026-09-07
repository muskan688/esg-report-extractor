"""Evaluate extracted metrics against a hand-labeled ground-truth set.

This is the single highest-signal piece of the project: it proves the
pipeline's numbers can be trusted, rather than just that it runs. For each
KPI field we report precision/recall on *presence* (did we find a number
where one exists, without hallucinating one where none does) and accuracy
*within tolerance* on the values we did extract (unit-normalization bugs or
the LLM reading the wrong column in a table both show up here).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from esg_extractor.schema.metrics import METRIC_KEYS


@dataclass
class FieldScore:
    field: str
    true_positive: int = 0
    wrong_value: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.wrong_value + self.false_positive
        return self.true_positive / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.wrong_value + self.false_negative
        return self.true_positive / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:  # NaN check without importing math
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def support(self) -> int:
        return self.true_positive + self.wrong_value + self.false_negative


def _within_tolerance(pred: float, truth: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(pred - truth) <= max(abs_tol, rel_tol * abs(truth))


def evaluate_extractions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    key_col: str = "source_file",
    rel_tol: float = 0.02,
    abs_tol: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """Score ``predictions`` (from ``ESGReportMetrics.to_flat_dict``) against
    a hand-labeled ``ground_truth`` sharing the same ``key_col`` and metric
    columns. A value counts as correct if it's within ``rel_tol`` relative
    error (default 2%, since printed figures are sometimes themselves
    rounded) of the ground truth.

    Returns (per-field score table, overall summary dict).
    """
    missing_cols = [c for c in METRIC_KEYS if c not in ground_truth.columns]
    if missing_cols:
        raise ValueError(f"ground_truth is missing metric columns: {missing_cols}")

    merged = ground_truth.merge(predictions, on=key_col, how="left", suffixes=("_true", "_pred"))

    scores: dict[str, FieldScore] = {k: FieldScore(field=k) for k in METRIC_KEYS}
    for _, row in merged.iterrows():
        for key in METRIC_KEYS:
            truth = row.get(f"{key}_true", row.get(key))
            pred = row.get(f"{key}_pred")
            truth_present = pd.notna(truth)
            pred_present = pd.notna(pred)
            score = scores[key]
            if truth_present and pred_present:
                if _within_tolerance(float(pred), float(truth), rel_tol, abs_tol):
                    score.true_positive += 1
                else:
                    score.wrong_value += 1
            elif truth_present and not pred_present:
                score.false_negative += 1
            elif not truth_present and pred_present:
                score.false_positive += 1
            else:
                score.true_negative += 1

    table = pd.DataFrame(
        [
            {
                "field": s.field,
                "support": s.support,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
                "true_positive": s.true_positive,
                "wrong_value": s.wrong_value,
                "false_negative": s.false_negative,
                "false_positive": s.false_positive,
            }
            for s in scores.values()
        ]
    ).set_index("field")

    valid = table.dropna(subset=["precision", "recall"])
    overall = {
        "n_reports": int(merged[key_col].nunique()),
        "macro_precision": float(valid["precision"].mean()) if not valid.empty else float("nan"),
        "macro_recall": float(valid["recall"].mean()) if not valid.empty else float("nan"),
        "macro_f1": float(valid["f1"].mean()) if not valid.empty else float("nan"),
        "total_wrong_value": int(table["wrong_value"].sum()),
    }
    return table, overall
