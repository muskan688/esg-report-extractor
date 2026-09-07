"""Structured schema for the ESG metrics this pipeline extracts.

Every metric is normalized to a fixed canonical unit so figures are
comparable across companies and years (reports mix t/kt/Mt CO2e, MWh/GWh,
absolute headcounts vs. percentages, etc.). The canonical unit and a short
description live in ``METRIC_FIELDS`` so extraction, normalization and
evaluation all read from one place instead of re-declaring the metric list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    description: str
    canonical_unit: str
    aliases: tuple[str, ...] = ()


METRIC_FIELDS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "ghg_scope1_tco2e",
        "Direct GHG emissions (Scope 1)",
        "t CO2e",
        ("scope 1", "scope1", "direct emissions"),
    ),
    MetricDefinition(
        "ghg_scope2_tco2e",
        "Indirect energy GHG emissions (Scope 2, market-based where disclosed)",
        "t CO2e",
        ("scope 2", "scope2", "indirect emissions"),
    ),
    MetricDefinition(
        "ghg_scope3_tco2e",
        "Value chain GHG emissions (Scope 3)",
        "t CO2e",
        ("scope 3", "scope3", "value chain emissions"),
    ),
    MetricDefinition(
        "total_energy_consumption_mwh",
        "Total energy consumption",
        "MWh",
        ("energy consumption", "energy use"),
    ),
    MetricDefinition(
        "renewable_energy_share_pct",
        "Share of renewable energy in total energy consumption",
        "%",
        ("renewable energy share", "renewable share"),
    ),
    MetricDefinition(
        "total_workforce_headcount",
        "Total workforce (headcount, FTE where headcount not disclosed)",
        "headcount",
        ("employees", "workforce", "headcount"),
    ),
    MetricDefinition(
        "women_share_total_workforce_pct",
        "Share of women in the total workforce",
        "%",
        ("women in workforce", "female employees", "gender diversity"),
    ),
    MetricDefinition(
        "women_share_management_pct",
        "Share of women in management positions",
        "%",
        ("women in management", "female managers", "women in leadership"),
    ),
    MetricDefinition(
        "water_consumption_m3",
        "Total water consumption/withdrawal",
        "m3",
        ("water consumption", "water withdrawal"),
    ),
    MetricDefinition(
        "waste_total_tonnes",
        "Total waste generated",
        "t",
        ("total waste", "waste generated"),
    ),
)

METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRIC_FIELDS)


class ExtractedMetric(BaseModel):
    """A single extracted, normalized value with provenance for auditing."""

    value: Optional[float] = Field(
        default=None, description="Value normalized to the metric's canonical unit."
    )
    raw_value: Optional[float] = Field(
        default=None, description="Value exactly as printed in the report, before unit conversion."
    )
    raw_unit: Optional[str] = Field(
        default=None, description="Unit exactly as printed in the report, e.g. 'kt CO2e', 'GWh'."
    )
    source_page: Optional[int] = Field(
        default=None, description="1-indexed PDF page the value was read from."
    )
    quote: Optional[str] = Field(
        default=None, description="Short verbatim snippet the value was read from, for auditing."
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Model's self-reported confidence, 0-1."
    )

    @property
    def is_present(self) -> bool:
        return self.value is not None


class ESGReportMetrics(BaseModel):
    """All extracted metrics for one company/report-year, plus metadata."""

    company: str
    report_year: Optional[int] = None
    source_file: str
    metrics: dict[str, ExtractedMetric] = Field(default_factory=dict)

    def get(self, key: str) -> ExtractedMetric:
        return self.metrics.get(key, ExtractedMetric())

    def to_flat_dict(self) -> dict:
        """Flatten to one row per report for pandas aggregation."""
        row = {
            "company": self.company,
            "report_year": self.report_year,
            "source_file": self.source_file,
        }
        for key in METRIC_KEYS:
            m = self.get(key)
            row[key] = m.value
            row[f"{key}__page"] = m.source_page
            row[f"{key}__confidence"] = m.confidence
        return row
