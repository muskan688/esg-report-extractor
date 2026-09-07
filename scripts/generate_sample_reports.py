"""Generate a small synthetic corpus of sustainability-report-style PDFs.

Real DAX-40 sustainability reports are freely downloadable from each
company's investor-relations site, but pulling and hand-labeling a real
corpus is a manual step deliberately left to the user (see README ->
"Using real reports"), since it involves bulk-downloading third-party PDFs.

This script instead builds a small set of *fictional* companies' reports
with realistic layout: narrative filler pages (to check the pipeline
doesn't hallucinate KPIs off marketing text), a KPI table page using
German/EU-style formatting and inconsistent units on purpose (kt vs t
CO2e, MWh vs GWh, comma-decimal numbers), a decoy financial table (to
check it doesn't confuse revenue/EBITDA for an ESG KPI), and a randomly
missing subset of KPIs (not every company discloses everything).

Run: python scripts/generate_sample_reports.py
It writes PDFs + manifest.json to data/raw/ and the matching hand-labeled
ground_truth.csv to data/ground_truth/, both keyed off the same underlying
"true" values so the eval set is internally consistent by construction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
GT_DIR = ROOT / "data" / "ground_truth"

COMPANIES = [
    "Nordwind Energie AG",
    "BlauStahl Werke SE",
    "Rheinaue Chemie AG",
    "Gruenbau Logistik SE",
    "Alpen Mobilitaet AG",
    "Ostsee Maschinenbau AG",
]
YEARS = [2023, 2024]

NARRATIVE_PARAGRAPHS = [
    "Sustainability remains central to our long-term strategy. In the reporting "
    "year, we continued to invest in the transition of our operations towards "
    "lower-carbon processes and a more resource-efficient supply chain.",
    "Our Supervisory Board and Management Board reviewed climate-related risks "
    "as part of the enterprise risk management process, in line with emerging "
    "CSRD and ESRS reporting requirements applicable to the Group.",
    "We remain committed to fair and inclusive working conditions across all "
    "sites. Employee wellbeing programmes were expanded during the year, and "
    "engagement survey participation increased across all business segments.",
    "Stakeholder dialogue continued throughout the year through supplier "
    "audits, community engagement initiatives, and regular disclosure updates "
    "published on the Group's investor relations website.",
]

# metric key -> list of (unit_label, value -> displayed_value) alternatives
UNIT_VARIANTS = {
    "ghg_scope1_tco2e": [("t CO2e", lambda v: v), ("kt CO2e", lambda v: v / 1000)],
    "ghg_scope2_tco2e": [("t CO2e", lambda v: v), ("kt CO2e", lambda v: v / 1000)],
    "ghg_scope3_tco2e": [("t CO2e", lambda v: v), ("Mio. t CO2e", lambda v: v / 1_000_000)],
    "total_energy_consumption_mwh": [("MWh", lambda v: v), ("GWh", lambda v: v / 1000)],
    "renewable_energy_share_pct": [("%", lambda v: v)],
    "total_workforce_headcount": [("employees", lambda v: v), ("thousand employees", lambda v: v / 1000)],
    "women_share_total_workforce_pct": [("%", lambda v: v)],
    "women_share_management_pct": [("%", lambda v: v)],
    "water_consumption_m3": [("m3", lambda v: v), ("thousand m3", lambda v: v / 1000)],
    "waste_total_tonnes": [("t", lambda v: v), ("kt", lambda v: v / 1000)],
}

METRIC_LABELS = {
    "ghg_scope1_tco2e": "GHG Emissions Scope 1",
    "ghg_scope2_tco2e": "GHG Emissions Scope 2 (market-based)",
    "ghg_scope3_tco2e": "GHG Emissions Scope 3",
    "total_energy_consumption_mwh": "Total energy consumption",
    "renewable_energy_share_pct": "Share of renewable energy",
    "total_workforce_headcount": "Total workforce",
    "women_share_total_workforce_pct": "Share of women in total workforce",
    "women_share_management_pct": "Share of women in management",
    "water_consumption_m3": "Total water consumption",
    "waste_total_tonnes": "Total waste generated",
}


def _fmt_de(value: float) -> str:
    """Format like German/EU reports: '.' thousands sep, ',' decimal."""
    s = f"{value:,.1f}"
    s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return s


def _true_values(rng: random.Random, base_scale: float) -> dict[str, float]:
    values = {
        "ghg_scope1_tco2e": rng.uniform(5_000, 80_000) * base_scale,
        "ghg_scope2_tco2e": rng.uniform(2_000, 40_000) * base_scale,
        "ghg_scope3_tco2e": rng.uniform(100_000, 2_000_000) * base_scale,
        "total_energy_consumption_mwh": rng.uniform(50_000, 900_000) * base_scale,
        "renewable_energy_share_pct": rng.uniform(15, 85),
        "total_workforce_headcount": rng.randint(1_500, 40_000),
        "women_share_total_workforce_pct": rng.uniform(20, 48),
        "women_share_management_pct": rng.uniform(15, 42),
        "water_consumption_m3": rng.uniform(50_000, 1_500_000) * base_scale,
        "waste_total_tonnes": rng.uniform(2_000, 120_000) * base_scale,
    }
    return {k: round(v, 2) for k, v in values.items()}


def _build_pdf(
    path: Path,
    company: str,
    year: int,
    true_values: dict[str, float],
    disclosed: set[str],
    rng: random.Random,
) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    story = []

    story.append(Paragraph(f"{company}", styles["Title"]))
    story.append(Paragraph(f"Sustainability Report {year}", styles["Heading2"]))
    story.append(Spacer(1, 1 * cm))
    for para in NARRATIVE_PARAGRAPHS:
        story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 0.4 * cm))
    story.append(PageBreak())

    story.append(Paragraph("Our approach to climate action", styles["Heading2"]))
    story.append(Paragraph(rng.choice(NARRATIVE_PARAGRAPHS), styles["BodyText"]))
    story.append(Paragraph(
        f"{company} continues to align its climate transition plan with the Paris "
        "Agreement and applicable EU regulation, integrating findings into group-wide "
        "capital allocation decisions.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    # Decoy financial table (should NOT be picked up as an ESG KPI).
    story.append(Paragraph("Selected financial highlights", styles["Heading3"]))
    fin_table = [
        ["Metric", str(year - 1), str(year)],
        ["Revenue (EUR m)", _fmt_de(rng.uniform(800, 6000)), _fmt_de(rng.uniform(800, 6000))],
        ["EBITDA (EUR m)", _fmt_de(rng.uniform(100, 900)), _fmt_de(rng.uniform(100, 900))],
    ]
    story.append(_styled_table(fin_table))
    story.append(PageBreak())

    story.append(Paragraph("ESG Facts & Figures", styles["Heading2"]))
    story.append(Paragraph(
        "The table below summarizes key non-financial performance indicators for the "
        "current and prior reporting year, prepared in accordance with the Group's "
        "sustainability reporting methodology.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    kpi_rows = [["Indicator", f"{year - 1}", f"{year}", "Unit"]]
    for key in disclosed:
        label = METRIC_LABELS[key]
        unit_label, convert = rng.choice(UNIT_VARIANTS[key])
        cur_true = true_values[key]
        prev_true = cur_true * rng.uniform(0.85, 1.15)
        kpi_rows.append([
            label,
            _fmt_de(convert(prev_true)),
            _fmt_de(convert(cur_true)),
            unit_label,
        ])
    story.append(_styled_table(kpi_rows))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Figures for prior years may be restated to reflect changes in the "
        "consolidation scope or measurement methodology.",
        styles["Italic"],
    ))

    doc.build(story)


def _styled_table(rows: list[list[str]]) -> Table:
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def main(seed: int = 42, disclosure_rate: float = 0.85) -> None:
    rng = random.Random(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    gt_rows = []

    for company in COMPANIES:
        base_scale = rng.uniform(0.5, 3.0)
        for year in YEARS:
            filename = f"{company.lower().replace(' ', '_').replace('.', '')}_{year}.pdf"
            true_values = _true_values(rng, base_scale)
            disclosed = {k for k in true_values if rng.random() < disclosure_rate}

            _build_pdf(RAW_DIR / filename, company, year, true_values, disclosed, rng)

            manifest[filename] = {"company": company, "report_year": year}
            row = {"source_file": filename, "company": company, "report_year": year}
            for key, val in true_values.items():
                row[key] = val if key in disclosed else None
            gt_rows.append(row)

    with open(RAW_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    import csv

    fieldnames = ["source_file", "company", "report_year"] + list(UNIT_VARIANTS.keys())
    with open(GT_DIR / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gt_rows)

    print(f"Wrote {len(manifest)} PDFs to {RAW_DIR}")
    print(f"Wrote ground truth for {len(gt_rows)} reports to {GT_DIR / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
