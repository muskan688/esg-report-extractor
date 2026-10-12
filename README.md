# ESG Report Extraction Assistant

Extracts structured sustainability metrics (GHG emissions by scope, energy
mix, workforce diversity, water, waste) from companies' public ESG/
sustainability PDF reports, and answers free-text questions over them —
built against the backdrop of the EU's **Corporate Sustainability Reporting
Directive (CSRD)**, which is pushing thousands of German and EU companies to
produce standardized, machine-checkable ESG disclosures.

Sustainability reports are genuinely messy documents: mixed narrative text,
charts-as-images, KPI tables with inconsistent layouts, and units that vary
company-to-company (`t CO2e` vs `kt CO2e`, `MWh` vs `GWh`, German
`1.234,5`-style decimal formatting vs `1,234.5`). This project treats that
as the actual problem to solve, not a footnote.

## What it does

1. **Ingest** — parse a PDF's per-page text and tables (`PyMuPDF` + `pdfplumber`).
2. **Locate** — cheap keyword pre-filter narrows a 100-300 page report down
   to the handful of pages that plausibly contain each KPI, instead of
   sending the whole document to an LLM.
3. **Extract** — an LLM (Claude or Gemini, via forced tool/function-calling
   with a JSON-schema output) pulls each KPI from its candidate pages with a
   verbatim source quote, page number, and confidence score attached —
   never inferred, never filled in from general knowledge.
4. **Normalize** — every value is converted to a canonical unit (t CO2e,
   MWh, m³, %, headcount) so figures are comparable across companies/years.
5. **Index** — report text is chunked and embedded (local `sentence-transformers`
   model, no extra API dependency) into a `chromadb` vector store.
6. **Ask** — a RAG layer answers free-text questions ("What's driving the
   Scope 3 increase?") grounded in retrieved excerpts, with page citations.
7. **Evaluate** — extracted values are scored against a ground truth set
   (for the bundled demo corpus: synthetic reports with known values; for real
   reports: hand-labeled): precision/recall on *presence* (did it find a number
   without hallucinating one) and accuracy *within tolerance* on values found.

## Results (synthetic benchmark)

Evaluated on **12 generated reports** (6 fictional companies × 2 years) with
**known ground truth**: 10 KPIs per report = 120 metric cells, of which **100 are
disclosed** in the PDF and **20 are deliberately left out** to test for
hallucination. The reports and their ground truth both come from
[`scripts/generate_sample_reports.py`](scripts/generate_sample_reports.py)
(fixed seed), so this is a benchmark of the pipeline, not of real company reports.

- **Model:** Gemini (`gemini-3.6-flash`). The Claude provider is implemented but was not
  run for these numbers.
- **Run:** 2026-09-26, 12 LLM calls, 0 failed after retries (so no API outage is
  counted as a miss).
- **Precision: 0.99** (99 of 100 extracted values correct within 2%)
- **Recall: 0.99** (99 of 100 disclosed values found)
- **F1: 0.99**
- **Hallucinated values: 0** (none of the 20 undisclosed cells was filled in)
- **Wrong values: 1** — Blaustahl Werke 2023 Scope 3. The model read the printed
  "0,9 Mio. t CO2e" correctly; the report rounds to one decimal in millions, so it is 4%
  from the underlying 939,745 t.

Per-field scores: [`results/eval_scores.csv`](results/eval_scores.csv). Ground truth:
[`data/ground_truth/ground_truth.csv`](data/ground_truth/ground_truth.csv).
Because the corpus is synthetic and clean, expect lower recall on real, messier reports.
To reproduce: `python scripts/generate_sample_reports.py`, then
`python -m esg_extractor.cli batch` and `python -m esg_extractor.cli eval` (needs a Gemini API key).

## License

MIT — see [LICENSE](LICENSE).
