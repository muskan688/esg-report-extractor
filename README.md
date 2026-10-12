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

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env  # set LLM_PROVIDER + the matching API key
```

Two providers are supported out of the box, picked by `LLM_PROVIDER`
(`anthropic` or `gemini`) — both go through the same
[`BaseLLMClient`](src/esg_extractor/extraction/base.py) interface, so
nothing else in the pipeline changes when you switch. Gemini has a free API
tier (via [Google AI Studio](https://aistudio.google.com/)); Anthropic
requires a paid key.

No PDFs on hand yet? Generate a small synthetic demo corpus (12 fictional
companies' reports with realistic mixed units/formatting and a matching
ground truth CSV with known values; to use real reports, put the PDFs in
`data/raw/`, list them in `data/raw/manifest.json`, and label your own ground
truth):

```bash
python scripts/generate_sample_reports.py
```

Extract KPIs from every report in the demo corpus and index them for QA:

```bash
python -m esg_extractor.cli batch
python -m esg_extractor.cli ask "What was Scope 1 last year?"
```

Score the extraction pipeline against ground truth:

```bash
python -m esg_extractor.cli eval
```

Or drive it all through the UI:

```bash
streamlit run app/streamlit_app.py
```

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

Per-field scores: [`results/eval_scores.csv`](results/eval_scores.csv). Raw extractions:
[`results/predictions/`](results/predictions). Ground truth:
[`data/ground_truth/ground_truth.csv`](data/ground_truth/ground_truth.csv).
Because the corpus is synthetic and clean, expect lower recall on real, messier reports.
To reproduce: `python scripts/generate_sample_reports.py`, then
`python -m esg_extractor.cli batch` and `python -m esg_extractor.cli eval` (needs a Gemini API key).

## Architecture

```
PDF ──▶ pdf_parser (PyMuPDF text + pdfplumber tables)
          │
          ├──▶ extractor.find_candidate_pages (keyword pre-filter)
          │      └──▶ provider.get_llm_client().extract_metrics (forced tool/function-calling)
          │             └──▶ units.normalize_unit ──▶ ESGReportMetrics (Pydantic)
          │
          └──▶ chunk_report ──▶ embeddings (sentence-transformers)
                                   └──▶ vectorstore (chromadb) ──▶ rag.qa (same LLM client)
```

- [`src/esg_extractor/schema/metrics.py`](src/esg_extractor/schema/metrics.py) — the single source of truth for
  which 10 KPIs are tracked, their canonical units, and their extraction
  schema.
- [`src/esg_extractor/ingestion/pdf_parser.py`](src/esg_extractor/ingestion/pdf_parser.py) — per-page text/table
  parsing and page-aligned chunking.
- [`src/esg_extractor/extraction/`](src/esg_extractor/extraction) — candidate-page location, unit
  normalization, the shared tool schema/prompts, and the Anthropic/Gemini
  clients behind `provider.get_llm_client()`.
- [`src/esg_extractor/rag/`](src/esg_extractor/rag) — embeddings, vector store, retrieval-augmented QA.
- [`src/esg_extractor/eval/evaluate.py`](src/esg_extractor/eval/evaluate.py) — precision/recall/tolerance scoring
  against a ground truth CSV.
- [`app/streamlit_app.py`](app/streamlit_app.py) — upload a report, view extracted KPIs, ask questions.
- [`Dockerfile`](Dockerfile) — container image for the Streamlit app; built and smoke-tested in CI.

## Tech stack

Python · PyMuPDF + pdfplumber (PDF layout) · Claude or Gemini (structured
extraction via tool/function-calling, RAG answer generation - swappable via
`LLM_PROVIDER`) · Pydantic (schema-validated output) · sentence-transformers
+ chromadb (local embeddings + vector store) · pandas (aggregation/eval) ·
Streamlit (UI) · Docker.

## Design notes

- **Candidate-page pre-filtering exists because reports are long.** A
  100-300 page PDF sent whole to an LLM per KPI is slow and expensive; a
  cheap alias keyword scan narrows each KPI down to a few candidate pages
  first (see `extractor.find_candidate_pages`).
- **Every extracted value carries a verbatim quote, page number, and
  confidence** — extraction without provenance isn't auditable, and ESG
  figures feeding compliance decisions need to be checkable against source.
- **The LLM is explicitly told not to infer or estimate** — a missing KPI
  should come back absent, not filled in with a plausible-looking number.
  The eval harness's `false_positive` count is what would catch a
  regression here.
- **Unit normalization is a data-cleaning problem, not a modeling one** —
  `extraction/units.py` handles magnitude prefixes (kt/Mt/million/thousand),
  unit families (mass/energy/volume/percent/count), and German vs.
  English decimal formatting, independently of the LLM call.
- **Failed API calls are retried and counted.** Extraction retries with
  exponential backoff, and each report records `total_calls`/`failed_calls`,
  so a provider outage shows up as understated recall instead of being
  mistaken for a model error.

## Limitations

- Charts-as-images (a common way KPIs are visualized) aren't read — only
  text and ruled tables. A vision-capable extraction pass over page images
  would be the natural extension.
- Extraction quality depends on how consistently a given report labels its
  KPI tables; heavily narrative or infographic-only disclosures will have
  lower recall than a report with an explicit "ESG Facts & Figures" table.
- Scope 2 is not split into location-based vs. market-based; if a report
  discloses both, whichever the model judges "market-based" (or the only
  one given) is taken.
- The candidate-page pre-filter matches keyword aliases, so it is sensitive to
  phrasing. Two KPIs were silently missed during development because the report
  said "share of renewable energy" and the alias list said "renewable energy
  share". Unusual labels in real reports can be missed the same way.
- The benchmark is synthetic (generated reports, generated ground truth), so
  the 0.99 scores validate the pipeline, not performance on real reports.
- The numbers come from a single run with Gemini. The Anthropic provider is
  implemented but has not been run against a live API.

## License

MIT — see [LICENSE](LICENSE).
