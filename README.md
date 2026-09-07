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
as the actual problem to solve, not a footnote — see [Design notes](#design-notes).

## What it does

1. **Ingest** — parse a PDF's per-page text and tables (`PyMuPDF` + `pdfplumber`).
2. **Locate** — cheap keyword pre-filter narrows a 100-300 page report down
   to the handful of pages that plausibly contain each KPI, instead of
   sending the whole document to an LLM.
3. **Extract** — an LLM (Claude, via forced tool-use / JSON-schema output)
   pulls each KPI from its candidate pages with a verbatim source quote,
   page number, and confidence score attached — never inferred, never
   filled in from general knowledge.
4. **Normalize** — every value is converted to a canonical unit (t CO2e,
   MWh, m³, %, headcount) so figures are comparable across companies/years.
5. **Index** — report text is chunked and embedded (local `sentence-transformers`
   model, no extra API dependency) into a `chromadb` vector store.
6. **Ask** — a RAG layer answers free-text questions ("What's driving the
   Scope 3 increase?") grounded in retrieved excerpts, with page citations.
7. **Evaluate** — extracted values are scored against a hand-labeled ground
   truth set: precision/recall on *presence* (did it find a number without
   hallucinating one) and accuracy *within tolerance* on values found.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env  # then fill in ANTHROPIC_API_KEY
```

No PDFs on hand yet? Generate a small synthetic demo corpus (12 fictional
companies' reports with realistic mixed units/formatting and a matching
hand-labeled ground truth CSV, built for exactly this purpose — see
[Using real reports](#using-real-reports) to point it at real DAX-40 filings
instead):

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

## Evaluation

`python -m esg_extractor.cli eval` scores the pipeline against
[`data/ground_truth/ground_truth.csv`](data/ground_truth) per KPI field:

| metric | meaning |
|---|---|
| `precision` | of the values extracted, what fraction were correct (within 2% tolerance) rather than wrong or hallucinated |
| `recall` | of the values actually disclosed in the report, what fraction were found |
| `wrong_value` | extracted *something*, but outside tolerance of the true value — usually a unit-parsing bug or the wrong column in a multi-year table |
| `false_positive` | extracted a value where the report disclosed nothing — hallucination |

Run it yourself and drop the numbers here — they'll depend on which model
you point `ANTHROPIC_MODEL` at and, for a real corpus, how consistently
those specific reports lay out their KPI tables.

## Architecture

```
PDF ──▶ pdf_parser (PyMuPDF text + pdfplumber tables)
          │
          ├──▶ extractor.find_candidate_pages (keyword pre-filter)
          │      └──▶ llm_client.extract_metrics (Claude, forced tool-use)
          │             └──▶ units.normalize_unit ──▶ ESGReportMetrics (Pydantic)
          │
          └──▶ chunk_report ──▶ embeddings (sentence-transformers)
                                   └──▶ vectorstore (chromadb) ──▶ rag.qa (Claude)
```

- [`src/esg_extractor/schema/metrics.py`](src/esg_extractor/schema/metrics.py) — the single source of truth for
  which 10 KPIs are tracked, their canonical units, and their extraction
  schema.
- [`src/esg_extractor/ingestion/pdf_parser.py`](src/esg_extractor/ingestion/pdf_parser.py) — per-page text/table
  parsing and page-aligned chunking.
- [`src/esg_extractor/extraction/`](src/esg_extractor/extraction) — candidate-page location, the LLM
  tool-use call, and unit normalization.
- [`src/esg_extractor/rag/`](src/esg_extractor/rag) — embeddings, vector store, retrieval-augmented QA.
- [`src/esg_extractor/eval/evaluate.py`](src/esg_extractor/eval/evaluate.py) — precision/recall/tolerance scoring
  against hand-labeled ground truth.
- [`app/streamlit_app.py`](app/streamlit_app.py) — upload a report, view extracted KPIs, ask questions.

## Tech stack

Python · PyMuPDF + pdfplumber (PDF layout) · Anthropic Claude (structured
extraction via tool-use, RAG answer generation) · Pydantic (schema-validated
output) · sentence-transformers + chromadb (local embeddings + vector store)
· pandas (aggregation/eval) · Streamlit (UI) · Docker.

## Using real reports

Drop real sustainability report PDFs (e.g. DAX 40 constituents — freely
published on each company's investor-relations pages) into `data/raw/`,
list them in `data/raw/manifest.json` as `{"filename.pdf": {"company": "...",
"report_year": 2024}}`, then hand-label 5-10 KPIs across a sample of them
into `data/ground_truth/ground_truth.csv` (same columns as the synthetic
one) before trusting the eval numbers. This repo intentionally ships without
a bulk-downloaded real-report corpus — bulk-downloading third-party PDFs
isn't something to automate silently, and it's a five-minute manual step per
company.

As an external sanity check, cross-reference extracted sector-level figures
against Destatis/Eurostat published emissions statistics, or the EU's ESEF/
European Single Access Point filings once broadly available.

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

## License

MIT — see [LICENSE](LICENSE).
