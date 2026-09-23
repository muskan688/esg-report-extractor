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
7. **Evaluate** — extracted values are scored against a hand-labeled ground
   truth set: precision/recall on *presence* (did it find a number without
   hallucinating one) and accuracy *within tolerance* on values found.



...

## License

MIT — see [LICENSE](LICENSE).
