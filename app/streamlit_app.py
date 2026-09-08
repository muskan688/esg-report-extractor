"""Streamlit UI: upload a sustainability report, view extracted KPIs, ask questions.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from esg_extractor import config  # noqa: E402
from esg_extractor.extraction.base import BaseLLMClient  # noqa: E402
from esg_extractor.extraction.provider import get_llm_client  # noqa: E402
from esg_extractor.pipeline import process_report  # noqa: E402
from esg_extractor.rag.qa import ReportQA  # noqa: E402
from esg_extractor.rag.vectorstore import ReportVectorStore  # noqa: E402
from esg_extractor.schema.metrics import METRIC_FIELDS  # noqa: E402

st.set_page_config(page_title="ESG Report Extractor", layout="wide")


@st.cache_resource
def get_vectorstore() -> ReportVectorStore:
    return ReportVectorStore(persist_dir=str(config.CHROMA_DIR))


@st.cache_resource
def get_client() -> BaseLLMClient:
    return get_llm_client()


st.title("ESG / Sustainability Report Extraction Assistant")
st.caption(
    "Upload a company sustainability/ESG PDF report to extract structured KPIs "
    "(emissions, energy, workforce diversity) and ask free-text questions over it."
)

_provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
_key_var = "GEMINI_API_KEY" if _provider == "gemini" else "ANTHROPIC_API_KEY"
if not (os.environ.get(_key_var) or (_provider == "gemini" and os.environ.get("GOOGLE_API_KEY"))):
    st.warning(
        f"LLM_PROVIDER is '{_provider}' but {_key_var} is not set. Extraction and QA calls "
        "will fail until you set it (see .env.example) and restart the app.",
        icon="⚠️",
    )

if "reports" not in st.session_state:
    st.session_state.reports = {}  # source_file -> ESGReportMetrics.model_dump()

with st.sidebar:
    st.header("Ingest a report")
    uploaded = st.file_uploader("Sustainability report (PDF)", type=["pdf"])
    company = st.text_input("Company name", value="")
    report_year = st.number_input("Report year", min_value=2000, max_value=2100, value=2024, step=1)
    run = st.button("Extract KPIs", type="primary", disabled=not (uploaded and company))

    if run and uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        with st.spinner(f"Parsing and extracting KPIs from {uploaded.name}..."):
            metrics = process_report(
                tmp_path,
                company=company,
                report_year=int(report_year),
                vectorstore=get_vectorstore(),
                client=get_client(),
            )
        record = metrics.model_dump()
        record["source_file"] = uploaded.name  # keep the human-readable name, not the temp path
        st.session_state.reports[uploaded.name] = record
        st.success(f"Extracted KPIs for {uploaded.name}")

    st.divider()
    st.caption(
        "No PDF handy? Generate a small synthetic demo corpus:\n\n"
        "`python scripts/generate_sample_reports.py`\n\n"
        "then batch-ingest it with `python -m esg_extractor.cli batch`."
    )

tab_metrics, tab_qa = st.tabs(["Extracted KPIs", "Ask a question"])

with tab_metrics:
    if not st.session_state.reports:
        st.info("Ingest a report from the sidebar to see extracted KPIs here.")
    else:
        report_name = st.selectbox("Report", list(st.session_state.reports.keys()))
        data = st.session_state.reports[report_name]
        st.subheader(f"{data['company']} — {data.get('report_year', '?')}")

        rows = []
        for m in METRIC_FIELDS:
            metric = data["metrics"].get(m.key, {})
            rows.append(
                {
                    "KPI": m.description,
                    "Value": metric.get("value"),
                    "Unit": m.canonical_unit,
                    "As printed": f"{metric.get('raw_value')} {metric.get('raw_unit') or ''}".strip()
                    if metric.get("raw_value") is not None
                    else None,
                    "Page": metric.get("source_page"),
                    "Confidence": metric.get("confidence"),
                    "Source quote": metric.get("quote"),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(st.session_state.reports) > 1:
            st.subheader("Compare across ingested reports")
            all_rows = []
            for record in st.session_state.reports.values():
                row = {"company": record["company"], "report_year": record.get("report_year")}
                for m in METRIC_FIELDS:
                    row[m.key] = record["metrics"].get(m.key, {}).get("value")
                all_rows.append(row)
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

with tab_qa:
    if get_vectorstore().count() == 0:
        st.info("Ingest at least one report first so there's indexed text to answer from.")
    else:
        question = st.text_input("Ask a question about the ingested report(s)", value="")
        scope = st.selectbox(
            "Scope",
            ["All ingested reports"] + list(st.session_state.reports.keys()),
        )
        if st.button("Ask") and question:
            source_filter = None if scope == "All ingested reports" else scope
            with st.spinner("Retrieving relevant excerpts and generating an answer..."):
                qa = ReportQA(get_vectorstore(), client=get_client())
                result = qa.ask(question, source_file=source_filter)
            st.markdown(result.answer)
            with st.expander("Sources used"):
                for s in result.sources:
                    st.write(f"{s['source_file']} — p.{s['page_start']}-{s['page_end']} (distance={s['distance']:.3f})")
