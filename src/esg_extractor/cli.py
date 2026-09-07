"""Command-line entrypoints: `python -m esg_extractor.cli <command>`."""

from __future__ import annotations

import json
import os

import click
import pandas as pd

from esg_extractor import config
from esg_extractor.eval.evaluate import evaluate_extractions
from esg_extractor.pipeline import process_corpus, process_report, save_metrics_json
from esg_extractor.rag.qa import ReportQA
from esg_extractor.rag.vectorstore import ReportVectorStore


@click.group()
def cli():
    """ESG report extraction & QA."""


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--company", required=True)
@click.option("--report-year", type=int, default=None)
@click.option("--out-dir", default=str(config.PROCESSED_DIR))
@click.option("--index/--no-index", default=True, help="Also add the report to the RAG index.")
def ingest(pdf_path, company, report_year, out_dir, index):
    """Extract structured KPIs from a single PDF report."""
    vs = ReportVectorStore(persist_dir=str(config.CHROMA_DIR)) if index else None
    metrics = process_report(pdf_path, company=company, report_year=report_year, vectorstore=vs)
    out_path = save_metrics_json(metrics, out_dir)
    click.echo(f"Wrote {out_path}")
    click.echo(json.dumps(metrics.model_dump(), indent=2))


@cli.command()
@click.option("--manifest", "manifest_path", default=str(config.RAW_DIR / "manifest.json"))
@click.option("--pdf-dir", default=str(config.RAW_DIR))
@click.option("--out-dir", default=str(config.PROCESSED_DIR))
def batch(manifest_path, pdf_dir, out_dir):
    """Process every PDF listed in a manifest.json ({filename: {company, report_year}})."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    results = process_corpus(pdf_dir, manifest, out_dir)
    click.echo(f"Processed {len(results)} reports into {out_dir}")


@cli.command()
@click.argument("question")
@click.option("--source-file", default=None, help="Restrict retrieval to one ingested PDF path.")
def ask(question, source_file):
    """Ask a free-text question over the indexed report corpus (RAG)."""
    vs = ReportVectorStore(persist_dir=str(config.CHROMA_DIR))
    qa = ReportQA(vs)
    result = qa.ask(question, source_file=source_file)
    click.echo(result.answer)
    click.echo("\nSources:")
    for s in result.sources:
        click.echo(f"  - {s['source_file']} p.{s['page_start']}-{s['page_end']}")


@cli.command(name="eval")
@click.option("--processed-dir", default=str(config.PROCESSED_DIR))
@click.option("--ground-truth", default=str(config.GROUND_TRUTH_DIR / "ground_truth.csv"))
def eval_cmd(processed_dir, ground_truth):
    """Score extracted metrics JSONs against the hand-labeled ground truth CSV."""
    rows = []
    for fname in os.listdir(processed_dir):
        if fname.endswith(".json"):
            with open(os.path.join(processed_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            row = {"company": data["company"], "report_year": data["report_year"], "source_file": data["source_file"]}
            for key, m in data["metrics"].items():
                row[key] = m.get("value")
            rows.append(row)
    if not rows:
        click.echo(f"No extracted JSON files found in {processed_dir}. Run `ingest`/`batch` first.")
        return

    predictions = pd.DataFrame(rows)
    predictions["source_file"] = predictions["source_file"].apply(os.path.basename)
    gt = pd.read_csv(ground_truth)
    gt["source_file"] = gt["source_file"].apply(os.path.basename)

    table, overall = evaluate_extractions(predictions, gt)
    pd.set_option("display.width", 120)
    click.echo(table.round(3).to_string())
    click.echo("\nOverall:")
    click.echo(json.dumps(overall, indent=2))


if __name__ == "__main__":
    cli()
