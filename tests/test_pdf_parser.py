from pathlib import Path

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from esg_extractor.ingestion.pdf_parser import chunk_report, parse_pdf


@pytest.fixture
def sample_pdf(tmp_path: Path) -> str:
    path = tmp_path / "sample.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    story = [
        Paragraph("Hello sustainability world.", styles["Title"]),
        Paragraph("Scope 1 emissions decreased year over year.", styles["BodyText"]),
        PageBreak(),
    ]
    table = Table([["Indicator", "2024"], ["Scope 1 GHG emissions", "12.345,0 t CO2e"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(table)
    story.append(PageBreak())
    story.append(Paragraph("Third page with no tables.", styles["BodyText"]))
    doc.build(story)
    return str(path)


def test_parse_pdf_extracts_text_per_page(sample_pdf):
    report = parse_pdf(sample_pdf)
    assert report.num_pages == 3
    assert "Hello sustainability world" in report.pages[0].text
    assert "Third page" in report.pages[2].text


def test_parse_pdf_extracts_tables(sample_pdf):
    report = parse_pdf(sample_pdf)
    page_with_table = report.pages[1]
    assert len(page_with_table.tables) >= 1
    flat = [cell for row in page_with_table.tables[0] for cell in row]
    assert any(cell and "Scope 1" in cell for cell in flat)
    assert "Table 1 on page 2" in page_with_table.as_markdown()


def test_chunk_report_groups_pages(sample_pdf):
    report = parse_pdf(sample_pdf)
    chunks = chunk_report(report, pages_per_chunk=2, overlap_pages=0)
    assert len(chunks) == 2
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert chunks[1].page_start == 3 and chunks[1].page_end == 3
    assert chunks[0].chunk_id.endswith("::p1-2")
