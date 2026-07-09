"""Structured extraction → numeric store (§04 Layer 3) and shared figure parsing."""

from gridfin.extraction import build_numeric_store, extract_figures, scan_value
from gridfin.ingestion import parse_file
from gridfin.ingestion.ingest import Block, ParsedDocument, Table


def test_extract_figures_applies_magnitude():
    figs = extract_figures("net income was $4,200 million")
    assert any(abs(f - 4_200_000_000) < 1 for f in figs)


def test_scan_value_prefers_specific_phrase():
    text = "Revenue grew from $29,000 million in 2021. Total revenue was $33,900 million."
    val, _raw = scan_value(text, "revenue")
    # 'total revenue' is the point figure; the bare 'revenue' growth sentence is not.
    assert abs(val - 33_900_000_000) < 1


def test_numeric_store_from_demo(demo_dir):
    from conftest import ACME, GAMMA

    docs = [parse_file(p) for p in sorted(demo_dir.glob("*.txt"))]
    store = build_numeric_store(docs)
    acme_rev = store.get(ACME, "revenue")
    assert acme_rev is not None
    assert abs(acme_rev.value - 33_900_000_000) < 1
    # Gamma deliberately omits net income → no fact, never a guess.
    assert store.get(GAMMA, "net_income") is None


def test_table_extraction_whole_table():
    table = Table(rows=[["Metric", "FY2023"], ["Net income", "4,200"], ["Total revenue", "33,900"]])
    doc = ParsedDocument(
        doc_id="t", filename="t.csv", filetype="csv",
        blocks=[Block(text=table.to_text(), kind="table", table=table)],
    )
    store = build_numeric_store([doc])
    assert store.get("t", "net_income").value == 4200
    assert store.get("t", "revenue").value == 33900
