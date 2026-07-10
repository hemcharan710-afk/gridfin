"""Evaluation on real SEC filings (not synthetic fixtures).

The figures in `data/sec/` are the exact amounts Apple, Microsoft, and NVIDIA
reported in their FY2023 Form 10-Ks (source: SEC EDGAR XBRL); `ground_truth.json`
holds those figures and the ratios derived from them. This test grounds the three
grid metrics on real filings rather than the toy demo corpus.
"""

import json
from pathlib import Path

import pytest

from gridfin.eval import evaluate_grid

SEC = Path(__file__).resolve().parent.parent / "data" / "sec"


@pytest.fixture
def sec_corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDFIN_CACHE_DIR", str(tmp_path / "cache"))
    from gridfin.config import get_settings

    get_settings.cache_clear()
    from gridfin.pipeline import GridFin

    return GridFin.from_path(SEC, cache_namespace=f"cells_{tmp_path.name}")


async def test_real_sec_metrics(sec_corpus):
    truth = json.loads((SEC / "ground_truth.json").read_text())
    truth = {k: v for k, v in truth.items() if not k.startswith("_")}
    assert len(truth) == 3  # Apple, Microsoft, NVIDIA

    answer = await sec_corpus.ask(
        "For each filing, report revenue, net income, gross profit, operating income, "
        "total assets, net profit margin, gross margin, current ratio, debt to equity, "
        "and return on equity."
    )
    report = evaluate_grid(answer.grid, truth)

    # Every reported figure must match the filing exactly, and every cited span must
    # actually contain its value — on real 10-Ks, not synthetic data.
    assert report.cell_numeric_accuracy == 1.0, report.details
    assert report.attribution_correctness == 1.0
    assert report.grid_completion >= 0.85
    assert report.numeric_cells_scored >= 30
