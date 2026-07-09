"""Evaluation additions (§09)."""

import json

from gridfin.eval import evaluate_grid


async def test_demo_metrics(corpus, demo_dir):
    truth = json.loads((demo_dir / "ground_truth.json").read_text())
    truth = {k: v for k, v in truth.items() if not k.startswith("_")}
    answer = await corpus.ask(
        "For each filing report revenue, net income, net profit margin, current "
        "ratio, debt to equity and return on equity."
    )
    report = evaluate_grid(answer.grid, truth)

    assert report.cell_numeric_accuracy == 1.0  # the headline number must be exact
    assert report.attribution_correctness == 1.0
    assert report.grid_completion >= 0.85
    assert report.numeric_cells_scored > 0
