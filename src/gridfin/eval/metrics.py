"""Evaluation metrics.

RAGAS covers retrieval quality (faithfulness / relevancy / precision / recall) and
is wired behind the optional [eval] extra. The grid adds three metrics of its own,
including the one that matters most for a financial tool and that RAGAS does not
measure: whether the number is actually right.

  - Cell Numeric Accuracy   — fraction of numeric cells equal to ground truth.
  - Grid Completion         — fraction of cells filled, not failed/empty.
  - Attribution Correctness — whether each cell's cited source contains its value.
"""

from __future__ import annotations

from pydantic import BaseModel

from gridfin.extraction import extract_figures
from gridfin.models import Grid

# Ground truth shape: {doc_id: {column_id: expected_number}}
GroundTruth = dict[str, dict[str, float]]


class GridEvalReport(BaseModel):
    cell_numeric_accuracy: float
    grid_completion: float
    attribution_correctness: float
    numeric_cells_scored: int
    details: list[str] = []


def grid_completion(grid: Grid) -> float:
    """Fraction of cells successfully filled rather than failed or empty."""
    return grid.completion()


def cell_numeric_accuracy(grid: Grid, truth: GroundTruth, tol: float = 0.01) -> tuple[float, int, list[str]]:
    """Fraction of numeric cells whose value equals ground truth (within `tol`)."""
    scored = 0
    correct = 0
    misses: list[str] = []
    for doc_id, cols in truth.items():
        for col_id, expected in cols.items():
            try:
                cell = grid.get(doc_id, col_id)
            except (KeyError, StopIteration):
                continue
            scored += 1
            got = cell.value if isinstance(cell.value, (int, float)) else None
            if got is not None and abs(got - expected) <= max(tol, abs(expected) * tol):
                correct += 1
            else:
                misses.append(f"{doc_id}/{col_id}: expected {expected}, got {got}")
    return (correct / scored if scored else 0.0), scored, misses


def attribution_correctness(grid: Grid) -> float:
    """Whether each done cell's cited source span actually contains its value."""
    relevant = [
        c for c in grid.cells.values()
        if c.status == "done" and c.path != "compute" and isinstance(c.value, (int, float))
    ]
    if not relevant:
        return 1.0
    ok = 0
    for c in relevant:
        if not c.detail:
            continue
        target = abs(c.value)
        if any(_close(v, target) for v in extract_figures(c.detail)):
            ok += 1
    return ok / len(relevant)


def _close(value: float, target: float) -> bool:
    return abs(abs(value) - target) <= max(1.0, target * 0.01)


def evaluate_grid(grid: Grid, truth: GroundTruth | None = None) -> GridEvalReport:
    completion = grid_completion(grid)
    attribution = attribution_correctness(grid)
    if truth:
        accuracy, scored, misses = cell_numeric_accuracy(grid, truth)
    else:
        accuracy, scored, misses = (float("nan"), 0, [])
    return GridEvalReport(
        cell_numeric_accuracy=accuracy,
        grid_completion=completion,
        attribution_correctness=attribution,
        numeric_cells_scored=scored,
        details=misses,
    )
