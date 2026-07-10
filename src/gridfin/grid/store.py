"""Grid store: flatten Pydantic cells into a table and export it.

The filled grid is turned into one row per cell (document, column, value, route,
confidence, source) and written to CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridfin.models import Grid


def grid_to_dataframe(grid: Grid) -> pd.DataFrame:
    records = []
    for row in grid.rows:
        for col in grid.columns:
            cell = grid.get(row.doc_id, col.column_id)
            records.append(
                {
                    "doc_id": cell.doc_id,
                    "document": row.title or row.doc_id,
                    "column_id": cell.column_id,
                    "column": col.name,
                    "type": col.type,
                    "status": cell.status,
                    "value": cell.display(),
                    "raw_value": cell.value if isinstance(cell.value, (int, float)) else None,
                    "unit": cell.unit,
                    "detail": cell.detail,
                    "path": cell.path,
                    "confidence": round(cell.confidence, 3),
                    "source": cell.source.short() if cell.source else None,
                    "file": cell.source.file if cell.source else None,
                    "cost_tokens": cell.cost_tokens,
                    "error": cell.error,
                }
            )
    return pd.DataFrame.from_records(records)


def export_csv(grid: Grid, path: str | Path) -> Path:
    df = grid_to_dataframe(grid)
    path = Path(path)
    df.to_csv(path, index=False)
    return path
