"""Grid store: flatten Pydantic cells into DuckDB.

The filled grid is written one row per cell so it can be queried with SQL (by
route, confidence, or document) and exported.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from gridfin.config import get_settings
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


def persist_grid(grid: Grid, db_path: Path | None = None, table: str = "cells") -> Path:
    db_path = db_path or get_settings().grid_db_path
    df = grid_to_dataframe(grid)  # noqa: F841 — referenced by DuckDB below
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df LIMIT 0")
        con.execute(f"DELETE FROM {table} WHERE document IN (SELECT DISTINCT document FROM df)")
        con.execute(f"INSERT INTO {table} SELECT * FROM df")
    finally:
        con.close()
    return Path(db_path)


def export_csv(grid: Grid, path: str | Path) -> Path:
    df = grid_to_dataframe(grid)
    path = Path(path)
    df.to_csv(path, index=False)
    return path
