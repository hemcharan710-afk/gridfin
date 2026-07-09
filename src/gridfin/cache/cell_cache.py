"""Cell-level caching.

A cell is a pure function of (document, sub-question), so repeated pairs are free.
Re-running a grid after one column changes recomputes only that column, which
matters a lot once a grid issues many model calls.

The signature deliberately includes the *content* of the column spec (question,
formula, type), so editing a column's question correctly invalidates only that
column's cells, while leaving every other cell a cache hit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from diskcache import Cache

from gridfin.config import get_settings
from gridfin.models import Cell, ColumnSpec


def cell_signature(doc_id: str, column: ColumnSpec) -> str:
    payload = {
        "doc_id": doc_id,
        "question": column.question,
        "type": column.type,
        "formula": column.formula,
        "depends_on": sorted(column.depends_on),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class CellCache:
    def __init__(self, namespace: str = "cells"):
        settings = get_settings()
        self._cache = Cache(str(settings.cache_dir / namespace))
        self.hits = 0
        self.misses = 0

    def get(self, doc_id: str, column: ColumnSpec) -> Optional[Cell]:
        sig = cell_signature(doc_id, column)
        raw = self._cache.get(sig)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        cell = Cell.model_validate_json(raw)
        return cell

    def put(self, column: ColumnSpec, cell: Cell) -> None:
        if cell.status not in ("done", "empty"):
            return  # never cache transient failures
        sig = cell_signature(cell.doc_id, column)
        self._cache.set(sig, cell.model_dump_json())

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def clear(self) -> None:
        self._cache.clear()
