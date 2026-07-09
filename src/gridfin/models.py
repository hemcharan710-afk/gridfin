"""The grid data model.

The cell is the unit of both work and audit. Parallelism, isolation, caching, and
traceability all fall out of this one structure.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# --- Enumerations ----------------------------------------------------------

CellStatus = Literal["pending", "running", "done", "failed", "empty"]
# The three routes a cell can take.
CellPath = Literal["numeric_store", "retrieval_extract", "compute"]
ColumnType = Literal["numeric", "text", "ratio"]


# --- Provenance ------------------------------------------------------------

class Source(BaseModel):
    """Exactly one source span backs every non-empty cell (audit-first)."""

    file: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_span: Optional[tuple[int, int]] = None

    def short(self) -> str:
        loc = f"p.{self.page}" if self.page is not None else ""
        sec = f" · {self.section}" if self.section else ""
        return f"{loc}{sec}".strip(" ·") or self.file


# --- The cell --------------------------------------------------------------

class Cell(BaseModel):
    """A single (document, sub-question) unit of work and audit.

    A cell is a pure function of (doc_id, column_id), which is why it caches
    cleanly.
    """

    doc_id: str
    column_id: str
    question: str

    status: CellStatus = "pending"
    value: Any = None
    unit: str = ""
    detail: Optional[str] = None  # e.g. the compute expression "4,200 / 33,900"
    source: Optional[Source] = None
    confidence: float = 0.0
    path: Optional[CellPath] = None
    cost_tokens: int = 0
    error: Optional[str] = None

    def display(self) -> str:
        """Human-facing rendering of the cell value (figures, percentages, ratios)."""
        if self.status == "empty":
            return "—"
        if self.status == "failed":
            return "refused"
        if self.value is None:
            return ""
        if isinstance(self.value, float):
            if self.unit == "%":
                return f"{self.value * 100:.1f}%"
            if self.unit == "x":
                return f"{self.value:.2f}x"
            if self.unit in ("USD", "$"):
                return _fmt_money(self.value)
            return f"{self.value:,.2f}".rstrip("0").rstrip(".")
        return str(self.value)

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_id, self.column_id)


def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.1f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.0f}M"
    if a >= 1e3:
        return f"{sign}${a / 1e3:,.0f}K"
    return f"{sign}${a:,.0f}"


# --- The plan (output of decomposition) ------------------------------------

class ColumnSpec(BaseModel):
    """One sub-question / field — a column of the grid."""

    column_id: str
    name: str = Field(..., description="Short header shown in the grid UI.")
    question: str = Field(..., description="The scoped sub-question for each cell.")
    type: ColumnType = "text"
    # For ratio columns: the deterministic formula and which other columns feed it.
    formula: Optional[str] = None
    depends_on: list[str] = Field(default_factory=list)


class DocumentRef(BaseModel):
    """A row of the grid — one document in scope."""

    doc_id: str
    title: str = ""


class GridPlan(BaseModel):
    """Structured output of the planner: the columns and rows of the grid."""

    question: str
    columns: list[ColumnSpec]
    rows: list[DocumentRef]

    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.columns))


# --- The grid (rows × columns of cells) ------------------------------------

class Grid(BaseModel):
    """Grid = rows(documents) × columns(sub-questions) — a 2-D array of Cells."""

    question: str
    columns: list[ColumnSpec]
    rows: list[DocumentRef]
    cells: dict[str, Cell] = Field(default_factory=dict)
    narrative: Optional[str] = None
    verification: list["VerificationNote"] = Field(default_factory=list)

    @staticmethod
    def _ck(doc_id: str, column_id: str) -> str:
        return f"{doc_id}::{column_id}"

    @classmethod
    def from_plan(cls, plan: GridPlan) -> "Grid":
        grid = cls(question=plan.question, columns=plan.columns, rows=plan.rows)
        for row in plan.rows:
            for col in plan.columns:
                cell = Cell(doc_id=row.doc_id, column_id=col.column_id, question=col.question)
                grid.cells[cls._ck(row.doc_id, col.column_id)] = cell
        return grid

    def get(self, doc_id: str, column_id: str) -> Cell:
        return self.cells[self._ck(doc_id, column_id)]

    def set(self, cell: Cell) -> None:
        self.cells[self._ck(cell.doc_id, cell.column_id)] = cell

    def column(self, column_id: str) -> list[Cell]:
        return [self.get(r.doc_id, column_id) for r in self.rows]

    def row(self, doc_id: str) -> list[Cell]:
        return [self.get(doc_id, c.column_id) for c in self.columns]

    def column_spec(self, column_id: str) -> ColumnSpec:
        return next(c for c in self.columns if c.column_id == column_id)

    def completion(self) -> float:
        """Grid-completion metric: fraction of cells filled, not failed/empty."""
        if not self.cells:
            return 0.0
        done = sum(1 for c in self.cells.values() if c.status == "done")
        return done / len(self.cells)


class VerificationNote(BaseModel):
    """One finding from cross-cell verification."""

    kind: Literal["column_consistency", "row_consistency", "citation_match"]
    level: Literal["ok", "warning", "error"]
    message: str
    cells: list[tuple[str, str]] = Field(default_factory=list)


Grid.model_rebuild()
