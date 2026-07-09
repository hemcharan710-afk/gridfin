"""Per-cell execution engine (the fan-out).

Each cell is a (document, sub-question) pair. Cells run concurrently; each checks
the cell cache, runs scoped retrieval, then takes one of three routes:

  1. numeric_store    — pull a pre-extracted figure (no LLM, the fast path)
  2. retrieval_extract — extract from its retrieved span (scoped to one document)
  3. compute          — call the deterministic calc engine with figures from peers

Concurrency is bounded by a semaphore, with tenacity retries, so a grid that
issues many model calls stays manageable.

Ratio (compute) cells depend on their input columns, so the grid is filled in two
dependency-respecting waves: figures first, then the ratios that consume them.
Both waves are fully parallel internally.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from gridfin.calc import CalcError, compute, supported_formulas
from gridfin.cache import CellCache
from gridfin.config import get_settings
from gridfin.extraction import NumericStore, scan_value
from gridfin.ingestion import ParsedDocument
from gridfin.llm import LLM
from gridfin.models import Cell, ColumnSpec, Grid, Source
from gridfin.retrieval import Retriever


class CellExtraction(BaseModel):
    """Structured output for the retrieval-extract route."""

    found: bool = False
    value: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CellContext:
    numeric_store: NumericStore
    retriever: Retriever
    llm: LLM
    cache: CellCache
    docs: dict[str, ParsedDocument] = field(default_factory=dict)


class _TransientCellError(Exception):
    """Retryable failure inside a cell (e.g. a flaky model call)."""


# Anthropic SDK error classes that represent *transient* failures worth retrying.
# Matched by class name so this module keeps no hard dependency on `anthropic`
# (offline/mock mode must import with the SDK absent).
_RETRYABLE_API_ERRORS = {
    "RateLimitError",        # 429
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",   # 500
    "OverloadedError",       # 529
    "APIStatusError",        # generic — gated on status_code below
}


def _is_retryable(exc: BaseException) -> bool:
    """Retry our own transient marker plus 429/5xx API errors — but never 4xx
    client errors (bad request, auth), which retrying cannot fix."""
    if isinstance(exc, _TransientCellError):
        return True
    if exc.__class__.__name__ in _RETRYABLE_API_ERRORS:
        status = getattr(exc, "status_code", None)
        if status is None:
            return True
        return status == 429 or status >= 500
    return False


class CellEngine:
    def __init__(self, ctx: CellContext):
        self.ctx = ctx
        self.settings = get_settings()
        self._sem = asyncio.Semaphore(self.settings.max_concurrency)
        self.on_cell = None  # optional callback(cell) for live UI streaming

    # -- public ------------------------------------------------------------
    async def execute_grid(self, grid: Grid) -> Grid:
        ratio_cols = [c for c in grid.columns if c.type == "ratio"]
        figure_cols = [c for c in grid.columns if c.type != "ratio"]

        # Wave 1: every figure/text cell, fully parallel.
        await self._run_wave(grid, figure_cols)
        # Wave 2: ratio cells, which read the now-filled figure cells.
        await self._run_wave(grid, ratio_cols)
        return grid

    # -- waves -------------------------------------------------------------
    async def _run_wave(self, grid: Grid, columns: list[ColumnSpec]) -> None:
        tasks = []
        for col in columns:
            for row in grid.rows:
                tasks.append(self._guarded(grid, row.doc_id, col))
        if tasks:
            await asyncio.gather(*tasks)

    async def _guarded(self, grid: Grid, doc_id: str, col: ColumnSpec) -> None:
        async with self._sem:  # concurrency cap (semaphore) — protects rate limits
            cell = await self._execute_cell(grid, doc_id, col)
        grid.set(cell)
        if self.on_cell:
            self.on_cell(cell)

    # -- single cell -------------------------------------------------------
    async def _execute_cell(self, grid: Grid, doc_id: str, col: ColumnSpec) -> Cell:
        # Cache: a cell is a pure function of (doc, sub-question).
        cached = self.ctx.cache.get(doc_id, col)
        if cached is not None:
            return cached

        cell = Cell(doc_id=doc_id, column_id=col.column_id, question=col.question, status="running")
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.settings.cell_max_retries),
                wait=wait_exponential(multiplier=0.2, max=2.0),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    if col.type == "ratio":
                        cell = await self._route_compute(grid, doc_id, col)
                    elif col.type == "numeric":
                        cell = await self._route_numeric(doc_id, col)
                    else:
                        cell = await self._route_text(doc_id, col)
        except Exception as e:  # terminal failure — recorded, never invented
            cell.status = "failed"
            cell.error = str(e)

        self.ctx.cache.put(col, cell)
        return cell

    # -- route 1 + 2: numeric (store fast-path, else retrieval-extract) ----
    async def _route_numeric(self, doc_id: str, col: ColumnSpec) -> Cell:
        metric = col.column_id
        # Fast path: pre-extracted figure, no LLM.
        fact = self.ctx.numeric_store.get(doc_id, metric)
        if fact is not None:
            return Cell(
                doc_id=doc_id,
                column_id=col.column_id,
                question=col.question,
                status="done",
                value=fact.value,
                unit="USD",
                detail=fact.raw,
                source=Source(file=self._file(doc_id), page=fact.page, section=fact.section, char_span=fact.char_span),
                confidence=fact.confidence,
                path="numeric_store",
            )

        # Fallback: scoped retrieval, then extract the figure from a span.
        hits = self.ctx.retriever.retrieve(col.question, doc_id=doc_id)
        for hit in hits:
            found = scan_value(hit.chunk.text, metric)
            if found is not None:
                value, raw = found
                return Cell(
                    doc_id=doc_id,
                    column_id=col.column_id,
                    question=col.question,
                    status="done",
                    value=value,
                    unit="USD",
                    detail=raw,
                    source=self._source(doc_id, hit.chunk),
                    # Confidence reflects the retrieval score of the span the figure
                    # came from, floored so any successful extract stays trusted.
                    confidence=min(0.95, 0.6 + max(0.0, hit.score)),
                    path="retrieval_extract",
                )
        # A missing figure is returned empty and flagged, never invented.
        return Cell(
            doc_id=doc_id,
            column_id=col.column_id,
            question=col.question,
            status="empty",
            path="retrieval_extract",
            confidence=0.0,
        )

    # -- route 2: text (retrieval-extract, qualitative) --------------------
    async def _route_text(self, doc_id: str, col: ColumnSpec) -> Cell:
        hits = self.ctx.retriever.retrieve(col.question, doc_id=doc_id)
        if not hits:
            return Cell(doc_id=doc_id, column_id=col.column_id, question=col.question, status="empty", path="retrieval_extract")
        top = hits[0]
        snippet = top.chunk.text.strip().replace("\n", " ")
        mock = CellExtraction(found=True, value=snippet[:240], confidence=min(0.9, 0.6 + top.score))
        prompt = (
            f"From the SOURCE below (a single document), answer concisely.\n\n"
            f"QUESTION: {col.question}\n\nSOURCE:\n{snippet[:1500]}\n\n"
            "If the source does not contain the answer, set found=false. Never use "
            "outside knowledge."
        )
        ext, tokens = await self.ctx.llm.structured(
            prompt, CellExtraction, tier="small", context={"mock": mock.model_dump()}
        )
        if not ext.found or not ext.value:
            return Cell(doc_id=doc_id, column_id=col.column_id, question=col.question, status="empty", path="retrieval_extract", cost_tokens=tokens)
        return Cell(
            doc_id=doc_id,
            column_id=col.column_id,
            question=col.question,
            status="done",
            value=ext.value,
            source=self._source(doc_id, top.chunk),
            confidence=ext.confidence,
            path="retrieval_extract",
            cost_tokens=tokens,
        )

    # -- route 3: compute (deterministic calc engine) ----------------------
    async def _route_compute(self, grid: Grid, doc_id: str, col: ColumnSpec) -> Cell:
        if not col.formula or col.formula not in [*supported_formulas(), *_aliases()]:
            return Cell(doc_id=doc_id, column_id=col.column_id, question=col.question, status="failed", path="compute", error=f"unsupported formula '{col.formula}'")

        # The calc engine expects specific parameter names. For most ratios the
        # dependency column ids already match (net_income, revenue, ...). For
        # positional formulas (cagr -> begin/end/years, yoy_growth -> current/prior)
        # the column ids are arbitrary, so map them by position onto the formula's
        # declared inputs.
        params = _formula_inputs(col.formula)
        inputs: dict[str, object] = {}
        missing: list[str] = []
        for pos, dep in enumerate(col.depends_on):
            dep_cell = grid.get(doc_id, dep)
            if dep_cell.status == "done" and isinstance(dep_cell.value, (int, float)):
                if dep in params:
                    param = dep
                elif pos < len(params):
                    param = params[pos]
                else:
                    param = dep
                inputs[param] = float(dep_cell.value)
            else:
                missing.append(dep)

        if missing:
            # Depends on a missing input, so it is refused rather than guessed.
            return Cell(
                doc_id=doc_id,
                column_id=col.column_id,
                question=col.question,
                status="failed",
                path="compute",
                error=f"refused: missing input(s) {missing}",
            )

        try:
            result = compute(col.formula, inputs)
        except CalcError as e:
            return Cell(doc_id=doc_id, column_id=col.column_id, question=col.question, status="failed", path="compute", error=str(e))

        # Provenance: the compute cell traces to its input cells' sources.
        src = next((grid.get(doc_id, d).source for d in col.depends_on if grid.get(doc_id, d).source), None)
        return Cell(
            doc_id=doc_id,
            column_id=col.column_id,
            question=col.question,
            status="done",
            value=result.result,
            unit=result.unit,
            detail=result.expression,
            source=src,
            confidence=1.0,  # deterministic
            path="compute",
        )

    # -- helpers -----------------------------------------------------------
    def _file(self, doc_id: str) -> str:
        doc = self.ctx.docs.get(doc_id)
        return doc.filename if doc else doc_id

    def _source(self, doc_id: str, chunk) -> Source:
        return Source(
            file=self._file(doc_id),
            page=chunk.page,
            section=chunk.section,
            char_span=chunk.char_span,
        )


def _formula_inputs(formula: str) -> tuple[str, ...]:
    """The ordered parameter names a formula expects, resolving aliases."""
    from gridfin.calc import FORMULAS

    spec = FORMULAS.get(formula) or FORMULAS.get((formula or "").lower())
    return spec.inputs if spec else ()


def _aliases() -> list[str]:
    from gridfin.calc import FORMULAS

    return list(FORMULAS.keys())
