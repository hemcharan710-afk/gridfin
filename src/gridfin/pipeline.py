"""The pipeline, in execution order.

The front half (ingest -> chunk -> numeric store -> dual index) is built once per
corpus and reused. Each query then runs the grid: decompose -> fan-out -> verify
-> synthesize.

    corpus = GridFin.from_path("data/demo")
    answer = await corpus.ask("Compare net margin across these filings")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from gridfin.cache import CellCache
from gridfin.chunking import chunk_document
from gridfin.extraction import NumericStore, build_numeric_store
from gridfin.grid import CellEngine, decompose, synthesize, verify_grid
from gridfin.grid.cell_engine import CellContext
from gridfin.indexing import DualIndex
from gridfin.ingestion import ParsedDocument, ingest_path
from gridfin.llm import LLM, get_llm
from gridfin.models import Cell, Grid
from gridfin.retrieval import Retriever


@dataclass
class GridAnswer:
    grid: Grid
    narrative: str

    @property
    def completion(self) -> float:
        return self.grid.completion()


class GridFin:
    """A built corpus: the reusable front half of the pipeline."""

    def __init__(
        self,
        docs: list[ParsedDocument],
        *,
        llm: Optional[LLM] = None,
        cache_namespace: str = "cells",
    ):
        self.docs = docs
        self.docs_by_id = {d.doc_id: d for d in docs}
        self.llm = llm or get_llm()

        # Chunking.
        self.chunks = [c for d in docs for c in chunk_document(d)]
        # Structured extraction -> numeric store (the cell fast path).
        self.numeric_store: NumericStore = build_numeric_store(docs)
        # Dual index + per-document filter.
        self.index = DualIndex()
        self.index.build(self.chunks)
        # Scoped retrieval & reranking.
        self.retriever = Retriever(self.index)
        # Cell cache.
        self.cache = CellCache(cache_namespace)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_path(cls, path: str | Path, **kw) -> "GridFin":
        # Format-aware ingestion.
        docs = ingest_path(path)
        return cls(docs, **kw)

    # -- query -------------------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        doc_ids: Optional[list[str]] = None,
        on_cell: Optional[Callable[[Cell], None]] = None,
        synthesize_answer: bool = True,
    ) -> GridAnswer:
        scope = (
            [self.docs_by_id[d] for d in doc_ids if d in self.docs_by_id]
            if doc_ids
            else self.docs
        )

        # Decompose the question into a grid plan, then build the empty grid.
        plan = await decompose(question, scope, self.llm)
        grid = Grid.from_plan(plan)

        # Fan out: run every cell.
        ctx = CellContext(
            numeric_store=self.numeric_store,
            retriever=self.retriever,
            llm=self.llm,
            cache=self.cache,
            docs=self.docs_by_id,
        )
        engine = CellEngine(ctx)
        if on_cell:
            engine.on_cell = on_cell
        await engine.execute_grid(grid)

        # Cross-cell verification, then synthesize the narrative.
        verify_grid(grid, self.numeric_store)
        narrative = await synthesize(grid, self.llm) if synthesize_answer else ""
        return GridAnswer(grid=grid, narrative=narrative)

    # -- introspection -----------------------------------------------------
    def info(self) -> dict:
        return {
            "documents": len(self.docs),
            "chunks": len(self.chunks),
            "numeric_facts": len(self.numeric_store),
            "embedding_backend": self.index.backend,
            "reranker_backend": self.retriever.reranker_backend,
            "llm": "live" if self.llm.live else "mock",
        }
