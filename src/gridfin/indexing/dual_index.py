"""Dual index (dense + sparse) with a per-document filter.

BGE -> FAISS for dense retrieval and BM25 for sparse, with a hard per-document
filter so a cell only ever retrieves within one document. That filter is what
keeps cells isolated: Company A's text can never leak into Company B's answer.

FAISS is used when available; otherwise a brute-force numpy cosine search stands
in. Either way the per-doc filter is enforced *before* scoring, not after.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from gridfin.chunking import Chunk
from gridfin.config import get_settings
from gridfin.indexing.embeddings import Embedder, _tokens


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


class DualIndex:
    """Dense (embeddings) + sparse (BM25), both filterable to a single document."""

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.chunks: list[Chunk] = []
        self._doc_rows: dict[str, list[int]] = {}  # doc_id -> row indices
        self._dense: Optional[np.ndarray] = None
        self._faiss: Optional[Any] = None  # faiss.IndexFlatIP over all chunk vectors
        self.ann_backend = "numpy"
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: list[list[str]] = []

    @property
    def backend(self) -> str:
        return self.embedder.backend

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._doc_rows = {}
        for i, c in enumerate(self.chunks):
            self._doc_rows.setdefault(c.doc_id, []).append(i)
        texts = [c.text for c in self.chunks]
        self._dense = self.embedder.encode(texts) if texts else np.zeros((0, self.embedder.dim))
        self._faiss = self._build_faiss(self._dense)
        self._tokenized = [_tokens(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def _build_faiss(self, dense: np.ndarray) -> Optional[Any]:
        """A FAISS inner-product index over the (L2-normalised) chunk vectors.

        Embeddings are unit-normalised, so inner product == cosine. Built only when
        faiss is installed and there are vectors to index; otherwise the numpy path
        below stands in. Row id in the index == global chunk row, so the per-document
        filter can be expressed as an id selector at search time.
        """
        if dense is None or dense.shape[0] == 0 or not get_settings().has_faiss:
            return None
        try:
            import faiss

            index = faiss.IndexFlatIP(dense.shape[1])
            index.add(np.ascontiguousarray(dense, dtype=np.float32))
            self.ann_backend = "faiss"
            return index
        except Exception:  # pragma: no cover - faiss import/runtime issues
            return None

    def _dense_scores(self, query: str, rows: list[int], k: int) -> dict[int, float]:
        if self._dense is None or not rows:
            return {}
        q = self.embedder.encode([query])[0].astype(np.float32)

        # FAISS path: restrict the search to this document's rows with an id selector,
        # so scoring still happens *only* within the one document (isolation preserved).
        if self._faiss is not None:
            import faiss

            # `row_ids` must outlive the search call — IDSelectorBatch holds a
            # pointer to the buffer, not a Python reference to the array.
            row_ids = np.asarray(rows, dtype=np.int64)
            sel = faiss.IDSelectorBatch(row_ids)
            params = faiss.SearchParametersFlat(sel=sel)
            sims, idx = self._faiss.search(q.reshape(1, -1), min(k, len(rows)), params=params)
            return {int(r): float(s) for r, s in zip(idx[0], sims[0]) if r != -1}

        # numpy fallback: hard per-doc filter applied before scoring.
        sub = self._dense[rows]
        sims = sub @ q
        return {row: float(s) for row, s in zip(rows, sims)}

    def _sparse_scores(self, query: str, rows: list[int]) -> dict[int, float]:
        if self._bm25 is None or not rows:
            return {}
        scores = self._bm25.get_scores(_tokens(query))
        return {row: float(scores[row]) for row in rows}

    def search(
        self, query: str, *, doc_id: str, k: int = 8
    ) -> tuple[list[tuple[Chunk, float]], list[tuple[Chunk, float]]]:
        """Return (dense_hits, sparse_hits) within a single document.

        `doc_id` is mandatory — there is no way to search across documents. Returns
        ranked (chunk, score) lists for each modality, to be fused downstream (RRF).
        """
        rows = self._doc_rows.get(doc_id, [])
        dense = self._dense_scores(query, rows, k)
        sparse = self._sparse_scores(query, rows)

        dense_hits = sorted(dense.items(), key=lambda kv: kv[1], reverse=True)[:k]
        sparse_hits = sorted(sparse.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return (
            [(self.chunks[i], s) for i, s in dense_hits],
            [(self.chunks[i], s) for i, s in sparse_hits],
        )

    def documents(self) -> list[str]:
        return list(self._doc_rows.keys())
