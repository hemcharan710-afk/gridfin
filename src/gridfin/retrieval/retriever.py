"""Scoped retrieval and reranking.

RRF fusion followed by a CrossEncoder rerank, run per cell with a mandatory
document filter rather than once over a blended context. It is a subroutine of the
per-cell execution engine.

The CrossEncoder is used when available; otherwise the fused RRF order stands.
"""

from __future__ import annotations

from typing import Optional

from gridfin.chunking import Chunk
from gridfin.config import get_settings
from gridfin.indexing import DualIndex, ScoredChunk


class Retriever:
    def __init__(self, index: DualIndex):
        self.index = index
        self.settings = get_settings()
        self._reranker = None
        if self.settings.has_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:  # pragma: no cover
                self._reranker = None

    @property
    def reranker_backend(self) -> str:
        return "cross-encoder" if self._reranker is not None else "rrf-only"

    def _rrf(
        self,
        dense: list[tuple[Chunk, float]],
        sparse: list[tuple[Chunk, float]],
    ) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion across the two modalities."""
        k = self.settings.rrf_k
        fused: dict[str, ScoredChunk] = {}
        for rank, (chunk, _) in enumerate(dense):
            sc = fused.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk, score=0.0))
            sc.score += 1.0 / (k + rank + 1)
            sc.dense_rank = rank
        for rank, (chunk, _) in enumerate(sparse):
            sc = fused.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk, score=0.0))
            sc.score += 1.0 / (k + rank + 1)
            sc.sparse_rank = rank
        return sorted(fused.values(), key=lambda s: s.score, reverse=True)

    def _rerank(self, query: str, fused: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not fused:
            return []
        if self._reranker is None:
            return fused[:top_n]
        pairs = [(query, sc.chunk.text) for sc in fused]
        scores = self._reranker.predict(pairs)
        for sc, s in zip(fused, scores):
            sc.score = float(s)
        return sorted(fused, key=lambda s: s.score, reverse=True)[:top_n]

    def retrieve(
        self, query: str, *, doc_id: str, k: Optional[int] = None, top_n: Optional[int] = None
    ) -> list[ScoredChunk]:
        """Hard-filtered to a single document. The cell's window onto its source."""
        k = k or self.settings.retrieval_k
        top_n = top_n or self.settings.rerank_top_n
        dense, sparse = self.index.search(query, doc_id=doc_id, k=k)
        fused = self._rrf(dense, sparse)
        return self._rerank(query, fused, top_n)
