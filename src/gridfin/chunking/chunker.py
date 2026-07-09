"""Chunking: keep tables whole, split prose with a sliding window.

Every chunk carries metadata, including the `doc_id` that the per-document
retrieval filter depends on.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from gridfin.ingestion import ParsedDocument


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    kind: str = "prose"
    page: Optional[int] = None
    section: Optional[str] = None
    char_span: tuple[int, int] = (0, 0)


def _window(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Sliding window over whitespace-delimited tokens, returned with char spans."""
    if len(text) <= size:
        return [(0, len(text), text)]
    out: list[tuple[int, int, str]] = []
    step = max(1, size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        # Prefer to break on a sentence/whitespace boundary near `end`.
        if end < len(text):
            boundary = text.rfind(" ", start + step, end)
            if boundary != -1:
                end = boundary
        out.append((start, end, text[start:end].strip()))
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return [(s, e, t) for (s, e, t) in out if t]


def chunk_document(doc: ParsedDocument, *, window: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for bi, block in enumerate(doc.blocks):
        if block.kind == "table":
            # Tables are kept whole — never windowed.
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:b{bi}",
                    doc_id=doc.doc_id,
                    text=block.text,
                    kind="table",
                    page=block.page,
                    section=block.section,
                    char_span=block.char_span,
                )
            )
            continue
        base = block.char_span[0]
        for wi, (s, e, text) in enumerate(_window(block.text, window, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:b{bi}:w{wi}",
                    doc_id=doc.doc_id,
                    text=text,
                    kind="prose",
                    page=block.page,
                    section=block.section,
                    char_span=(base + s, base + e),
                )
            )
    return chunks
