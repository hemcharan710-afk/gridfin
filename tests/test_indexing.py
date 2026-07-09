"""Dual indexing per-document filter (§04 Layer 4) — the isolation guarantee."""

from gridfin.chunking import chunk_document
from gridfin.indexing import DualIndex
from gridfin.ingestion.ingest import Block, ParsedDocument
from gridfin.retrieval import Retriever


def _doc(doc_id: str, text: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id, filename=f"{doc_id}.txt", filetype="txt",
        blocks=[Block(text=text, kind="prose")],
    )


def test_retrieval_never_crosses_documents():
    a = _doc("A", "Acme robotics revenue grew strongly in fiscal 2023.")
    b = _doc("B", "Beta consumer products revenue was flat in fiscal 2023.")
    chunks = chunk_document(a) + chunk_document(b)
    index = DualIndex()
    index.build(chunks)

    hits = index.search("revenue", doc_id="A", k=5)
    for modality in hits:
        for chunk, _score in modality:
            assert chunk.doc_id == "A"  # hard per-doc filter — no contamination


def test_retriever_scoped_to_single_doc():
    a = _doc("A", "Acme robotics revenue.")
    b = _doc("B", "Beta revenue.")
    index = DualIndex()
    index.build(chunk_document(a) + chunk_document(b))
    r = Retriever(index)
    scored = r.retrieve("revenue", doc_id="B")
    assert all(s.chunk.doc_id == "B" for s in scored)
