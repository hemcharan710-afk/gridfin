"""Format-aware ingestion.

Route by file type and parse tables and prose into one shared metadata schema.
Real PDF/DOCX/XLSX parsing sits behind optional dependencies, so the bundled
text/CSV demo runs with no native libraries installed.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

BlockKind = Literal["prose", "table"]


class Table(BaseModel):
    """A whole table, kept intact (chunking never splits tables)."""

    rows: list[list[str]]
    page: Optional[int] = None
    section: Optional[str] = None
    confidence: float = 1.0  # table-parse confidence

    def to_text(self) -> str:
        return "\n".join("\t".join(str(c) for c in r) for r in self.rows)


class Block(BaseModel):
    """The unified ingestion unit: one span of prose or one table, with metadata."""

    text: str
    kind: BlockKind = "prose"
    page: Optional[int] = None
    section: Optional[str] = None
    char_span: tuple[int, int] = (0, 0)
    table: Optional[Table] = None


class ParsedDocument(BaseModel):
    doc_id: str
    filename: str
    filetype: str
    title: str = ""
    blocks: list[Block] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def _doc_id(path: Path) -> str:
    # Hash the filename (not the absolute path) so ids are reproducible across
    # machines and runs — the evaluation ground truth references them directly.
    h = hashlib.sha1(path.name.encode()).hexdigest()[:8]
    return f"{path.stem}-{h}"


def _section_for(line: str) -> Optional[str]:
    """Detect lightweight section headers (markdown `#`, ALL CAPS, or `Item N.`)."""
    s = line.strip()
    if not s:
        return None
    if s.startswith("#"):
        return s.lstrip("# ").strip()
    if s.lower().startswith("item ") and len(s) < 80:
        return s
    if 3 < len(s) < 70 and s == s.upper() and any(ch.isalpha() for ch in s):
        return s.title()
    return None


# --- text / markdown -------------------------------------------------------

def _parse_text(path: Path, raw: str) -> list[Block]:
    blocks: list[Block] = []
    section: Optional[str] = None
    cursor = 0
    # Split on blank lines into paragraph-sized prose blocks; track sections.
    for para in raw.split("\n\n"):
        start = raw.find(para, cursor)
        cursor = start + len(para)
        stripped = para.strip()
        if not stripped:
            continue
        first = stripped.splitlines()[0]
        detected = _section_for(first)
        if detected:
            section = detected
        blocks.append(
            Block(
                text=stripped,
                kind="prose",
                section=section,
                char_span=(start, start + len(para)),
            )
        )
    return blocks


# --- csv -------------------------------------------------------------------

def _parse_csv(path: Path, raw: str) -> list[Block]:
    reader = csv.reader(io.StringIO(raw))
    rows = [row for row in reader if any(c.strip() for c in row)]
    if not rows:
        return []
    table = Table(rows=rows, section=path.stem, confidence=1.0)
    return [Block(text=table.to_text(), kind="table", section=path.stem, table=table)]


# --- pdf (optional) --------------------------------------------------------

def _parse_pdf(path: Path) -> list[Block]:
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover - exercised only with [ingest] extra
        raise RuntimeError(
            "PDF ingestion needs the optional dependency: pip install 'gridfin[ingest]'"
        ) from e
    blocks: list[Block] = []
    offset = 0
    with pdfplumber.open(str(path)) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables() or []:
                rows = [[c or "" for c in row] for row in table]
                t = Table(rows=rows, page=pageno, confidence=0.9)
                blocks.append(Block(text=t.to_text(), kind="table", page=pageno, table=t))
            text = page.extract_text() or ""
            if text.strip():
                blocks.append(
                    Block(
                        text=text,
                        kind="prose",
                        page=pageno,
                        char_span=(offset, offset + len(text)),
                    )
                )
                offset += len(text)
    return blocks


# --- docx (optional) -------------------------------------------------------

def _parse_docx(path: Path) -> list[Block]:
    try:
        import docx
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "DOCX ingestion needs the optional dependency: pip install 'gridfin[ingest]'"
        ) from e
    document = docx.Document(str(path))
    blocks: list[Block] = []
    section: Optional[str] = None
    offset = 0
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        detected = _section_for(text)
        if detected:
            section = detected
        blocks.append(
            Block(text=text, kind="prose", section=section, char_span=(offset, offset + len(text)))
        )
        offset += len(text)
    for table in document.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        t = Table(rows=rows, confidence=0.9)
        blocks.append(Block(text=t.to_text(), kind="table", table=t))
    return blocks


_TEXT_EXT = {".txt", ".md", ".markdown"}
# Extensions treated as ingestible documents during directory scans.
_DOC_EXT = _TEXT_EXT | {".csv", ".pdf", ".docx", ".xlsx"}


def parse_file(path: str | Path) -> ParsedDocument:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _TEXT_EXT:
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks = _parse_text(path, raw)
    elif ext == ".csv":
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks = _parse_csv(path, raw)
    elif ext == ".pdf":
        blocks = _parse_pdf(path)
    elif ext in {".docx"}:
        blocks = _parse_docx(path)
    else:
        # Best-effort: treat anything else as text.
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks = _parse_text(path, raw)

    title = path.stem.replace("_", " ").replace("-", " ").strip()
    return ParsedDocument(
        doc_id=_doc_id(path),
        filename=path.name,
        filetype=ext.lstrip("."),
        title=title,
        blocks=blocks,
    )


def ingest_path(path: str | Path) -> list[ParsedDocument]:
    """Parse a single file or every supported file in a directory."""
    path = Path(path)
    if path.is_dir():
        docs = []
        for child in sorted(path.iterdir()):
            if (
                child.is_file()
                and not child.name.startswith(".")
                and child.suffix.lower() in _DOC_EXT
            ):
                docs.append(parse_file(child))
        return docs
    return [parse_file(path)]
