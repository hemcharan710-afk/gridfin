"""Structured extraction into a numeric store.

A regex pre-filter, then optionally an LLM, pulls reported figures into a validated
store at ingestion time. The store doubles as a fast path: a cell that can answer
from it skips the LLM entirely.

The reference implementation extracts deterministically with regex so the demo
needs no model. A live LLM, when configured, can refine ambiguous figures — but it
never invents one, and the store is always validated.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from pydantic import BaseModel

from gridfin.ingestion import ParsedDocument

# --- Metric vocabulary ----------------------------------------------------
# Canonical key -> the surface phrases that denote it in filings.
METRIC_VOCAB: dict[str, list[str]] = {
    "revenue": ["total revenue", "net revenue", "net sales", "total sales", "revenue", "sales"],
    "net_income": ["net income", "net earnings", "net profit", "profit for the year"],
    "gross_profit": ["gross profit", "gross margin dollars"],
    "operating_income": ["operating income", "income from operations", "operating profit"],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities"],
    "total_equity": ["total equity", "shareholders' equity", "stockholders' equity", "total shareholders equity"],
    "current_assets": ["total current assets", "current assets"],
    "current_liabilities": ["total current liabilities", "current liabilities"],
    "total_debt": ["total debt", "long-term debt", "long term debt"],
    "cash": ["cash and cash equivalents", "cash equivalents", "cash"],
    "shares_outstanding": ["shares outstanding", "diluted shares", "weighted average shares"],
    "rd_expense": ["research and development", "r&d expense", "r&d"],
}

# Map any surface phrase back to its canonical key, longest-first so
# "total current assets" wins over "current assets".
_PHRASE_TO_KEY: list[tuple[str, str]] = sorted(
    ((phrase, key) for key, phrases in METRIC_VOCAB.items() for phrase in phrases),
    key=lambda kv: len(kv[0]),
    reverse=True,
)

# A monetary / numeric figure, optionally with a magnitude suffix and parens (negatives).
_NUM = re.compile(
    r"\(?-?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion|thousand|m|bn|b|k)?\)?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:FY\s*)?(19|20)\d{2}")
_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}


class NumericFact(BaseModel):
    doc_id: str
    metric: str
    value: float
    raw: str
    unit: str = "USD"
    period: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    char_span: tuple[int, int] = (0, 0)
    confidence: float = 0.95


def _parse_number(text: str, is_negative: bool) -> Optional[float]:
    m = _NUM.search(text)
    if not m:
        return None
    base = float(m.group(1).replace(",", ""))
    suffix = (m.group(2) or "").lower()
    base *= _MULT.get(suffix, 1.0)
    # Parenthesised figures denote negatives in financial statements.
    if is_negative or (text.strip().startswith("(") and text.strip().endswith(")")):
        base = -base
    return base


def _canonical_metric(label: str) -> Optional[str]:
    low = " " + label.lower().strip() + " "
    for phrase, key in _PHRASE_TO_KEY:
        if f" {phrase} " in low or low.strip() == phrase:
            return key
    return None


def _phrase_confidence(phrase: str, base: float = 0.6) -> float:
    """More specific (longer) phrases are trusted more, so 'total revenue' beats a
    bare 'revenue' that happens to appear inside an MD&A growth sentence."""
    return min(0.92, base + 0.06 * len(phrase.split()))


def _extract_from_table(doc_id: str, block) -> Iterable[NumericFact]:
    table = block.table
    if not table:
        return
    period = None
    # Try to read a fiscal year out of the header row.
    header = " ".join(table.rows[0]) if table.rows else ""
    ym = _YEAR.search(header)
    if ym:
        period = ym.group(0).replace("FY", "").strip()
    for row in table.rows:
        if not row:
            continue
        label = row[0]
        key = _canonical_metric(label)
        if not key:
            continue
        # Use the first numeric-looking cell after the label.
        for cell in row[1:]:
            val = _parse_number(cell, is_negative=False)
            if val is not None:
                yield NumericFact(
                    doc_id=doc_id,
                    metric=key,
                    value=val,
                    raw=f"{label}: {cell}",
                    period=period,
                    page=block.page,
                    section=block.section or "table",
                    char_span=block.char_span,
                    confidence=0.97,
                )
                break


def _extract_from_prose(doc_id: str, block) -> Iterable[NumericFact]:
    # Normalise whitespace so a metric phrase split across a line break
    # ("Net\nincome") still matches. Spans become approximate (acceptable here).
    text = re.sub(r"\s+", " ", block.text)
    low = text.lower()
    period = None
    ym = _YEAR.search(text)
    if ym:
        period = ym.group(0).replace("FY", "").strip()
    seen: set[str] = set()
    for phrase, key in _PHRASE_TO_KEY:
        if key in seen:
            continue
        idx = low.find(phrase)
        if idx == -1:
            continue
        window = text[idx : idx + 80]
        val = _parse_number(window[len(phrase):], is_negative="loss" in window.lower())
        if val is not None:
            seen.add(key)
            start = block.char_span[0] + idx
            yield NumericFact(
                doc_id=doc_id,
                metric=key,
                value=val,
                raw=window.strip(),
                period=period,
                page=block.page,
                section=block.section,
                char_span=(start, start + len(window)),
                confidence=_phrase_confidence(phrase),
            )


def extract_figures(text: str) -> list[float]:
    """All monetary/numeric figures in `text`, with magnitude suffixes applied.

    Shared by attribution checks so that a stated value of 4.2e9 is recognised in a
    span reading "$4,200 million" (digits 4,200 + suffix 'million')."""
    figures: list[float] = []
    for m in _NUM.finditer(text):
        base = float(m.group(1).replace(",", ""))
        suffix = (m.group(2) or "").lower()
        figures.append(base * _MULT.get(suffix, 1.0))
    return figures


def scan_value(text: str, metric: str) -> Optional[tuple[float, str]]:
    """Find a figure for `metric` inside an arbitrary text span (cell fallback path).

    Used by the per-cell retrieval-extract route when the numeric store has no
    pre-extracted fact: scan the retrieved chunk for the metric's phrases and parse
    the adjacent number. Returns (value, raw snippet) or None — never a guess.
    """
    text = re.sub(r"\s+", " ", text)
    low = text.lower()
    phrases = METRIC_VOCAB.get(metric, [metric.replace("_", " ")])
    # Longest phrase first to avoid matching "revenue" inside "total revenue".
    for phrase in sorted(phrases, key=len, reverse=True):
        idx = low.find(phrase)
        if idx == -1:
            continue
        window = text[idx : idx + 90]
        val = _parse_number(window[len(phrase):], is_negative="loss" in window.lower())
        if val is not None:
            return val, window.strip()
    return None


class NumericStore:
    """Validated numeric figures keyed by (doc_id, metric). The cell fast path."""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], NumericFact] = {}

    def add(self, fact: NumericFact) -> None:
        key = (fact.doc_id, fact.metric)
        # Prefer higher-confidence sources (tables over prose).
        existing = self._facts.get(key)
        if existing is None or fact.confidence > existing.confidence:
            self._facts[key] = fact

    def get(self, doc_id: str, metric: str) -> Optional[NumericFact]:
        return self._facts.get((doc_id, metric))

    def for_doc(self, doc_id: str) -> dict[str, NumericFact]:
        return {m: f for (d, m), f in self._facts.items() if d == doc_id}

    def __len__(self) -> int:
        return len(self._facts)


def build_numeric_store(docs: Iterable[ParsedDocument]) -> NumericStore:
    store = NumericStore()
    for doc in docs:
        for block in doc.blocks:
            facts = (
                _extract_from_table(doc.doc_id, block)
                if block.kind == "table"
                else _extract_from_prose(doc.doc_id, block)
            )
            for fact in facts:
                store.add(fact)
    return store
