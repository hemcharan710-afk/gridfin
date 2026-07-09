"""Cross-cell verification and synthesis.

Three checks, then a narrative:
  - column consistency : same metric comparable across docs, units normalised.
  - row consistency    : accounting identities where applicable.
  - citation-match     : the stated figure equals the stored/source figure.

The grid is then synthesised into a narrative answer with citations back to each
cell's source.
"""

from __future__ import annotations

from gridfin.extraction import NumericStore, extract_figures
from gridfin.llm import LLM
from gridfin.models import Grid, VerificationNote


def _row_identity_checks(grid: Grid, doc_id: str) -> list[VerificationNote]:
    notes: list[VerificationNote] = []

    def val(col_id: str):
        if not any(c.column_id == col_id for c in grid.columns):
            return None
        cell = grid.get(doc_id, col_id)
        return cell.value if cell.status == "done" and isinstance(cell.value, (int, float)) else None

    revenue = val("revenue")
    gross_profit = val("gross_profit")
    net_income = val("net_income")
    total_assets = val("total_assets")
    total_equity = val("total_equity")

    # Identity: gross profit must not exceed revenue.
    if revenue is not None and gross_profit is not None and gross_profit > revenue * 1.001:
        notes.append(
            VerificationNote(
                kind="row_consistency",
                level="error",
                message=f"{doc_id}: gross profit ({gross_profit:,.0f}) exceeds revenue ({revenue:,.0f}).",
                cells=[(doc_id, "gross_profit"), (doc_id, "revenue")],
            )
        )
    # Identity: net income must not exceed revenue.
    if revenue is not None and net_income is not None and net_income > revenue * 1.001:
        notes.append(
            VerificationNote(
                kind="row_consistency",
                level="warning",
                message=f"{doc_id}: net income ({net_income:,.0f}) exceeds revenue ({revenue:,.0f}).",
                cells=[(doc_id, "net_income"), (doc_id, "revenue")],
            )
        )
    # Identity: equity must not exceed total assets.
    if total_assets is not None and total_equity is not None and total_equity > total_assets * 1.001:
        notes.append(
            VerificationNote(
                kind="row_consistency",
                level="error",
                message=f"{doc_id}: total equity ({total_equity:,.0f}) exceeds total assets ({total_assets:,.0f}).",
                cells=[(doc_id, "total_equity"), (doc_id, "total_assets")],
            )
        )
    return notes


def _column_consistency_checks(grid: Grid) -> list[VerificationNote]:
    notes: list[VerificationNote] = []
    for col in grid.columns:
        if col.type == "ratio" and col.formula in ("net_profit_margin", "gross_margin", "operating_margin"):
            for cell in grid.column(col.column_id):
                if cell.status == "done" and isinstance(cell.value, (int, float)):
                    if cell.value > 1.0 or cell.value < -1.0:
                        notes.append(
                            VerificationNote(
                                kind="column_consistency",
                                level="warning",
                                message=f"{cell.doc_id}/{col.column_id}: margin {cell.value:.1%} outside the plausible [-100%, 100%] band.",
                                cells=[(cell.doc_id, col.column_id)],
                            )
                        )
    return notes


def _citation_match_checks(grid: Grid, store: NumericStore | None) -> list[VerificationNote]:
    """Attribution correctness: the cited source actually contains the stated value."""
    notes: list[VerificationNote] = []
    for cell in grid.cells.values():
        if cell.status != "done" or cell.path == "compute":
            continue
        if not isinstance(cell.value, (int, float)):
            continue
        # The cell's own cited span should contain the figure (magnitude-aware).
        if cell.detail:
            target = abs(cell.value)
            ok = any(abs(v - target) <= max(1.0, target * 0.01) for v in extract_figures(cell.detail))
            if not ok:
                notes.append(
                    VerificationNote(
                        kind="citation_match",
                        level="warning",
                        message=f"{cell.doc_id}/{cell.column_id}: stated value {cell.display()} not found in cited span.",
                        cells=[(cell.doc_id, cell.column_id)],
                    )
                )
    return notes


def verify_grid(grid: Grid, store: NumericStore | None = None) -> list[VerificationNote]:
    notes: list[VerificationNote] = []
    notes.extend(_column_consistency_checks(grid))
    for row in grid.rows:
        notes.extend(_row_identity_checks(grid, row.doc_id))
    notes.extend(_citation_match_checks(grid, store))
    grid.verification = notes
    return notes


def _grid_table_text(grid: Grid) -> str:
    lines = []
    header = "Document | " + " | ".join(c.name for c in grid.columns)
    lines.append(header)
    for row in grid.rows:
        cells = grid.row(row.doc_id)
        rendered = []
        for c in cells:
            piece = c.display()
            if c.source:
                piece += f" [{c.source.short()}]"
            rendered.append(piece)
        lines.append(f"{row.title or row.doc_id} | " + " | ".join(rendered))
    return "\n".join(lines)


async def synthesize(grid: Grid, llm: LLM) -> str:
    """Turn the filled, verified grid into a cited narrative."""
    table = _grid_table_text(grid)
    warnings = [n.message for n in grid.verification if n.level in ("warning", "error")]
    warn_text = ("\n\nVerification flags:\n- " + "\n- ".join(warnings)) if warnings else ""

    # Deterministic fallback narrative (used in mock mode).
    mock_lines = [f"Across {len(grid.rows)} document(s), GridFin answered: {grid.question}", ""]
    for row in grid.rows:
        parts = []
        for c in grid.row(row.doc_id):
            if c.status == "done":
                spec = grid.column_spec(c.column_id)
                cite = f" ({c.source.short()})" if c.source else ""
                parts.append(f"{spec.name.lower()} = {c.display()}{cite}")
            elif c.status == "empty":
                parts.append(f"{grid.column_spec(c.column_id).name.lower()} not disclosed")
        if parts:
            mock_lines.append(f"- {row.title or row.doc_id}: " + "; ".join(parts) + ".")
    if warnings:
        mock_lines.append("")
        mock_lines.append("Flags: " + "; ".join(warnings))
    mock_text = "\n".join(mock_lines)

    prompt = (
        "Write a concise, decision-useful answer to the QUESTION using ONLY the grid "
        "below. Cite each figure with its bracketed source. Note any verification "
        "flags. Do not compute new numbers.\n\n"
        f"QUESTION: {grid.question}\n\nGRID:\n{table}{warn_text}"
    )
    result = await llm.complete(prompt, tier="large", context={"mock_text": mock_text})
    grid.narrative = result.text
    return result.text
