"""Interactive grid front end.

Cells stream in as they finish; clicking a cell reveals its source span,
confidence, and route.

Run:  streamlit run app/streamlit_app.py
Install the UI extra first:  pip install -e ".[ui]"
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from gridfin.models import Grid
from gridfin.pipeline import GridFin

try:
    from st_aggrid import AgGrid, GridOptionsBuilder

    HAS_AGGRID = True
except Exception:  # pragma: no cover - optional dependency
    HAS_AGGRID = False

ROUTE_COLORS = {
    "numeric_store": "#2b6cb0",      # blue  — store (no LLM)
    "retrieval_extract": "#dd6b20",  # orange — retrieval + extract
    "compute": "#2f855a",            # green — deterministic compute
}

st.set_page_config(page_title="GridFin — Grid", layout="wide")


_UPLOAD_TYPES = ["txt", "md", "markdown", "csv", "pdf", "docx", "xlsx"]


@st.cache_resource(show_spinner="Building corpus (ingest · chunk · numeric store · index)…")
def build_corpus(data_dir: str) -> GridFin:
    return GridFin.from_path(data_dir)


def _uploads_signature(files) -> str:
    """Content hash over the uploaded files — changes whenever any byte changes,
    so re-uploading a same-named file with new numbers busts the cell cache
    (cells are keyed on doc_id = hash(filename), not on document content)."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name):
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner="Building corpus from uploads (ingest · chunk · numeric store · index)…")
def build_corpus_from_uploads(signature: str, _payload: tuple) -> GridFin:
    """`signature` is the cache key (Streamlit ignores the underscored payload).

    Uploaded files are written to a per-signature temp directory, and the cell
    cache is namespaced to the same signature so different content never collides
    with a stale cached answer."""
    tmp = Path(tempfile.mkdtemp(prefix=f"gridfin_upload_{signature}_"))
    for name, data in _payload:
        (tmp / Path(name).name).write_bytes(data)
    return GridFin.from_path(str(tmp), cache_namespace=f"upload-{signature}")


def render_aggrid(grid: Grid) -> None:
    # Pivot the flat cell table into the document × column matrix.
    matrix = {}
    for row in grid.rows:
        rec = {"Document": row.title or row.doc_id}
        for col in grid.columns:
            rec[col.name] = grid.get(row.doc_id, col.column_id).display()
        matrix[row.doc_id] = rec
    df = pd.DataFrame(list(matrix.values()))

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
    gb.configure_selection("single")
    grid_options = gb.build()
    AgGrid(df, gridOptions=grid_options, allow_unsafe_jscode=True, fit_columns_on_grid_load=True, height=260)


def render_fallback(grid: Grid) -> None:
    rows = []
    for row in grid.rows:
        rec = {"Document": row.title or row.doc_id}
        for col in grid.columns:
            rec[col.name] = grid.get(row.doc_id, col.column_id).display()
        rows.append(rec)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_cell_detail(grid: Grid) -> None:
    st.markdown("#### Click-through · per-cell provenance")
    st.caption("Every cell traces to exactly one source span, with its route and confidence.")
    options = []
    for row in grid.rows:
        for col in grid.columns:
            cell = grid.get(row.doc_id, col.column_id)
            label = f"{row.title or row.doc_id} → {col.name}"
            options.append((label, cell))
    labels = [o[0] for o in options]
    chosen = st.selectbox("Inspect a cell", labels, index=0)
    cell = dict(options)[chosen]

    route = cell.path or "—"
    color = ROUTE_COLORS.get(cell.path, "#666")
    c1, c2, c3 = st.columns(3)
    c1.metric("Value", cell.display())
    c2.metric("Confidence", f"{cell.confidence:.0%}")
    c3.markdown(
        f"**Route**<br><span style='color:{color};font-weight:700'>{route}</span>",
        unsafe_allow_html=True,
    )
    if cell.detail:
        st.markdown(f"**Computed / extracted:** `{cell.detail}`")
    if cell.source:
        st.markdown(f"**Source:** {cell.source.file} · {cell.source.short()}")
    if cell.error:
        st.warning(f"Status `{cell.status}`: {cell.error}")
    elif cell.status == "empty":
        st.info("Returned empty: the figure was not disclosed, so it was not invented.")


def main() -> None:
    st.title("GridFin — grid-based financial document analysis")
    st.caption(
        "Decompose, fan out, verify, synthesize. Every number goes through a "
        "deterministic engine rather than the model."
    )

    with st.sidebar:
        st.header("Corpus")
        uploaded = st.file_uploader(
            "Upload filings",
            type=_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="txt / csv work out of the box. pdf / docx / xlsx need the "
            "[ingest] extra: pip install -e \".[ingest]\".",
        )
        data_dir = st.text_input("…or load a folder", value="data/demo")

        if uploaded:
            signature = _uploads_signature(uploaded)
            payload = tuple((f.name, f.getvalue()) for f in uploaded)
            corpus = build_corpus_from_uploads(signature, payload)
            st.caption(f"Loaded {len(uploaded)} uploaded file(s).")
        else:
            corpus = build_corpus(data_dir)
            st.caption(f"Loaded from folder: {data_dir}")

        info = corpus.info()
        st.json(info)
        st.markdown(
            f"<small>Route legend: "
            f"<span style='color:{ROUTE_COLORS['numeric_store']}'>● store</span> · "
            f"<span style='color:{ROUTE_COLORS['retrieval_extract']}'>● retrieval</span> · "
            f"<span style='color:{ROUTE_COLORS['compute']}'>● compute</span></small>",
            unsafe_allow_html=True,
        )

    question = st.text_input(
        "Ask a cross-document question",
        value="Compare net profit margin and revenue trend across these filings",
    )
    if st.button("Run grid", type="primary") or question:
        with st.spinner("Filling the grid (parallel cells)…"):
            answer = asyncio.run(corpus.ask(question))
        grid = answer.grid

        cols = st.columns(3)
        cols[0].metric("Grid shape", f"{len(grid.rows)} × {len(grid.columns)}")
        cols[1].metric("Completion", f"{grid.completion():.0%}")
        cols[2].metric("LLM", info["llm"])

        st.markdown("### Grid")
        if HAS_AGGRID:
            render_aggrid(grid)
        else:
            st.info("Install the UI extra for the interactive AG Grid: `pip install -e \".[ui]\"`")
            render_fallback(grid)

        flagged = [n for n in grid.verification if n.level in ("warning", "error")]
        if flagged:
            st.markdown("### Verification flags")
            for n in flagged:
                (st.error if n.level == "error" else st.warning)(n.message)

        render_cell_detail(grid)

        st.markdown("### Synthesized answer")
        st.write(answer.narrative)


if __name__ == "__main__":
    main()
