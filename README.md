# GridFin — Grid-Based Analysis of Financial Documents

[![CI](https://github.com/hemcharan51/gridfin/actions/workflows/ci.yml/badge.svg)](https://github.com/hemcharan51/gridfin/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GridFin answers analytical questions that span many financial filings. It takes a
question like *"compare net profit margin and the revenue trend across these
filings"* and turns it into a grid: documents down the rows, sub-questions across
the columns. Each cell is answered on its own, scoped to a single document, and any
figure that has to be computed goes through a plain Python engine instead of the
model. The filled grid is then checked for consistency and written up as a short
answer that cites its sources.

```
Query -> decompose -> fan out (one cell per doc x sub-question) -> verify -> synthesize
         rows=docs     each cell scoped to one document            cross-cell   grid + cited answer
         cols=q's                                                  checks
```

Design notes and diagrams: [docs/ARCHITECTURE_AND_PLAN.md](docs/ARCHITECTURE_AND_PLAN.md).

## Why a grid

Answering a multi-document question in a single blended pass works for a handful of
filings but doesn't hold up well as the set grows. Splitting the work into a grid
of independent cells buys three things:

- Scale: cells don't depend on each other, so the work fans out across many documents.
- Isolation: one document per cell, so text from Company A can't bleed into Company B's answer.
- Traceability: every answer points back to exactly one source span.

## The cell

The cell is the unit of both work and audit:

```python
class Cell:
    doc_id      : str          # the one document this cell reads
    column_id   : str          # which sub-question / field
    question    : str          # the scoped sub-question text
    status      : "pending" | "running" | "done" | "failed" | "empty"
    value       : Any          # answer, figure, or computed result
    source      : {file, page, section, char_span}   # exactly one span
    confidence  : float
    path        : "numeric_store" | "retrieval_extract" | "compute"
    cost_tokens : int
```

Each cell takes one of three routes:

1. `numeric_store` — read a figure that was pre-extracted at ingestion. No LLM call.
2. `retrieval_extract` — retrieve within the one document (RRF + rerank), then extract from the span.
3. `compute` — call the deterministic calc engine with exact figures from other cells.

If a figure isn't in the document, the cell comes back empty and flagged rather
than invented, and any ratio that depends on it is refused instead of computed on a
guess.

## Quickstart (runs offline, no API key)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Look at the demo corpus and which backends are active
gridfin info

# Ask a cross-document question; prints the grid and a cited answer
gridfin ask "Compare net profit margin and revenue trend across these filings"

# Run the evaluation metrics
gridfin eval

# Point any command at a different corpus with --data
gridfin ask "Compare gross margin across these filings" --data data/sec
```

Three corpora ship with the repo: `data/demo/` (small synthetic filings, the
default), `data/sample/` (a couple more synthetic filings plus a CSV), and
`data/sec/` — the real FY2023 Form 10-Ks for Apple, Microsoft, and NVIDIA with
ground-truth figures pulled from SEC EDGAR XBRL.

With no API key set, GridFin runs in mock mode: the whole pipeline still runs end
to end on the bundled demo dataset (`data/demo/`) with no network calls. See
[Switching to a live model](#switching-to-a-live-model) to route the language steps
through Claude or OpenAI.

### Switching to a live model

Offline is the default. In live mode the model handles only planning, text
extraction, and the final narrative; every figure still comes from the
deterministic engine, so the numbers are identical either way.

1. Create your env file:
   ```bash
   cp .env.example .env
   ```
2. Pick a provider and add its key in `.env`.

   For Claude:
   ```
   GRIDFIN_LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   For OpenAI:
   ```
   GRIDFIN_LLM_PROVIDER=openai
   GRIDFIN_OPENAI_API_KEY=sk-proj-...
   ```
3. Make sure mock mode isn't forced: `GRIDFIN_MOCK_LLM=false`.
4. Confirm it took — this should print `llm : live`, not `mock`:
   ```bash
   gridfin info
   ```
5. If the Streamlit UI is already running, restart it — settings are read once per
   process, so it won't pick up `.env` changes until you relaunch it.

Model routing is configurable: `GRIDFIN_MODEL_SMALL` / `GRIDFIN_MODEL_LARGE` for
Claude, or `GRIDFIN_OPENAI_MODEL_SMALL` / `GRIDFIN_OPENAI_MODEL_LARGE` for OpenAI.
The small model handles simple extraction cells; the large one handles planning,
hard cells, and synthesis. To go back offline, set `GRIDFIN_MOCK_LLM=true` (or
remove the key) and restart.

### Interactive grid UI

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Cells stream in as they finish; click a cell to see its source span, confidence,
and which route filled it.

## How it fits together

| Step | What it does | Where |
|------|--------------|-------|
| Ingestion | Parse each file into text + tables with shared metadata | [`ingestion/`](src/gridfin/ingestion/ingest.py) |
| Chunking | Keep tables whole, sliding window over prose | [`chunking/`](src/gridfin/chunking/chunker.py) |
| Numeric store | Pull reported figures into a validated store (the fast path) | [`extraction/`](src/gridfin/extraction/numeric_store.py) |
| Dual index | Dense + sparse index with a per-document filter | [`indexing/`](src/gridfin/indexing/dual_index.py) |
| Decompose | Turn the question into columns and rows | [`grid/decompose.py`](src/gridfin/grid/decompose.py) |
| Cell engine | Run every cell concurrently (the fan-out) | [`grid/cell_engine.py`](src/gridfin/grid/cell_engine.py) |
| Retrieval | RRF + rerank, scoped to one document | [`retrieval/`](src/gridfin/retrieval/retriever.py) |
| Calc engine | Deterministic ratios and growth rates | [`calc/`](src/gridfin/calc/engine.py) |
| Verify + synthesize | Cross-cell checks, then a cited narrative | [`grid/verify.py`](src/gridfin/grid/verify.py) |
| Front end | Interactive grid | [`app/streamlit_app.py`](app/streamlit_app.py) |

Cell-level caching ([`cache/`](src/gridfin/cache/cell_cache.py)) and the evaluation
harness ([`eval/`](src/gridfin/eval/metrics.py)) sit across the whole pipeline.

Everything is wired together in [`pipeline.py`](src/gridfin/pipeline.py):

```python
from gridfin.pipeline import GridFin

corpus = GridFin.from_path("data/demo")            # ingest + index once
answer = await corpus.ask("Compare net margin across these filings")
print(answer.narrative)
print(answer.grid.completion())                    # fraction of cells filled
```

## Keeping the cost down

A grid can issue up to `documents x columns` model calls, so a few things keep the
common case cheap:

- A concurrency cap (`GRIDFIN_MAX_CONCURRENCY`) bounds how many calls are in flight.
- Cells are cached on `(document, sub-question)`, so changing one column only recomputes that column.
- Simple extraction cells use the small model; only hard cells, planning, and synthesis use the large one.
- Cells answerable from the numeric store skip the model entirely.

## The deterministic boundary

The one thing GridFin never hands to the model is arithmetic. Any cell that needs a
computed ratio goes through [`calc/engine.py`](src/gridfin/calc/engine.py) with exact
figures from the store and gets back the formula, the inputs, and the result.
Unsupported formulas and missing inputs are refused rather than approximated, so a
number in the grid is either right or absent, never guessed.

## Evaluation

```bash
# Default: the synthetic demo corpus
gridfin eval

# Against the real SEC filings (Apple, Microsoft, NVIDIA FY2023 10-Ks)
gridfin eval --data data/sec --truth data/sec/ground_truth.json
```

- Cell Numeric Accuracy — fraction of numeric cells that match ground truth. This is the number that matters most for a financial tool.
- Grid Completion — fraction of cells filled rather than failed or empty.
- Attribution Correctness — whether each cell's cited source actually contains its value.
- RAGAS (faithfulness / relevancy / precision / recall) is available behind the optional `[eval]` extra.

The real-SEC set grounds these metrics on actual 10-Ks rather than the toy corpus:
figures in `data/sec/ground_truth.json` are the exact amounts each company reported
in FY2023 (source: SEC EDGAR XBRL), and `tests/test_eval_real.py` asserts exact
numeric accuracy and attribution against them.

## Light vs. heavy stack

The default install is deliberately light so the demo runs anywhere. Heavier
dependencies are optional and the code falls back gracefully when they're missing.

| Concern | Default (light) | Heavy stack (`pip install -e ".[ml]"`) |
|---|---|---|
| Dense embeddings | hashed bag-of-words (numpy) | BGE via sentence-transformers |
| ANN index | brute-force numpy cosine | FAISS (per-doc id selector keeps cells isolated) |
| Reranking | RRF order | CrossEncoder |
| Sparse | rank-bm25 | rank-bm25 |
| LLM | deterministic mock | Claude (structured output) |
| Ingestion | txt / csv | pdfplumber / PyMuPDF / python-docx (`[ingest]`) |

Optional extras: `[ml]`, `[ingest]`, `[ui]`, `[eval]`, `[dev]`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```
src/gridfin/
  models.py            # the grid data model: Cell, Grid, GridPlan, Source
  config.py            # settings + optional-dependency flags
  pipeline.py          # ties the stages together
  cli.py               # gridfin info | ask | eval
  llm/                 # structured-output client + offline mock backend
  ingestion/           # parse files into text + tables
  chunking/            # chunk documents
  extraction/          # numeric store (the cell fast path)
  indexing/            # dual index + per-document filter
  grid/                # decompose, cell engine, verify/synthesize, store
  retrieval/           # RRF + rerank, scoped to one document
  calc/                # deterministic calculation engine
  cache/               # cell-level cache
  eval/                # evaluation metrics
app/streamlit_app.py   # interactive grid
data/demo/             # three synthetic demo filings + ground truth (default corpus)
data/sample/           # extra synthetic filings + a CSV
data/sec/              # real Apple / Microsoft / NVIDIA FY2023 10-Ks + EDGAR ground truth
tests/                 # unit + integration tests, incl. real-SEC eval (test_eval_real.py)
```
