# GridFin — Architecture & Design Notes

Notes on how GridFin is built and where I'd take it next. It's written against the
actual code in `src/gridfin/`, so every layer maps to a module you can open. The
diagrams are [Mermaid](https://mermaid.js.org/) and render on GitHub, VS Code, and
most Markdown viewers.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [The core thesis: why a grid](#2-the-core-thesis-why-a-grid)
3. [System context](#3-system-context)
4. [End-to-end query-to-answer flow](#4-end-to-end-query-to-answer-flow)
5. [The ten-layer pipeline](#5-the-ten-layer-pipeline)
6. [Front half — built once per corpus (Layers 1–4)](#6-front-half--built-once-per-corpus-layers-14)
7. [Back half — run per query (Layers 5–10)](#7-back-half--run-per-query-layers-510)
8. [The cell — atomic unit of work & audit](#8-the-cell--atomic-unit-of-work--audit)
9. [Cell routing: the three engines](#9-cell-routing-the-three-engines)
10. [Two-wave dependency execution](#10-two-wave-dependency-execution)
11. [Request sequence (full trace)](#11-request-sequence-full-trace)
12. [The D × C cost tax & its four mitigations](#12-the-d--c-cost-tax--its-four-mitigations)
13. [The determinism boundary](#13-the-determinism-boundary)
14. [Configuration & deployment modes](#14-configuration--deployment-modes)
15. [Evaluation harness](#15-evaluation-harness)
16. [Code map](#16-code-map)
17. [Implementation plan & roadmap](#17-implementation-plan--roadmap)
18. [Risks & mitigations](#18-risks--mitigations)
19. [Testing strategy](#19-testing-strategy)

---

## 1. Executive summary

GridFin answers analytical questions that span **many** financial filings by decomposing each question into a matrix of **documents (rows) × sub-questions (columns)**, answering every **cell** independently and in parallel — scoped to a single document — then verifying across cells and synthesizing a cited narrative.

```
Query → Decompose → Fan-out (parallel cells) → Verify → Synthesize
        rows=docs    each cell scoped to        cross-cell    grid +
        cols=q's     one document               checks        cited answer
```

The core idea is in the control flow: instead of one blended retrieval-and-reason
pass over everything, there's a planner plus many isolated per-cell workers. And
every number stays out of the model's hands and goes through a deterministic Python
engine instead.

| Property | How GridFin delivers it |
|---|---|
| Scale | Cells are independent, so they fan out across many documents (bounded by a semaphore). |
| Isolation | One document per cell; retrieval is hard-filtered, so Company A's text can't leak into Company B's answer. |
| Auditability | Every non-empty cell traces to exactly one source span; every ratio returns formula + inputs + result. |
| Trust | Missing figures come back empty and flagged, never invented; unsupported ratios are refused, never approximated. |
| Runs offline | With no API key, a mock LLM runs the full pipeline on the bundled demo with no network calls. |

---

## 2. The core thesis: why a grid

A blended ReAct loop is excellent for a *few* documents but leaves performance on the table when a question spans *many* filings: context dilutes, cross-company facts bleed together, and there is no clean audit trail. The grid reframes the problem as a 2-D array of small, independent, cacheable units of work.

```mermaid
flowchart LR
    Q["Analytical question<br/>'Compare net margin & D/E<br/>across these filings'"]:::q
    DECOMP["Decompose into<br/>rows = documents<br/>cols = sub-questions"]:::step
    GRID["The grid<br/>every cell answered independently,<br/>scoped to ONE document, in parallel"]:::grid
    ANS["Cited narrative answer<br/>+ verification flags"]:::ans

    Q --> DECOMP --> GRID --> ANS

    classDef q fill:#1f2937,color:#fff,stroke:#111;
    classDef step fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
    classDef grid fill:#eef2ff,color:#3730a3,stroke:#6366f1;
    classDef ans fill:#f0fdf4,color:#166534,stroke:#16a34a;
```

The grid for *"Compare net margin & debt-to-equity across these filings"* (🔵 = pulled from the numeric store, no LLM · 🟢 = deterministically computed):

| Document | Revenue 🔵 | Net income 🔵 | Net margin 🟢 | Debt / equity 🟢 |
|---|---|---|---|---|
| **Acme 10-K** | $33.9B | $4.2B | 12.4% | 0.50x |
| **Beta 10-K** | $21.2B | $1.9B | 9.0% | 0.70x |
| **Gamma Annual** | $8.4B | **— (empty)** | **refused** | 1.08x |

> The Gamma row shows the discipline: Gamma discloses no net income, so its **net income cell is empty (—)** and its **net-margin cell is refused** rather than computed on a guess. Its debt/equity cell still computes because both *its* inputs are present.

---

## 3. System context

How GridFin sits between its inputs (filings, a question, optional model provider) and its outputs (an interactive grid and a CSV export).

```mermaid
flowchart TB
    subgraph IN["Inputs"]
        FILES["Filings<br/>txt · csv · pdf · docx · xlsx"]:::in
        QUESTION["User question"]:::in
        KEY["ANTHROPIC_API_KEY / OPENAI_API_KEY<br/>(optional)"]:::opt
    end

    subgraph CORE["GridFin engine (src/gridfin)"]
        FRONT["Front half<br/>ingest · chunk · numeric store · index<br/><i>built once per corpus</i>"]:::core
        BACK["Back half<br/>decompose · fan-out · verify · synthesize<br/><i>run per query</i>"]:::core
        FRONT --> BACK
    end

    subgraph EXT["External (only when a key is set)"]
        CLAUDE["Claude / OpenAI<br/>planning · text extract · synthesis"]:::ext
    end

    subgraph OUT["Outputs / surfaces"]
        UI["Streamlit grid UI<br/>app/streamlit_app.py"]:::out
        CLI["CLI: gridfin info | ask | eval"]:::out
        CSV["CSV export"]:::out
    end

    FILES --> FRONT
    QUESTION --> BACK
    KEY -.-> CLAUDE
    BACK <-.->|"structured output<br/>(mock if no key)"| CLAUDE
    BACK --> UI & CLI & CSV

    classDef in fill:#e0f2fe,color:#075985,stroke:#0284c7;
    classDef opt fill:#f4f4f5,color:#52525b,stroke:#a1a1aa,stroke-dasharray:4 3;
    classDef core fill:#eef2ff,color:#3730a3,stroke:#6366f1;
    classDef ext fill:#fdf4ff,color:#86198f,stroke:#c026d3;
    classDef out fill:#f0fdf4,color:#166534,stroke:#16a34a;
```

**Key seam:** the LLM is an *optional* collaborator. Without a key (`LLM.live == False`), planning falls back to a deterministic heuristic planner, text cells fall back to the top retrieved snippet, and synthesis falls back to a templated narrative — so the full ten layers still execute. Numbers are **never** the LLM's job in either mode.

---

## 4. End-to-end query-to-answer flow

The canonical path a question takes. The front half is reused across every query; only the back half re-runs.

```mermaid
flowchart TB
    START(["corpus.ask(question)"]):::entry

    subgraph BUILT["BUILT ONCE PER CORPUS — GridFin.from_path()"]
        direction LR
        L1["①  Ingestion<br/>format-aware parse → blocks"]:::keep
        L2["②  Chunking<br/>tables whole · prose windowed"]:::keep
        L3["③  Numeric store<br/>regex extract → validated facts"]:::keep
        L4["④  Dual index<br/>dense + sparse · per-doc filter"]:::wire
        L1 --> L2 --> L4
        L1 --> L3
    end

    subgraph PERQ["RUN PER QUERY"]
        L5["⑤  Decompose<br/>question → GridPlan (cols × rows)"]:::new
        L6["⑥  Cell engine — FAN-OUT<br/>every cell, in parallel"]:::new
        L7["⑦  Scoped retrieval<br/>RRF + rerank, one doc"]:::wire
        L8["⑧  Calc engine<br/>deterministic formulas"]:::keep
        L9["⑨  Verify + Synthesize<br/>cross-cell checks → narrative"]:::new
        L10["⑩  Grid frontend<br/>stream cells · click-through"]:::wire

        L5 --> L6
        L6 -.->|"retrieval_extract route"| L7
        L6 -.->|"compute route"| L8
        L6 --> L9 --> L10
    end

    BUILT --> L5
    L3 -.->|"numeric fast path (no LLM)"| L6
    L4 -.-> L7
    START --> L5
    L10 --> DONE(["GridAnswer{ grid, narrative }"]):::entry

    classDef entry fill:#1f2937,color:#fff,stroke:#111;
    classDef keep fill:#f0fdf4,color:#166534,stroke:#16a34a;
    classDef wire fill:#fffbeb,color:#92400e,stroke:#d97706;
    classDef new fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
```

Colour legend: green = front-half build steps, amber = retrieval steps that run
per cell with a document filter, blue = the grid machinery (decompose, fan-out,
verify/synthesize).

---

## 5. The ten-layer pipeline

The pipeline is orchestrated in [`pipeline.py`](../src/gridfin/pipeline.py). Each layer is one module.

| # | Layer | Module |
|---|-------|--------|
| 1 | Format-aware ingestion | [`ingestion/ingest.py`](../src/gridfin/ingestion/ingest.py) |
| 2 | Chunking | [`chunking/chunker.py`](../src/gridfin/chunking/chunker.py) |
| 3 | Structured extraction → numeric store | [`extraction/numeric_store.py`](../src/gridfin/extraction/numeric_store.py) |
| 4 | Dual index + per-document filter | [`indexing/dual_index.py`](../src/gridfin/indexing/dual_index.py) |
| 5 | Query decomposition & grid construction | [`grid/decompose.py`](../src/gridfin/grid/decompose.py) |
| 6 | Per-cell execution engine (fan-out) | [`grid/cell_engine.py`](../src/gridfin/grid/cell_engine.py) |
| 7 | Scoped retrieval & reranking | [`retrieval/retriever.py`](../src/gridfin/retrieval/retriever.py) |
| 8 | Deterministic calculation engine | [`calc/engine.py`](../src/gridfin/calc/engine.py) |
| 9 | Cross-cell verification & synthesis | [`grid/verify.py`](../src/gridfin/grid/verify.py) |
| 10 | Grid front end | [`app/streamlit_app.py`](../app/streamlit_app.py) |

**Cross-cutting:** cell-level caching ([`cache/cell_cache.py`](../src/gridfin/cache/cell_cache.py)), the evaluation harness ([`eval/metrics.py`](../src/gridfin/eval/metrics.py)), the grid store ([`grid/store.py`](../src/gridfin/grid/store.py)), config & capability flags ([`config.py`](../src/gridfin/config.py)), and the tiered LLM client ([`llm/client.py`](../src/gridfin/llm/client.py)).

---

## 6. Front half — built once per corpus (Layers 1–4)

Construction is `GridFin.from_path("data/demo")`. This is the expensive, reusable part: parse → chunk → extract figures → build the searchable index.

```mermaid
flowchart LR
    RAW["Raw files"]:::raw

    subgraph L1["① Ingestion — ingest.py"]
        ROUTE{"by extension"}:::dec
        PT["txt/md → paragraphs"]:::box
        PC["csv → whole table"]:::box
        PP["pdf → pdfplumber*"]:::box
        PD["docx → python-docx*"]:::box
        ROUTE --> PT & PC & PP & PD
    end

    PARSED["ParsedDocument<br/>blocks[] (prose | table)<br/>+ page · section · char_span"]:::data

    subgraph L2["② Chunking — chunker.py"]
        TW["tables kept whole"]:::box
        SW["prose: 1000-char window,<br/>150 overlap, break on space"]:::box
    end
    CHUNKS["Chunk[]<br/>carries doc_id (key for filter)"]:::data

    subgraph L3["③ Numeric store — numeric_store.py"]
        TBL["table rows → label→figure"]:::box
        PROSE["prose → metric phrase + nearby number"]:::box
        VOCAB["METRIC_VOCAB<br/>13 canonical metrics"]:::box
    end
    STORE["NumericStore<br/>(doc_id, metric) → NumericFact<br/><b>the cell fast path</b>"]:::store

    subgraph L4["④ Dual index — dual_index.py"]
        DENSE["dense: BGE* / hashed BoW<br/>(numpy cosine)"]:::box
        SPARSE["sparse: BM25"]:::box
        DOCROWS["_doc_rows: doc_id → row idx<br/><b>per-doc filter</b>"]:::box
    end
    INDEX["DualIndex (filterable)"]:::data

    RAW --> ROUTE
    PT & PC & PP & PD --> PARSED
    PARSED --> TW & SW --> CHUNKS
    PARSED --> TBL & PROSE
    VOCAB -.-> TBL & PROSE
    TBL & PROSE --> STORE
    CHUNKS --> DENSE & SPARSE & DOCROWS --> INDEX

    classDef raw fill:#e0f2fe,color:#075985,stroke:#0284c7;
    classDef data fill:#f8fafc,color:#334155,stroke:#94a3b8;
    classDef store fill:#dbeafe,color:#1e40af,stroke:#2563eb;
    classDef box fill:#ffffff,color:#27272a,stroke:#d4d4d8;
    classDef dec fill:#fef3c7,color:#92400e,stroke:#d97706;
```
<sub>*optional heavy/ingest dependencies; the demo runs on the pure-Python fallbacks.</sub>

**What each layer guarantees:**

- **① Ingestion** normalizes every format into one `ParsedDocument` schema of `Block`s, each tagged with `page`, `section`, and `char_span` — the provenance that later becomes a cell's `Source`. `doc_id = "{stem}-{sha1(filename)[:8]}"` so ids are reproducible across machines (the eval ground truth references them directly).
- **② Chunking** never splits a table (financial tables lose meaning when cut); prose gets a sliding window with overlap. Every chunk carries `doc_id`, which the per-document filter downstream relies on.
- **③ Numeric store** is the trust engine's supply line. Deterministic regex (`_NUM`, `METRIC_VOCAB`) pulls figures with magnitude suffixes (`$4,200 million` → `4.2e9`) and parenthesis-as-negative handling. Table facts (confidence 0.97) beat prose facts (≤0.92). This store lets numeric cells skip the LLM entirely.
- **④ Dual index** does BGE+BM25 dual retrieval and adds `_doc_rows`, so search can be hard-filtered to a single document before scoring. That filter is what makes cell isolation real.

---

## 7. Back half — run per query (Layers 5–10)

### Layer 5 — Decompose (the brain)

[`decompose.py`](../src/gridfin/grid/decompose.py) turns a free-text question into a strict `GridPlan` (Pydantic): a list of **column specs** (sub-questions / fields) and **row refs** (documents in scope).

```mermaid
flowchart TB
    Q["question + docs in scope"]:::in
    LIVE{"llm.live?"}:::dec

    HEUR["build_heuristic_plan()<br/>keyword → column templates"]:::box
    LLMP["LLM planner (tier=large)<br/>structured GridPlan"]:::llm
    REPAIR["_repair_dependencies()<br/>every ratio's inputs must exist as columns"]:::fix

    PLAN["GridPlan<br/>columns[] × rows[]"]:::out
    G["Grid.from_plan() → empty cells"]:::out

    Q --> LIVE
    LIVE -->|"no (offline)"| HEUR --> PLAN
    LIVE -->|"yes"| LLMP --> REPAIR --> PLAN
    HEUR -.->|"mock hint"| LLMP
    PLAN --> G

    classDef in fill:#e0f2fe,color:#075985,stroke:#0284c7;
    classDef dec fill:#fef3c7,color:#92400e,stroke:#d97706;
    classDef box fill:#fff,color:#27272a,stroke:#d4d4d8;
    classDef llm fill:#fdf4ff,color:#86198f,stroke:#c026d3;
    classDef fix fill:#fffbeb,color:#92400e,stroke:#d97706;
    classDef out fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
```

The critical invariant: **a ratio column auto-pulls its input columns**. You cannot compute net margin without a `net_income` column and a `revenue` column in the grid, so the catalog's `requires` (offline) and `_repair_dependencies` (LLM path) guarantee they are present. Columns are typed `numeric | ratio | text` and ordered numeric → ratio → text so figures fill before the ratios that consume them.

### Layers 7 & 8 — the two engines a cell can call

- **Layer 7 (Retrieval)** is now a *subroutine of a cell*: `retriever.retrieve(question, doc_id=…)` does RRF fusion of dense+sparse, then CrossEncoder rerank (or RRF order as fallback) — always within one document.
- **Layer 8 (Calc)** is the determinism boundary: `compute(formula, inputs)` runs a pure function from a fixed library of 11 formulas, returning `formula + inputs + expression + result`, or raising `CalcError` (a *refusal*).

### Layer 9 — Verify & Synthesize

```mermaid
flowchart LR
    GRID["filled Grid"]:::in
    subgraph V["verify_grid()"]
        CC["column consistency<br/>margins within ±100%"]:::chk
        RC["row identity checks<br/>gross profit ≤ revenue,<br/>equity ≤ assets, …"]:::chk
        CM["citation match<br/>stated value ∈ cited span"]:::chk
    end
    NOTES["VerificationNote[]<br/>ok | warning | error"]:::data
    SYN["synthesize() — tier=large<br/>cited narrative from grid only<br/>(deterministic fallback offline)"]:::llm
    OUT["narrative + flags"]:::out

    GRID --> CC & RC & CM --> NOTES --> SYN --> OUT
    classDef in fill:#e0f2fe,color:#075985,stroke:#0284c7;
    classDef chk fill:#fff,color:#27272a,stroke:#d4d4d8;
    classDef data fill:#f8fafc,color:#334155,stroke:#94a3b8;
    classDef llm fill:#fdf4ff,color:#86198f,stroke:#c026d3;
    classDef out fill:#f0fdf4,color:#166534,stroke:#16a34a;
```

Synthesis is constrained: *"use ONLY the grid below, cite each figure, do not compute new numbers."* The verification flags are passed in so the narrative can surface them.

### Layer 10 — Grid frontend

[`streamlit_app.py`](../app/streamlit_app.py): upload filings or point at a folder, ask a question, watch cells fill, then click any cell to reveal its **value · confidence · route · source span**. Routes are color-coded (🔵 store · 🟠 retrieval · 🟢 compute), mirrored in the CLI's ANSI grid.

---

## 8. The cell — atomic unit of work & audit

Everything good about the grid — parallelism, isolation, caching, traceability — falls out of this one structure ([`models.py`](../src/gridfin/models.py)).

```mermaid
classDiagram
    class Cell {
        +str doc_id
        +str column_id
        +str question
        +CellStatus status
        +Any value
        +str unit
        +str detail
        +Source source
        +float confidence
        +CellPath path
        +int cost_tokens
        +str error
        +display() str
        +key() tuple
    }
    note for Cell "status: pending|running|done|failed|empty\npath: numeric_store|retrieval_extract|compute\nsource: exactly one span · detail: e.g. '4,200 / 33,900'"
    class Source {
        +str file
        +int page
        +str section
        +tuple char_span
        +short() str
    }
    class Grid {
        +str question
        +ColumnSpec[] columns
        +DocumentRef[] rows
        +dict~str,Cell~ cells
        +str narrative
        +VerificationNote[] verification
        +from_plan(GridPlan)$ Grid
        +get(doc,col) Cell
        +column(col) Cell[]
        +row(doc) Cell[]
        +completion() float
    }
    class GridPlan {
        +str question
        +ColumnSpec[] columns
        +DocumentRef[] rows
        +shape() tuple
    }
    class ColumnSpec {
        +str column_id
        +str name
        +str question
        +ColumnType type
        +str formula
        +str[] depends_on
    }
    class VerificationNote {
        +str kind
        +str level
        +str message
        +tuple[] cells
    }

    Grid "1" *-- "many" Cell
    Cell "1" o-- "0..1" Source
    Grid "1" *-- "many" ColumnSpec
    Grid "1" *-- "many" VerificationNote
    GridPlan ..> Grid : from_plan()
    GridPlan "1" *-- "many" ColumnSpec
```

A cell is a **pure function of `(doc_id, column_id)`** — which is precisely why it caches so cleanly and why one document's data can never leak into another's.

---

## 9. Cell routing: the three engines

The heart of the fan-out. `CellEngine._execute_cell` checks the cache, then routes by column type. Each route either produces a `done`/`empty` cell or, on terminal failure, a `failed` cell — but **never an invented number**.

```mermaid
flowchart TB
    START(["execute cell (doc_id, column)"]):::entry
    CACHE{"cache hit?<br/>sig = hash(doc_id, question, type, formula, deps)"}:::dec
    HIT["return cached cell"]:::done

    TYPE{"column.type"}:::dec

    %% numeric
    subgraph NUM["type = numeric"]
        FAST{"in NumericStore?"}:::dec
        STORE["✓ pull fact — NO LLM<br/>path=numeric_store · conf=fact"]:::store
        RET1["scoped retrieval → scan_value()"]:::box
        FOUND{"figure found<br/>in a span?"}:::dec
        EXT["✓ extract figure<br/>path=retrieval_extract"]:::retr
        EMPTY1["∅ empty + flagged<br/>(never invented)"]:::empty
    end

    %% text
    subgraph TXT["type = text"]
        RET2["scoped retrieval (top hit)"]:::box
        LLMX["LLM extract (tier=small)<br/>found? value? — source only"]:::llm
        TOK{"found?"}:::dec
        TEXTOK["✓ qualitative answer<br/>path=retrieval_extract"]:::retr
        EMPTY2["∅ empty"]:::empty
    end

    %% ratio
    subgraph RAT["type = ratio"]
        SUPP{"formula supported?"}:::dec
        DEPS{"all depends_on cells<br/>done & numeric?"}:::dec
        CALC["✓ compute() — deterministic<br/>path=compute · conf=1.0"]:::comp
        REF1["✗ refused: unsupported formula"]:::ref
        REF2["✗ refused: missing input(s)"]:::ref
    end

    START --> CACHE
    CACHE -->|yes| HIT
    CACHE -->|no| TYPE

    TYPE -->|numeric| FAST
    FAST -->|yes| STORE
    FAST -->|no| RET1 --> FOUND
    FOUND -->|yes| EXT
    FOUND -->|no| EMPTY1

    TYPE -->|text| RET2 --> LLMX --> TOK
    TOK -->|yes| TEXTOK
    TOK -->|no| EMPTY2

    TYPE -->|ratio| SUPP
    SUPP -->|no| REF1
    SUPP -->|yes| DEPS
    DEPS -->|yes| CALC
    DEPS -->|no| REF2

    classDef entry fill:#1f2937,color:#fff,stroke:#111;
    classDef dec fill:#fef3c7,color:#92400e,stroke:#d97706;
    classDef store fill:#dbeafe,color:#1e40af,stroke:#2563eb;
    classDef retr fill:#ffedd5,color:#9a3412,stroke:#ea580c;
    classDef comp fill:#dcfce7,color:#166534,stroke:#16a34a;
    classDef empty fill:#f4f4f5,color:#71717a,stroke:#a1a1aa;
    classDef ref fill:#fee2e2,color:#991b1b,stroke:#dc2626;
    classDef done fill:#e0e7ff,color:#3730a3,stroke:#6366f1;
    classDef box fill:#fff,color:#27272a,stroke:#d4d4d8;
    classDef llm fill:#fdf4ff,color:#86198f,stroke:#c026d3;
```

| Route | Cost | When it fires | Guarantee |
|---|---|---|---|
| **`numeric_store`** | **No LLM** (fast path) | numeric metric already extracted at ingest | exact figure + its source span |
| **`retrieval_extract`** | small model (text) / none (numeric scan) | metric/text not pre-extracted | extracted from one document's span, or `empty` |
| **`compute`** | No LLM (pure Python) | ratio columns | `formula + inputs + result`, or `refused` |

Each cell is wrapped in **tenacity retry** (`AsyncRetrying`) that retries only *transient* errors — `_TransientCellError` plus 429/5xx API errors — and never retries 4xx client errors. Terminal failures are recorded as `status="failed"` with the error string.

---

## 10. Two-wave dependency execution

Ratio cells depend on figure cells, so `execute_grid` fills the grid in two dependency-respecting waves. **Both waves are fully parallel internally**, bounded by `asyncio.Semaphore(max_concurrency)`.

```mermaid
flowchart TB
    PLAN["Grid (empty cells)"]:::in
    SPLIT{"split columns by type"}:::dec

    subgraph W1["WAVE 1 — figures & text (parallel)"]
        direction LR
        F1["revenue cells"]:::num
        F2["net_income cells"]:::num
        F3["risk text cells"]:::txt
    end

    subgraph W2["WAVE 2 — ratios (parallel)"]
        direction LR
        R1["net_margin cells<br/>reads net_income, revenue"]:::comp
        R2["debt_to_equity cells<br/>reads total_debt, total_equity"]:::comp
    end

    PLAN --> SPLIT
    SPLIT -->|"type ≠ ratio"| W1
    W1 ==>|"barrier: all figures filled"| W2
    SPLIT -->|"type = ratio"| W2
    W2 --> FILLED["filled Grid → verify → synthesize"]:::out

    SEM["⚙ Semaphore(max_concurrency=8)<br/>caps in-flight cells across both waves"]:::sem
    SEM -.-> W1
    SEM -.-> W2

    classDef in fill:#e0f2fe,color:#075985,stroke:#0284c7;
    classDef dec fill:#fef3c7,color:#92400e,stroke:#d97706;
    classDef num fill:#fff7ed,color:#9a3412,stroke:#dd6b20;
    classDef txt fill:#faf5ff,color:#6b21a8,stroke:#9333ea;
    classDef comp fill:#dcfce7,color:#166534,stroke:#16a34a;
    classDef out fill:#f0fdf4,color:#166534,stroke:#16a34a;
    classDef sem fill:#f1f5f9,color:#334155,stroke:#64748b,stroke-dasharray:4 3;
```

The barrier between waves is the only synchronization point; within a wave, every `(row × column)` cell runs concurrently. The `on_cell` callback fires as each cell completes, which is how the UI streams cells in live.

---

## 11. Request sequence (full trace)

A complete `corpus.ask()` from question to cited answer.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Pipe as GridFin.ask()
    participant Dec as decompose (L5)
    participant Eng as CellEngine (L6)
    participant Cache as CellCache
    participant Store as NumericStore (L3)
    participant Ret as Retriever (L7)
    participant Calc as CalcEngine (L8)
    participant LLM as LLM (small/large)
    participant Ver as verify+synth (L9)

    User->>Pipe: ask("Compare net margin & D/E")
    Pipe->>Dec: decompose(question, scope, llm)
    alt live LLM
        Dec->>LLM: structured GridPlan (tier=large)
        LLM-->>Dec: columns × rows
        Dec->>Dec: _repair_dependencies()
    else offline
        Dec->>Dec: build_heuristic_plan()
    end
    Dec-->>Pipe: GridPlan
    Pipe->>Eng: execute_grid(Grid.from_plan)

    Note over Eng: WAVE 1 — figures & text (parallel, semaphore-bounded)
    loop each figure/text cell
        Eng->>Cache: get(doc, column)
        alt cache hit
            Cache-->>Eng: cell
        else miss · numeric
            Eng->>Store: get(doc, metric)
            alt fast path
                Store-->>Eng: NumericFact (NO LLM)
            else fallback
                Eng->>Ret: retrieve(q, doc_id)
                Ret-->>Eng: scoped spans → scan_value()
            end
        else miss · text
            Eng->>Ret: retrieve(q, doc_id)
            Eng->>LLM: structured extract (tier=small)
            LLM-->>Eng: {found, value, confidence}
        end
        Eng->>Cache: put(done/empty cell)
    end

    Note over Eng: WAVE 2 — ratios (parallel)
    loop each ratio cell
        Eng->>Eng: gather depends_on cells
        alt all inputs present
            Eng->>Calc: compute(formula, inputs)
            Calc-->>Eng: formula+inputs+result (conf=1.0)
        else missing input
            Eng-->>Eng: status=failed (refused)
        end
    end
    Eng-->>Pipe: filled Grid

    Pipe->>Ver: verify_grid() + synthesize()
    Ver->>LLM: narrative from grid only (tier=large)
    LLM-->>Ver: cited narrative
    Ver-->>Pipe: notes + narrative
    Pipe-->>User: GridAnswer{ grid, narrative }
```

---

## 12. The D × C cost tax & its four mitigations

A grid issues up to **D documents × C columns** model calls, which is the main cost concern. All four mitigations are implemented (`config.py`, `cell_engine.py`, `cell_cache.py`).

```mermaid
flowchart LR
    TAX["D × C potential model calls"]:::tax

    M1["① Concurrency cap<br/>Semaphore(max_concurrency)<br/>GRIDFIN_MAX_CONCURRENCY"]:::mit
    M2["② Cell cache<br/>pure fn of (doc, sub-question)<br/>edit one column → recompute one column"]:::mit
    M3["③ Model routing<br/>simple → small (Haiku)<br/>hard / plan / synth → large (Opus)"]:::mit
    M4["④ Numeric fast path<br/>store-answerable cells skip the LLM"]:::mit

    TAX --> M1 & M2 & M3 & M4 --> CHEAP["common case: fast & cheap"]:::out

    classDef tax fill:#fee2e2,color:#991b1b,stroke:#dc2626;
    classDef mit fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
    classDef out fill:#dcfce7,color:#166534,stroke:#16a34a;
```

- **Cache signature** deliberately includes column *content* (`question`, `type`, `formula`, `depends_on`), so editing a column's question invalidates only that column's cells. Transient failures are never cached.
- **Routing knobs** live in `Settings`: `model_small` (Haiku 4.5), `model_large` (Opus 4.8), `max_concurrency=8`, `cell_max_retries=3`, `retrieval_k=8`, `rerank_top_n=4`, `rrf_k=60`.

---

## 13. The determinism boundary

The deterministic calculation boundary is the rule I care most about keeping: the
model handles language, but never arithmetic.

```mermaid
flowchart TB
    subgraph LLM_SIDE["LLM may touch (language)"]
        direction LR
        A["planning the grid"]:::llm
        B["locating / extracting a figure's span"]:::llm
        C["qualitative text answers"]:::llm
        D["writing the final narrative"]:::llm
    end

    BOUNDARY{{"═══  DETERMINISM BOUNDARY  ═══<br/>exact figures cross here, in only"}}:::wall

    subgraph DET_SIDE["LLM may NEVER touch (numbers)"]
        direction LR
        E["NumericStore figures<br/>(validated at ingest)"]:::det
        F["calc/engine.py<br/>11 pure formulas"]:::det
        G["refusals<br/>missing input · unsupported formula · ÷0"]:::det
    end

    LLM_SIDE --> BOUNDARY --> DET_SIDE
    DET_SIDE --> OUT["formula + inputs + expression + result<br/>or an explicit refusal"]:::out

    classDef llm fill:#fdf4ff,color:#86198f,stroke:#c026d3;
    classDef wall fill:#1f2937,color:#fff,stroke:#000;
    classDef det fill:#dcfce7,color:#166534,stroke:#16a34a;
    classDef out fill:#f0fdf4,color:#166534,stroke:#16a34a;
```

**Contract enforced by `calc/engine.py`:**
- Unsupported formulas → `CalcError` (refused, never approximated).
- Missing / non-numeric inputs → `CalcError` (refused, never guessed).
- `revenue == 0` or any division by zero → refused.
- Every success returns a full audit trail: `formula · inputs · human-readable expression · result · unit`.

Supported formulas: `net_profit_margin`, `gross_margin`, `operating_margin`, `yoy_growth`, `cagr`, `current_ratio`, `debt_to_equity`, `return_on_equity`, `return_on_assets`, `eps` (+ aliases like `roe`, `roa`, `leverage`).

---

## 14. Configuration & deployment modes

GridFin is a **runnable reference implementation**: all ten layers are present, but heavy dependencies degrade gracefully (`config.py` capability flags) so the demo runs with no GPU or model download.

```mermaid
flowchart LR
    subgraph REF["Reference (default · pip install -e .)"]
        R1["hashed bag-of-words (numpy)"]:::ref
        R2["brute-force numpy cosine"]:::ref
        R3["RRF order (no rerank)"]:::ref
        R4["deterministic mock LLM"]:::ref
        R5["txt / csv ingestion"]:::ref
    end
    subgraph HEAVY["Faithful heavy stack (extras)"]
        H1["BGE via sentence-transformers — [ml]"]:::heavy
        H2["FAISS — [ml]"]:::heavy
        H3["CrossEncoder rerank — [ml]"]:::heavy
        H4["Claude / OpenAI — set API key"]:::heavy
        H5["pdfplumber / PyMuPDF / docx — [ingest]"]:::heavy
    end
    R1 -.->|"upgrade"| H1
    R2 -.-> H2
    R3 -.-> H3
    R4 -.-> H4
    R5 -.-> H5

    classDef ref fill:#f1f5f9,color:#334155,stroke:#64748b;
    classDef heavy fill:#ecfeff,color:#155e75,stroke:#0891b2;
```

| Mode | Command | LLM | Use |
|---|---|---|---|
| **Offline demo** | `pip install -e .` | deterministic mock | CI, dev, no-network demo |
| **Live models** | copy `.env.example` → `.env`, set key | Claude (Haiku+Opus) / OpenAI | real planning, extraction, synthesis |
| **Heavy retrieval** | `pip install -e ".[ml]"` | — | BGE + FAISS + CrossEncoder |
| **Real documents** | `pip install -e ".[ingest]"` | — | PDF / DOCX / XLSX parsing |
| **Interactive UI** | `pip install -e ".[ui]"` then `streamlit run app/streamlit_app.py` | any | grid with click-through |

Provider/model routing is selected by `llm_provider` (`anthropic`|`openai`); the TLS client is pinned to certifi's CA bundle to avoid macOS `CERTIFICATE_VERIFY_FAILED`.

---

## 15. Evaluation harness

`gridfin eval` ([`eval/metrics.py`](../src/gridfin/eval/metrics.py)) reports the metrics a financial tool actually needs — including the one none of the four RAGAS metrics capture: *is the number right.*

```mermaid
flowchart LR
    GRID["filled Grid"]:::in
    GT["ground_truth.json<br/>{doc_id:{column_id:expected}}"]:::in

    CNA["Cell Numeric Accuracy<br/>cells == truth within tol"]:::m1
    GC["Grid Completion<br/>fraction filled (not failed/empty)"]:::m2
    AC["Attribution Correctness<br/>cited span contains stated value"]:::m3
    RAGAS["RAGAS (optional [eval])<br/>faithfulness · relevancy · precision · recall"]:::m4

    GRID --> GC & AC & CNA
    GT --> CNA
    GRID -.-> RAGAS
    CNA & GC & AC --> REPORT["GridEvalReport"]:::out

    classDef in fill:#e0f2fe,color:#075985,stroke:#0284c7;
    classDef m1 fill:#dcfce7,color:#166534,stroke:#16a34a;
    classDef m2 fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
    classDef m3 fill:#fff7ed,color:#9a3412,stroke:#dd6b20;
    classDef m4 fill:#faf5ff,color:#6b21a8,stroke:#9333ea;
    classDef out fill:#f0fdf4,color:#166534,stroke:#16a34a;
```

---

## 16. Code map

```
src/gridfin/
  models.py            # grid data model: Cell, Grid, GridPlan, Source, ColumnSpec
  config.py            # settings + optional-dependency capability flags
  pipeline.py          # GridFin: orchestrates all 10 layers (from_path → ask)
  cli.py               # gridfin info | ask | eval  (+ ANSI route-colored grid)
  llm/
    client.py          #   tiered structured-output client (Anthropic | OpenAI)
    mock.py            #   deterministic offline backend
  ingestion/ingest.py  # L1  format-aware parse → ParsedDocument(blocks)
  chunking/chunker.py  # L2  tables whole · prose windowed
  extraction/
    numeric_store.py   # L3  regex figures → NumericStore (cell fast path)
  indexing/
    dual_index.py      # L4  dense + sparse, per-doc filter
    embeddings.py      #     BGE / hashed-BoW embedder
  grid/
    decompose.py       # L5  question → GridPlan
    cell_engine.py     # L6  fan-out: 3 routes, 2 waves, semaphore, retry
    verify.py          # L9  cross-cell checks + synthesize
    store.py           # grid → pandas → CSV export
  retrieval/retriever.py  # L7  RRF + rerank, scoped to one doc
  calc/engine.py       # L8  deterministic formula library (the boundary)
  cache/cell_cache.py  # cell-level diskcache
  eval/metrics.py      # numeric accuracy · completion · attribution
app/streamlit_app.py   # L10 interactive grid (streamlit-aggrid)
data/demo/             # 3 demo filings + ground_truth.json
data/sample/           # extra filings (txt + csv)
tests/                 # extraction · indexing · calc · pipeline · eval
docs/                  # this document
```

---

## 17. Implementation plan & roadmap

The architecture is fully implemented as a reference. This plan organizes **hardening and productionization** into phases. Status reflects the current `Initial commit` baseline.

```mermaid
flowchart LR
    P0["Phase 0 ✅<br/>Reference impl<br/>10 layers · mock mode · CLI · UI · tests"]:::done
    P1["Phase 1<br/>Faithful heavy stack<br/>BGE · FAISS · CrossEncoder · live LLM e2e"]:::next
    P2["Phase 2<br/>Real-document robustness<br/>PDF/DOCX tables · multi-period · sectioning"]:::next
    P3["Phase 3<br/>Scale & cost<br/>persistent grid · streaming UI · batch eval"]:::later
    P4["Phase 4<br/>Trust & coverage<br/>more formulas · richer verify · RAGAS gates"]:::later
    P5["Phase 5<br/>Productionize<br/>API service · auth · observability · multi-tenant"]:::later

    P0 --> P1 --> P2 --> P3 --> P4 --> P5
    classDef done fill:#dcfce7,color:#166534,stroke:#16a34a;
    classDef next fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
    classDef later fill:#f1f5f9,color:#334155,stroke:#64748b;
```

### Phase 0 — Reference implementation ✅ (done)
- All ten layers + cross-cutting cache/eval/store present and wired in `pipeline.py`.
- Offline deterministic mode (mock LLM, hashed embeddings, RRF-only) runs end-to-end.
- CLI (`info`/`ask`/`eval`), Streamlit grid, and unit + integration tests in place.

### Phase 1 — Faithful heavy stack & live LLM
- **Goal:** make the `[ml]` path and live Claude/OpenAI path first-class, not just fallbacks.
- Validate BGE embeddings + FAISS ANN + CrossEncoder rerank against the RRF-only baseline on the demo set.
- End-to-end live-LLM runs for decompose → extract → synthesize; capture token cost per grid.
- Add a smoke test that runs `ask()` against a mocked Anthropic client (assert routing: small for text cells, large for planning/synthesis).
- **Exit:** Cell Numeric Accuracy ≥ baseline with live models; documented cost-per-grid.

### Phase 2 — Real-document robustness
- Harden table extraction for messy PDFs/DOCX (merged cells, multi-column layouts, footnotes).
- Multi-period awareness: today a figure carries an optional `period`; extend the store + grid to expose year-over-year columns (`yoy_growth`, `cagr`) end-to-end.
- Improve `_section_for` and char-span fidelity so citations land on the exact line.
- **Exit:** correct grids on a held-out set of 10+ real 10-Ks.

### Phase 3 — Scale & cost
- Persist filled grids to a queryable store and add a `gridfin query` command for filtering cells (by route, confidence, document).
- True streaming UI: push `on_cell` callbacks to the browser as cells complete (currently filled then rendered).
- Batch evaluation across a question suite with regression tracking.
- **Exit:** a 50-doc × 8-column grid completes within target wall-clock and cost budgets.

### Phase 4 — Trust & coverage
- Expand the formula library (coverage ratios, margins, turnover, EBITDA-derived metrics) — each pure and refusable.
- Richer verification: more accounting identities, cross-document outlier detection, unit-mismatch detection.
- Wire RAGAS as a CI gate behind `[eval]`.
- **Exit:** Attribution Correctness ≥ 0.95 on the demo + sample corpora.

### Phase 5 — Productionize
- Wrap `GridFin` in an async API service (FastAPI) with corpus lifecycle, auth, and per-tenant cache namespaces (the UI already namespaces the cache per upload signature).
- Observability: per-cell timing/cost metrics, structured logs, a cost dashboard for the D×C tax.
- **Exit:** multi-tenant deployment with isolation and cost controls.

---

## 18. Risks & mitigations

| Risk | Impact | Mitigation (status) |
|---|---|---|
| **D × C cost blowup** on large corpora | high | Semaphore + cell cache + model routing + numeric fast path (✅ implemented); cost dashboard (Phase 3/5). |
| **Wrong number reaches the answer** | critical | Determinism boundary: numbers only from store/calc; refusals on missing inputs (✅). Attribution checks in verify (✅). |
| **Bad decomposition** (planner emits wrong/missing columns) | high | `_repair_dependencies` guarantees ratio inputs exist; heuristic fallback; planner uses the frontier model (✅). Add planner eval (Phase 1). |
| **Cross-document contamination** | critical | Mandatory per-doc filter in `DualIndex.search` — no cross-doc search path exists (✅). |
| **Messy real-world tables** mis-extracted | med | Table confidence scoring; tables-over-prose preference; harden in Phase 2. |
| **Stale cache after content change** | med | Cache signature keyed on column content; UI namespaces cache per upload content hash (✅). |
| **API flakiness** | med | Tenacity retry on 429/5xx only; never on 4xx; terminal failures recorded, not invented (✅). |

---

## 19. Testing strategy

Current suite (`tests/`): `test_extraction`, `test_indexing`, `test_calc`, `test_pipeline`, `test_eval` — run with `pytest` (async mode enabled in `pyproject.toml`).

```mermaid
flowchart TB
    subgraph UNIT["Unit"]
        T1["calc: every formula + every refusal path"]:::t
        T2["extraction: figures, suffixes, negatives, vocab"]:::t
        T3["indexing: per-doc filter isolation"]:::t
    end
    subgraph INTEG["Integration"]
        T4["pipeline: from_path → ask end-to-end (mock)"]:::t
        T5["eval: numeric accuracy vs ground_truth.json"]:::t
    end
    subgraph FUTURE["Planned"]
        T6["live-LLM smoke (mocked SDK) — routing assertions"]:::f
        T7["real-PDF extraction fixtures (Phase 2)"]:::f
        T8["RAGAS CI gate (Phase 4)"]:::f
    end
    UNIT --> INTEG --> FUTURE
    classDef t fill:#dcfce7,color:#166534,stroke:#16a34a;
    classDef f fill:#eff6ff,color:#1e40af,stroke:#3b82f6;
```

**Invariants worth a dedicated test at every phase:**
1. A missing figure yields `status="empty"`, never a fabricated number.
2. A ratio with a missing input yields `status="failed"` (refused), never a computed guess.
3. Retrieval for `doc_id=A` never returns a chunk from `doc_id=B`.
4. Editing one column's question is a cache miss for that column only.

---

*The diagrams show the design shape; exact ranking, chunking, and prompt details
live in the code. Worth updating these alongside the code when things change.*
