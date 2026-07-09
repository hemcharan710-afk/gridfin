"""Turn a question into a grid: pick the columns and rows.

An LLM planner reads the question and document set and emits a column spec (the
sub-questions/fields) and a row spec (the documents in scope) as structured
Pydantic output.

Offline, a heuristic stands in for the planner: it maps financial phrasing in the
question to numeric, ratio, and trend columns, and auto-adds the input columns a
ratio depends on (net margin needs a net-income column and a revenue column).
"""

from __future__ import annotations

from dataclasses import dataclass

from gridfin.ingestion import ParsedDocument
from gridfin.llm import LLM
from gridfin.models import ColumnSpec, DocumentRef, GridPlan


@dataclass(frozen=True)
class _ColumnTemplate:
    keywords: tuple[str, ...]
    spec: ColumnSpec
    # Numeric input metrics this column needs as their own columns (for ratios).
    requires: tuple[tuple[str, str], ...] = ()  # (column_id, metric question)


def _num_col(metric_id: str, name: str, question: str) -> ColumnSpec:
    return ColumnSpec(column_id=metric_id, name=name, question=question, type="numeric")


# Ratio templates pull their inputs from the numeric store via dependency columns.
_CATALOG: list[_ColumnTemplate] = [
    _ColumnTemplate(
        ("net profit margin", "net margin", "profit margin", "profitability"),
        ColumnSpec(
            column_id="net_profit_margin",
            name="Net profit margin",
            question="What is the net profit margin?",
            type="ratio",
            formula="net_profit_margin",
            depends_on=["net_income", "revenue"],
        ),
        requires=(
            ("net_income", "What is net income / net earnings?"),
            ("revenue", "What is total revenue / net sales?"),
        ),
    ),
    _ColumnTemplate(
        ("gross margin",),
        ColumnSpec(
            column_id="gross_margin",
            name="Gross margin",
            question="What is the gross margin?",
            type="ratio",
            formula="gross_margin",
            depends_on=["gross_profit", "revenue"],
        ),
        requires=(
            ("gross_profit", "What is gross profit?"),
            ("revenue", "What is total revenue / net sales?"),
        ),
    ),
    _ColumnTemplate(
        ("current ratio", "liquidity"),
        ColumnSpec(
            column_id="current_ratio",
            name="Current ratio",
            question="What is the current ratio?",
            type="ratio",
            formula="current_ratio",
            depends_on=["current_assets", "current_liabilities"],
        ),
        requires=(
            ("current_assets", "What are total current assets?"),
            ("current_liabilities", "What are total current liabilities?"),
        ),
    ),
    _ColumnTemplate(
        ("debt to equity", "debt-to-equity", "leverage", "d/e"),
        ColumnSpec(
            column_id="debt_to_equity",
            name="Debt / equity",
            question="What is the debt-to-equity ratio?",
            type="ratio",
            formula="debt_to_equity",
            depends_on=["total_debt", "total_equity"],
        ),
        requires=(
            ("total_debt", "What is total debt?"),
            ("total_equity", "What is total shareholders' equity?"),
        ),
    ),
    _ColumnTemplate(
        ("return on equity", "roe"),
        ColumnSpec(
            column_id="return_on_equity",
            name="Return on equity",
            question="What is the return on equity?",
            type="ratio",
            formula="return_on_equity",
            depends_on=["net_income", "total_equity"],
        ),
        requires=(
            ("net_income", "What is net income / net earnings?"),
            ("total_equity", "What is total shareholders' equity?"),
        ),
    ),
]

# Plain numeric metric phrases (no formula).
_NUMERIC_PHRASES: list[tuple[tuple[str, ...], ColumnSpec]] = [
    (("net income", "net earnings", "net profit"), _num_col("net_income", "Net income", "What is net income?")),
    (("revenue", "net sales", "total sales", "top line"), _num_col("revenue", "Revenue", "What is total revenue?")),
    (("gross profit",), _num_col("gross_profit", "Gross profit", "What is gross profit?")),
    (("operating income", "operating profit"), _num_col("operating_income", "Operating income", "What is operating income?")),
    (("total assets",), _num_col("total_assets", "Total assets", "What are total assets?")),
    (("total equity", "shareholders' equity", "stockholders' equity"), _num_col("total_equity", "Total equity", "What is total equity?")),
    (("cash",), _num_col("cash", "Cash", "What is cash and cash equivalents?")),
    (("r&d", "research and development"), _num_col("rd_expense", "R&D expense", "What is research and development expense?")),
]

# Retrieval/text trend phrases.
_TREND_PHRASES: list[tuple[tuple[str, ...], ColumnSpec]] = [
    (
        ("trend", "growth", "cagr", "year over year", "yoy", "over the", "3yr", "3-year"),
        ColumnSpec(
            column_id="revenue_trend",
            name="Revenue trend",
            question="Summarize the multi-year revenue trend, with a growth rate if stated.",
            type="text",
        ),
    ),
    (
        ("risk", "risks"),
        ColumnSpec(
            column_id="key_risks",
            name="Key risks",
            question="What are the key risks disclosed?",
            type="text",
        ),
    ),
    (
        ("outlook", "guidance", "forecast"),
        ColumnSpec(
            column_id="outlook",
            name="Outlook / guidance",
            question="What forward-looking outlook or guidance is given?",
            type="text",
        ),
    ),
]


def build_heuristic_plan(question: str, docs: list[ParsedDocument]) -> GridPlan:
    q = question.lower()
    columns: dict[str, ColumnSpec] = {}

    def add(spec: ColumnSpec) -> None:
        columns.setdefault(spec.column_id, spec)

    # Ratio templates first (they pull in their dependency columns).
    for tmpl in _CATALOG:
        if any(kw in q for kw in tmpl.keywords):
            for dep_id, dep_q in tmpl.requires:
                add(_num_col(dep_id, dep_id.replace("_", " ").title(), dep_q))
            add(tmpl.spec)

    for phrases, spec in _NUMERIC_PHRASES:
        if any(p in q for p in phrases):
            add(spec)

    for phrases, spec in _TREND_PHRASES:
        if any(p in q for p in phrases):
            add(spec)

    # Fallback: if nothing matched, ask each document the question directly as text.
    if not columns:
        add(ColumnSpec(column_id="answer", name="Answer", question=question, type="text"))

    rows = [DocumentRef(doc_id=d.doc_id, title=d.title) for d in docs]
    # Stable ordering: numeric inputs, then ratios, then text.
    order = {"numeric": 0, "ratio": 1, "text": 2}
    ordered = sorted(columns.values(), key=lambda c: (order[c.type], c.column_id))
    return GridPlan(question=question, columns=ordered, rows=rows)


def _repair_dependencies(plan: GridPlan) -> GridPlan:
    """Guarantee every ratio's depends_on columns exist (LLM plans can omit them)."""
    have = {c.column_id for c in plan.columns}
    extra: list[ColumnSpec] = []
    for col in plan.columns:
        for dep in col.depends_on:
            if dep not in have:
                extra.append(_num_col(dep, dep.replace("_", " ").title(), f"What is {dep.replace('_', ' ')}?"))
                have.add(dep)
    if extra:
        order = {"numeric": 0, "ratio": 1, "text": 2}
        plan.columns = sorted([*extra, *plan.columns], key=lambda c: (order[c.type], c.column_id))
    return plan


async def decompose(
    question: str, docs: list[ParsedDocument], llm: LLM
) -> GridPlan:
    heuristic = build_heuristic_plan(question, docs)

    if not llm.live:
        return heuristic

    doc_lines = "\n".join(f"- {d.doc_id}: {d.title}" for d in docs)
    prompt = (
        "Decompose this analytical question into a GRID for financial-document "
        "analysis.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"DOCUMENTS IN SCOPE (rows):\n{doc_lines}\n\n"
        "Emit COLUMNS as scoped sub-questions. Use type 'numeric' for a single "
        "reported figure, 'ratio' for a computed metric (set `formula` to one of: "
        "net_profit_margin, gross_margin, operating_margin, current_ratio, "
        "debt_to_equity, return_on_equity, return_on_assets, eps, cagr, yoy_growth; "
        "and list its input columns in `depends_on`), and 'text' for a qualitative "
        "answer. Every ratio's depends_on columns MUST also appear as numeric "
        "columns. Include every document as a row."
    )
    plan, _tokens = await llm.structured(
        prompt,
        GridPlan,
        tier="large",  # planning is the highest-leverage call — use the frontier model
        context={"mock": heuristic.model_dump()},
    )
    # The LLM owns `question`/`rows` truthfully; trust but repair structurally.
    plan.question = question
    if not plan.rows:
        plan.rows = heuristic.rows
    return _repair_dependencies(plan)
