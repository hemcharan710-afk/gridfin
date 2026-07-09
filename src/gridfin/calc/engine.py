"""Deterministic calculation engine: exact figures in, formula + result out.

The model never does arithmetic; ratios are computed here in plain Python so the
numbers are reproducible and auditable.

Rules:
  - Unsupported formulas raise CalcError instead of being approximated.
  - Missing or non-numeric inputs raise CalcError instead of being guessed.
  - Every result carries its formula name, the inputs used, a readable expression,
    and the numeric result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel


class CalcError(Exception):
    """Raised when a formula is unsupported or its inputs are missing/invalid."""


class CalcResult(BaseModel):
    formula: str
    inputs: dict[str, float]
    result: float
    expression: str
    unit: str = ""


@dataclass(frozen=True)
class Formula:
    name: str
    inputs: tuple[str, ...]
    fn: Callable[..., float]
    render: Callable[..., str]
    unit: str = ""
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# --- Formula library -------------------------------------------------------
# Each formula is a pure function over named numeric inputs.

_DEFS: list[Formula] = [
    Formula(
        name="net_profit_margin",
        inputs=("net_income", "revenue"),
        fn=lambda net_income, revenue: net_income / revenue,
        render=lambda net_income, revenue, r: f"{net_income:,.0f} / {revenue:,.0f} = {_pct(r)}",
        unit="%",
        description="Net income divided by revenue.",
        aliases=("profit_margin", "net_margin"),
    ),
    Formula(
        name="gross_margin",
        inputs=("gross_profit", "revenue"),
        fn=lambda gross_profit, revenue: gross_profit / revenue,
        render=lambda gross_profit, revenue, r: f"{gross_profit:,.0f} / {revenue:,.0f} = {_pct(r)}",
        unit="%",
        description="Gross profit divided by revenue.",
    ),
    Formula(
        name="operating_margin",
        inputs=("operating_income", "revenue"),
        fn=lambda operating_income, revenue: operating_income / revenue,
        render=lambda operating_income, revenue, r: f"{operating_income:,.0f} / {revenue:,.0f} = {_pct(r)}",
        unit="%",
        description="Operating income divided by revenue.",
    ),
    Formula(
        name="yoy_growth",
        inputs=("current", "prior"),
        fn=lambda current, prior: (current - prior) / prior,
        render=lambda current, prior, r: f"({current:,.0f} - {prior:,.0f}) / {prior:,.0f} = {_pct(r)}",
        unit="%",
        description="Year-over-year growth between two periods.",
        aliases=("growth", "yoy"),
    ),
    Formula(
        name="cagr",
        inputs=("begin", "end", "years"),
        fn=lambda begin, end, years: (end / begin) ** (1.0 / years) - 1.0,
        render=lambda begin, end, years, r: f"({end:,.0f}/{begin:,.0f})^(1/{years:g}) - 1 = {_pct(r)}",
        unit="%",
        description="Compound annual growth rate over N years.",
    ),
    Formula(
        name="current_ratio",
        inputs=("current_assets", "current_liabilities"),
        fn=lambda current_assets, current_liabilities: current_assets / current_liabilities,
        render=lambda current_assets, current_liabilities, r: f"{current_assets:,.0f} / {current_liabilities:,.0f} = {r:.2f}",
        unit="x",
        description="Current assets divided by current liabilities.",
    ),
    Formula(
        name="debt_to_equity",
        inputs=("total_debt", "total_equity"),
        fn=lambda total_debt, total_equity: total_debt / total_equity,
        render=lambda total_debt, total_equity, r: f"{total_debt:,.0f} / {total_equity:,.0f} = {r:.2f}",
        unit="x",
        description="Total debt divided by shareholders' equity.",
        aliases=("d_e", "leverage"),
    ),
    Formula(
        name="return_on_equity",
        inputs=("net_income", "total_equity"),
        fn=lambda net_income, total_equity: net_income / total_equity,
        render=lambda net_income, total_equity, r: f"{net_income:,.0f} / {total_equity:,.0f} = {_pct(r)}",
        unit="%",
        description="Net income divided by shareholders' equity.",
        aliases=("roe",),
    ),
    Formula(
        name="return_on_assets",
        inputs=("net_income", "total_assets"),
        fn=lambda net_income, total_assets: net_income / total_assets,
        render=lambda net_income, total_assets, r: f"{net_income:,.0f} / {total_assets:,.0f} = {_pct(r)}",
        unit="%",
        description="Net income divided by total assets.",
        aliases=("roa",),
    ),
    Formula(
        name="eps",
        inputs=("net_income", "shares_outstanding"),
        fn=lambda net_income, shares_outstanding: net_income / shares_outstanding,
        render=lambda net_income, shares_outstanding, r: f"{net_income:,.0f} / {shares_outstanding:,.0f} = {r:.2f}",
        unit="$",
        description="Earnings per share.",
    ),
]

FORMULAS: dict[str, Formula] = {}
for _f in _DEFS:
    FORMULAS[_f.name] = _f
    for _alias in _f.aliases:
        FORMULAS[_alias] = _f


def supported_formulas() -> list[str]:
    # Canonical names only (skip aliases).
    return [f.name for f in _DEFS]


def _coerce(value: object, name: str) -> float:
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise CalcError(f"input '{name}' must be numeric, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            raise CalcError(f"input '{name}' is not numeric: {value!r}")
    raise CalcError(f"input '{name}' is not numeric: {value!r}")


def compute(formula: str, inputs: dict[str, object]) -> CalcResult:
    """Run a supported formula on exact inputs, or raise CalcError.

    A missing figure is never filled in with a guess; the calculation is refused
    instead.
    """
    spec = FORMULAS.get(formula) or FORMULAS.get(formula.lower())
    if spec is None:
        raise CalcError(
            f"unsupported formula '{formula}'. Supported: {', '.join(supported_formulas())}"
        )

    missing = [k for k in spec.inputs if k not in inputs or inputs[k] is None]
    if missing:
        raise CalcError(f"refused: missing input(s) {missing} for '{spec.name}'")

    coerced = {k: _coerce(inputs[k], k) for k in spec.inputs}

    if "revenue" in coerced and coerced["revenue"] == 0:
        raise CalcError("refused: division by zero (revenue == 0)")
    try:
        result = spec.fn(**coerced)
    except ZeroDivisionError:
        raise CalcError("refused: division by zero")

    return CalcResult(
        formula=spec.name,
        inputs=coerced,
        result=result,
        expression=spec.render(**coerced, r=result),
        unit=spec.unit,
    )
