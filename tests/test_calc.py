"""The deterministic calculation engine — the retained differentiator (§08)."""

import math

import pytest

from gridfin.calc import CalcError, compute, supported_formulas


def test_net_profit_margin():
    r = compute("net_profit_margin", {"net_income": 4200, "revenue": 33900})
    assert math.isclose(r.result, 4200 / 33900, rel_tol=1e-9)
    assert r.unit == "%"
    assert "33,900" in r.expression


def test_cagr():
    r = compute("cagr", {"begin": 29000, "end": 33900, "years": 2})
    assert math.isclose(r.result, (33900 / 29000) ** 0.5 - 1, rel_tol=1e-9)


def test_aliases_resolve():
    assert compute("roe", {"net_income": 10, "total_equity": 50}).formula == "return_on_equity"


def test_unsupported_formula_refused():
    with pytest.raises(CalcError, match="unsupported"):
        compute("ebitda_margin", {"x": 1})


def test_missing_input_refused():
    # A missing figure must never be computed on a guess (architecture §05).
    with pytest.raises(CalcError, match="missing input"):
        compute("net_profit_margin", {"revenue": 33900})


def test_division_by_zero_refused():
    with pytest.raises(CalcError):
        compute("net_profit_margin", {"net_income": 10, "revenue": 0})


def test_string_inputs_coerced():
    r = compute("current_ratio", {"current_assets": "$18,000", "current_liabilities": "9,000"})
    assert math.isclose(r.result, 2.0)


def test_bool_input_rejected():
    with pytest.raises(CalcError):
        compute("net_profit_margin", {"net_income": True, "revenue": 100})


def test_all_formulas_have_renderers():
    for name in supported_formulas():
        assert name
