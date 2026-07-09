"""End-to-end grid pipeline — reproduces the §05 worked example."""

import math


from conftest import ACME, BETA, GAMMA


async def test_worked_example_grid(corpus):
    answer = await corpus.ask("Compare net profit margin across these filings")
    grid = answer.grid

    # Acme: net income $4,200M, margin 12.4%, route store + compute.
    acme_ni = grid.get(ACME, "net_income")
    assert acme_ni.path == "numeric_store"
    assert abs(acme_ni.value - 4_200_000_000) < 1
    acme_margin = grid.get(ACME, "net_profit_margin")
    assert acme_margin.path == "compute"
    assert math.isclose(acme_margin.value, 4200 / 33900, rel_tol=1e-3)

    # Beta: 9.0%.
    assert math.isclose(grid.get(BETA, "net_profit_margin").value, 1910 / 21200, rel_tol=1e-3)


async def test_missing_figure_is_empty_not_invented(corpus):
    answer = await corpus.ask("Compare net profit margin across these filings")
    grid = answer.grid
    # Gamma omits net income → cell empty, and the dependent margin is refused.
    gamma_ni = grid.get(GAMMA, "net_income")
    assert gamma_ni.status == "empty"
    assert gamma_ni.value is None
    gamma_margin = grid.get(GAMMA, "net_profit_margin")
    assert gamma_margin.status == "failed"
    assert "missing input" in (gamma_margin.error or "")


async def test_every_done_cell_has_one_source(corpus):
    answer = await corpus.ask("revenue and current ratio")
    for cell in answer.grid.cells.values():
        if cell.status == "done" and cell.path != "compute":
            assert cell.source is not None  # exactly one span backs every answer


async def test_cache_hits_on_rerun(corpus):
    await corpus.ask("revenue and net income")
    before = corpus.cache.stats()["hits"]
    await corpus.ask("revenue and net income")
    assert corpus.cache.stats()["hits"] > before


async def test_ratio_columns_pull_in_dependencies(corpus):
    # Asking only for the ratio must still produce its numeric input columns.
    answer = await corpus.ask("net profit margin")
    col_ids = {c.column_id for c in answer.grid.columns}
    assert {"net_income", "revenue", "net_profit_margin"} <= col_ids
