"""GridFin command-line interface.

    gridfin info  [--data data/demo]
    gridfin ask   "Compare net profit margin across these filings" [--data ...] [--csv out.csv]
    gridfin eval  [--data data/demo] [--truth data/demo/ground_truth.json]

Runs fully offline (deterministic mock LLM) unless ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from gridfin.config import get_settings
from gridfin.eval import evaluate_grid
from gridfin.grid.store import export_csv
from gridfin.models import Grid
from gridfin.pipeline import GridFin

_DEFAULT_DATA = "data/demo"

# ANSI colours marking which route filled each cell.
_ROUTE_COLOR = {
    "numeric_store": "\033[38;5;39m",   # blue  — store (no LLM)
    "retrieval_extract": "\033[38;5;208m",  # orange — retrieval + extract
    "compute": "\033[38;5;42m",         # green — deterministic compute
}
_RESET = "\033[0m"
_DIM = "\033[2m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _render_grid(grid: Grid) -> str:
    color = _supports_color()
    headers = ["Document", *[c.name for c in grid.columns]]
    rows = []
    for row in grid.rows:
        cells = [row.title or row.doc_id]
        for col in grid.columns:
            cell = grid.get(row.doc_id, col.column_id)
            text = cell.display()
            if len(text) > 52:  # keep qualitative cells readable in the terminal grid
                text = text[:49].rstrip() + "…"
            if color and cell.path in _ROUTE_COLOR and cell.status == "done":
                text = f"{_ROUTE_COLOR[cell.path]}{text}{_RESET}"
            cells.append(text)
        rows.append(cells)

    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(_strip_ansi(c)))  # ignore ANSI in width

    def fmt_row(cells: list[str]) -> str:
        out = []
        for i, c in enumerate(cells):
            pad = widths[i] - len(_strip_ansi(c))
            out.append(c + " " * pad)
        return "  ".join(out)

    lines = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def _strip_ansi(s: str) -> str:
    import re

    return re.sub(r"\033\[[0-9;]*m", "", s)


def _print_legend() -> None:
    if _supports_color():
        print(
            f"\nRoute legend:  "
            f"{_ROUTE_COLOR['numeric_store']}numeric store (no LLM){_RESET}   "
            f"{_ROUTE_COLOR['retrieval_extract']}retrieval + extract{_RESET}   "
            f"{_ROUTE_COLOR['compute']}deterministic compute{_RESET}"
        )


async def _cmd_ask(args) -> int:
    corpus = GridFin.from_path(args.data)
    answer = await corpus.ask(args.question, doc_ids=args.doc_ids)
    grid = answer.grid

    if args.json:
        print(grid.model_dump_json(indent=2))
        return 0

    print(f"\nQuestion: {grid.question}")
    print(f"Grid shape: {grid.shape() if hasattr(grid, 'shape') else (len(grid.rows), len(grid.columns))}  "
          f"·  completion {grid.completion():.0%}  ·  LLM {corpus.info()['llm']}\n")
    print(_render_grid(grid))
    _print_legend()

    if grid.verification:
        flagged = [n for n in grid.verification if n.level in ("warning", "error")]
        if flagged:
            print("\nVerification flags:")
            for n in flagged:
                print(f"  [{n.level}] {n.message}")

    print("\n--- Synthesized answer ---")
    print(answer.narrative)

    if args.csv:
        path = export_csv(grid, args.csv)
        print(f"\nGrid exported to {path}")
    return 0


async def _cmd_info(args) -> int:
    corpus = GridFin.from_path(args.data)
    info = corpus.info()
    print("GridFin — corpus info")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    print(f"  {'cache_dir':20s}: {get_settings().cache_dir}")
    return 0


async def _cmd_eval(args) -> int:
    corpus = GridFin.from_path(args.data)
    truth_path = Path(args.truth)
    truth = json.loads(truth_path.read_text())
    truth = {k: v for k, v in truth.items() if not k.startswith("_")}

    # Ask the broad question that exercises every ground-truth column.
    answer = await corpus.ask(
        "For each filing report revenue, net income, net profit margin, current "
        "ratio, debt to equity and return on equity."
    )
    report = evaluate_grid(answer.grid, truth)
    print("GridFin — evaluation\n")
    acc = report.cell_numeric_accuracy
    print(f"  Cell Numeric Accuracy  : {acc:.1%}  ({report.numeric_cells_scored} cells scored)")
    print(f"  Grid Completion        : {report.grid_completion:.1%}")
    print(f"  Attribution Correctness: {report.attribution_correctness:.1%}")
    if report.details:
        print("\n  Misses:")
        for d in report.details:
            print(f"    - {d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gridfin", description="Grid-native financial-document analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show corpus / backend info.")
    p_info.add_argument("--data", default=_DEFAULT_DATA)
    p_info.set_defaults(func=_cmd_info)

    p_ask = sub.add_parser("ask", help="Ask a question; render the grid + narrative.")
    p_ask.add_argument("question")
    p_ask.add_argument("--data", default=_DEFAULT_DATA)
    p_ask.add_argument("--doc-ids", nargs="*", dest="doc_ids")
    p_ask.add_argument("--json", action="store_true", help="Emit the full grid as JSON.")
    p_ask.add_argument("--csv", help="Export the grid to a CSV path.")
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="Run the grid evaluation metrics.")
    p_eval.add_argument("--data", default=_DEFAULT_DATA)
    p_eval.add_argument("--truth", default=f"{_DEFAULT_DATA}/ground_truth.json")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
