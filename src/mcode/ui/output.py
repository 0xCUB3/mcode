"""Small reusable output helpers.

Used by status, doctor, results, and similar list-or-key/value commands so
their formatting is consistent.
"""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table

from mcode.ui.styles import Symbol


def status_line(symbol: Symbol | str, text: str) -> str:
    """Render `<symbol> <text>` for plain-text command output."""
    s = symbol.value if isinstance(symbol, Symbol) else symbol
    return f"{s} {text}"


def kv_table(rows: Iterable[tuple[str, str]], *, title: str | None = None) -> Table:
    """A two-column key-value Rich Table."""
    table = Table(title=title, show_header=False, pad_edge=False, box=None)
    table.add_column("key", style="dim")
    table.add_column("value")
    for k, v in rows:
        table.add_row(k, v)
    return table


def section_header(text: str) -> str:
    """A visually distinct one-liner for grouping sections in plain output."""
    return f"-- {text} --"


__all__ = ["kv_table", "section_header", "status_line"]
