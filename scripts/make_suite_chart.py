#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRow:
    benchmark: str
    backend: str
    model: str
    loop_budget: int
    timeout_s: int
    total: int
    passed: int
    pass_rate: float
    config: dict


def _load_runs_csv(path: Path) -> list[RunRow]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows: list[RunRow] = []
        for row in r:
            config_json = (row.get("config_json") or "").strip()
            config: dict = {}
            if config_json:
                try:
                    config = json.loads(config_json)
                except Exception:
                    config = {"config_json": config_json}
            rows.append(
                RunRow(
                    benchmark=str(row["benchmark"]),
                    backend=str(row["backend_name"]),
                    model=str(row["model_id"]),
                    loop_budget=int(row["loop_budget"]),
                    timeout_s=int(row["timeout_s"]),
                    total=int(row["total"]),
                    passed=int(row["passed"]),
                    pass_rate=float(row["pass_rate"]),
                    config=config,
                )
            )
    return rows


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _bar_chart_svg(
    *,
    title: str,
    rows: list[RunRow],
    y_label: str,
    label_fn,
    width: int,
    x0: int,
    y0: int,
) -> tuple[list[str], int]:
    """
    Returns SVG lines and height used.
    """

    if not rows:
        lines = [
            f'<text x="{x0}" y="{y0}" font-size="18" font-weight="700">{_esc(title)}</text>',
            f'<text x="{x0}" y="{y0+26}" font-size="14" fill="#666">No data.</text>',
        ]
        return lines, 52

    # Layout
    title_h = 28
    row_h = 34
    label_w = 360
    bar_h = 16
    chart_w = width - x0 - 40
    bar_w = chart_w - label_w - 140  # reserve room for right-side text
    if bar_w < 220:
        bar_w = 220

    # Colors
    bar_fill = "#3b82f6"
    bar_bg = "#e5e7eb"
    grid = "#d1d5db"
    text = "#111827"
    subtext = "#374151"

    lines: list[str] = []
    lines.append(
        f'<text x="{x0}" y="{y0}" font-size="18" font-weight="700" fill="{text}">'
        f"{_esc(title)}</text>"
    )

    # Axis/grid
    axis_y = y0 + 10
    axis_x = x0 + label_w
    axis_top = axis_y + 18
    axis_bottom = axis_top + row_h * len(rows) + 10
    for pct in (0, 20, 40, 60, 80, 100):
        gx = axis_x + int(bar_w * (pct / 100.0))
        lines.append(
            f'<line x1="{gx}" y1="{axis_top}" x2="{gx}" y2="{axis_bottom}" '
            f'stroke="{grid}" stroke-width="1" />'
        )
        lines.append(
            f'<text x="{gx}" y="{axis_top-6}" font-size="12" fill="{subtext}" '
            f'text-anchor="middle">{pct}%</text>'
        )

    # Y label
    lines.append(
        f'<text x="{x0}" y="{axis_top-6}" font-size="12" fill="{subtext}">{_esc(y_label)}</text>'
    )

    # Rows
    y = axis_top + 18
    for rr in rows:
        label = str(label_fn(rr))
        bar_len = max(0, min(bar_w, int(bar_w * (rr.pass_rate))))
        # left label
        lines.append(
            f'<text x="{x0}" y="{y+bar_h}" font-size="13" fill="{text}">{_esc(label)}</text>'
        )
        # background bar
        lines.append(
            f'<rect x="{axis_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{bar_bg}" rx="4" />'
        )
        # filled bar
        lines.append(
            f'<rect x="{axis_x}" y="{y}" width="{bar_len}" height="{bar_h}" '
            f'fill="{bar_fill}" rx="4" />'
        )
        # right text: passed/total + pct
        pct_txt = f"{rr.pass_rate*100:.1f}%"
        right = f"{rr.passed}/{rr.total}  ({pct_txt})"
        lines.append(
            f'<text x="{axis_x + bar_w + 12}" y="{y+bar_h}" font-size="13" fill="{text}">'
            f"{_esc(right)}</text>"
        )
        y += row_h

    used_h = title_h + (axis_bottom - axis_y) + 10
    return lines, used_h


def write_suite_chart(*, suite_dir: Path) -> Path:
    runs_csv = suite_dir / "suite.runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(
            f"Missing {runs_csv}. Run: uv run mcode export-csv -i {suite_dir} "
            f"--out-dir {suite_dir} --prefix suite"
        )

    rows = _load_runs_csv(runs_csv)

    def cfg_label(rr: RunRow) -> str:
        if rr.benchmark == "swebench-lite":
            mode = rr.config.get("swebench_mode") or rr.config.get("mode") or "unknown"
            return f"{mode}  (timeout={rr.timeout_s}s)"
        return f"budget={rr.loop_budget}  t={rr.timeout_s}s"

    humaneval = [r for r in rows if r.benchmark == "humaneval"]
    mbpp = [r for r in rows if r.benchmark == "mbpp"]
    swe = [r for r in rows if r.benchmark == "swebench-lite"]

    # Sort for readability
    humaneval.sort(key=lambda r: (r.loop_budget, r.timeout_s))
    mbpp.sort(key=lambda r: (r.loop_budget, r.timeout_s))
    swe.sort(key=lambda r: (str(r.config.get("swebench_mode") or ""), r.timeout_s))

    # Global header text
    model = rows[0].model if rows else ""
    backend = rows[0].backend if rows else ""

    width = 1200
    x0 = 40
    y = 48

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="10" '
        f'viewBox="0 0 {width} 10">'
    )
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />')
    title_text = "Suite pass rates"
    if model and backend:
        title_text = f"Suite pass rates — {model} ({backend})"
    elif model:
        title_text = f"Suite pass rates — {model}"
    elif backend:
        title_text = f"Suite pass rates — {backend}"
    svg.append(
        f'<text x="{x0}" y="28" font-size="22" font-weight="800" fill="#111827">'
        f"{_esc(title_text)}"
        f"</text>"
    )
    if model or backend:
        subtitle = " · ".join([s for s in [f"backend={backend}", f"model={model}"] if s])
        svg.append(f'<text x="{x0}" y="46" font-size="13" fill="#374151">{_esc(subtitle)}</text>')

    # Legend (keep short + readable in screenshots)
    svg.append(
        f'<text x="{x0}" y="66" font-size="12" fill="#374151">'
        f"{_esc('Legend: budget=loop budget (mellea retries), t=timeout seconds')}"
        f"</text>"
    )
    svg.append(
        f'<text x="{x0}" y="82" font-size="12" fill="#374151">'
        f"{_esc('SWE-bench Lite: gold=dataset patch (sanity check), model=model-generated patch')}"
        f"</text>"
    )

    sections: list[tuple[str, list[RunRow], str]] = [
        ("HumanEval", humaneval, "Config (budget/timeout)"),
        ("MBPP", mbpp, "Config (budget/timeout)"),
        ("SWE-bench Lite (subset)", swe, "Mode"),
    ]

    y = 108
    content: list[str] = []
    for title, data, y_label in sections:
        chunk, used_h = _bar_chart_svg(
            title=title,
            rows=data,
            y_label=y_label,
            label_fn=cfg_label,
            width=width,
            x0=x0,
            y0=y,
        )
        content.extend(chunk)
        y += used_h + 24

    height = y + 10
    svg[0] = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    svg.extend(content)
    svg.append("</svg>")

    out_path = suite_dir / "suite.summary.svg"
    out_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return out_path


def render_png(*, svg_path: Path, png_path: Path) -> None:
    svg = str(svg_path)
    png = str(png_path)

    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", svg, "-o", png], check=True)
        return
    if shutil.which("inkscape"):
        subprocess.run(
            ["inkscape", svg, "--export-type=png", f"--export-filename={png}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    if shutil.which("magick"):
        subprocess.run(["magick", "-background", "white", svg, png], check=True)
        return
    if shutil.which("convert"):
        subprocess.run(["convert", "-background", "white", svg, png], check=True)
        return

    raise RuntimeError(
        "No SVG renderer found. Install one of: librsvg (rsvg-convert), Inkscape, or ImageMagick."
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate a simple SVG summary chart from suite CSVs.")
    p.add_argument("suite_dir", type=Path, help="Suite directory (contains suite.runs.csv)")
    args = p.parse_args()

    svg_path = write_suite_chart(suite_dir=args.suite_dir)
    png_path = svg_path.with_suffix(".png")
    render_png(svg_path=svg_path, png_path=png_path)
    print(svg_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
