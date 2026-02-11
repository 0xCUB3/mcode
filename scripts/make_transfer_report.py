#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mcode.bench.results import ResultsDB


@dataclass(frozen=True)
class ConfigKey:
    samples: int
    max_debug_iterations: int
    timeout_s: int

    def label(self) -> str:
        return f"s={self.samples} d={self.max_debug_iterations} t={self.timeout_s}s"


@dataclass
class BenchMetric:
    benchmark: str
    pass_rate: float
    sec_per_solve: float
    timeout_rate: float
    total: int
    passed: int
    timed_out: int
    runs: int


@dataclass
class ConfigScore:
    config: ConfigKey
    max_regret: float
    mean_regret: float
    worst_benchmark: str
    mean_pass_rate: float
    mean_sec_per_solve: float
    mean_timeout_rate: float
    per_benchmark: dict[str, BenchMetric]


def _collect_db_paths(db_dirs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for db_dir in db_dirs:
        if not db_dir.exists() or not db_dir.is_dir():
            raise FileNotFoundError(f"--db-dir must be an existing directory: {db_dir}")
        out.extend(sorted(db_dir.rglob("*.db")))
    out = [path for path in out if "-shard-" in path.name]
    if not out:
        raise FileNotFoundError("No shard DB files found under the provided --db-dir values.")
    return out


def _normalize(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 1.0
    if hi - lo <= 1e-12:
        return 0.0
    return (value - lo) / (hi - lo)


def _load_metrics(
    *,
    db_paths: list[Path],
    benchmarks: list[str],
    model_id: str | None,
    backend_name: str | None,
) -> dict[ConfigKey, dict[str, BenchMetric]]:
    with tempfile.TemporaryDirectory(prefix="mcode-transfer-") as td:
        merged = Path(td) / "merged.db"
        rdb = ResultsDB(merged)
        try:
            rdb.merge_from(db_paths)
            out: dict[ConfigKey, dict[str, BenchMetric]] = {}
            for benchmark in benchmarks:
                rows = rdb.run_metrics_grouped(
                    benchmark=benchmark,
                    model_id=model_id,
                    backend_name=backend_name,
                    group_by=("backend_name", "max_debug_iterations", "timeout_s", "samples"),
                    retrieval=None,
                    samples=None,
                    include_percentiles=True,
                )
                for row in rows:
                    sec_per_solve = row.get("sec_per_solve")
                    if sec_per_solve is None:
                        continue
                    if not math.isfinite(float(sec_per_solve)):
                        continue
                    cfg = ConfigKey(
                        samples=int(row["samples"]),
                        max_debug_iterations=int(row["max_debug_iterations"]),
                        timeout_s=int(row["timeout_s"]),
                    )
                    metric = BenchMetric(
                        benchmark=benchmark,
                        pass_rate=float(row["pass_rate"]),
                        sec_per_solve=float(sec_per_solve),
                        timeout_rate=float(row["timeout_rate"]),
                        total=int(row["total"]),
                        passed=int(row["passed"]),
                        timed_out=int(row["timed_out"]),
                        runs=int(row.get("runs") or 0),
                    )
                    out.setdefault(cfg, {})[benchmark] = metric
            return out
        finally:
            rdb.close()


def _score_configs(
    *,
    metrics_by_config: dict[ConfigKey, dict[str, BenchMetric]],
    benchmarks: list[str],
    w_pass: float,
    w_speed: float,
    w_timeout: float,
) -> list[ConfigScore]:
    common_configs = [
        cfg
        for cfg, per_bench in metrics_by_config.items()
        if all(benchmark in per_bench for benchmark in benchmarks)
    ]
    if not common_configs:
        return []

    mins_maxs: dict[str, tuple[float, float, float, float, float, float]] = {}
    for benchmark in benchmarks:
        pass_vals = [metrics_by_config[cfg][benchmark].pass_rate for cfg in common_configs]
        speed_vals = [metrics_by_config[cfg][benchmark].sec_per_solve for cfg in common_configs]
        timeout_vals = [metrics_by_config[cfg][benchmark].timeout_rate for cfg in common_configs]
        mins_maxs[benchmark] = (
            min(pass_vals),
            max(pass_vals),
            min(speed_vals),
            max(speed_vals),
            min(timeout_vals),
            max(timeout_vals),
        )

    ranked: list[ConfigScore] = []
    for cfg in common_configs:
        per_benchmark = metrics_by_config[cfg]
        regrets: dict[str, float] = {}
        for benchmark in benchmarks:
            metric = per_benchmark[benchmark]
            (
                min_pass,
                max_pass,
                min_speed,
                max_speed,
                min_timeout,
                max_timeout,
            ) = mins_maxs[benchmark]

            pass_loss = _normalize(max_pass - metric.pass_rate, 0.0, max_pass - min_pass)
            speed_loss = _normalize(metric.sec_per_solve, min_speed, max_speed)
            timeout_loss = _normalize(metric.timeout_rate, min_timeout, max_timeout)
            regrets[benchmark] = (
                (w_pass * pass_loss)
                + (w_speed * speed_loss)
                + (w_timeout * timeout_loss)
            )

        worst_benchmark, max_regret = max(regrets.items(), key=lambda item: item[1])
        mean_regret = sum(regrets.values()) / len(regrets)
        mean_pass = sum(per_benchmark[b].pass_rate for b in benchmarks) / len(benchmarks)
        mean_speed = sum(per_benchmark[b].sec_per_solve for b in benchmarks) / len(benchmarks)
        mean_timeout = sum(per_benchmark[b].timeout_rate for b in benchmarks) / len(benchmarks)
        ranked.append(
            ConfigScore(
                config=cfg,
                max_regret=max_regret,
                mean_regret=mean_regret,
                worst_benchmark=worst_benchmark,
                mean_pass_rate=mean_pass,
                mean_sec_per_solve=mean_speed,
                mean_timeout_rate=mean_timeout,
                per_benchmark=per_benchmark,
            )
        )

    ranked.sort(key=lambda score: (score.max_regret, score.mean_regret, score.mean_sec_per_solve))
    return ranked


def _write_csv(
    *,
    scores: list[ConfigScore],
    benchmarks: list[str],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "samples",
        "max_debug_iterations",
        "timeout_s",
        "config",
        "max_regret",
        "mean_regret",
        "worst_benchmark",
        "mean_pass_rate",
        "mean_sec_per_solve",
        "mean_timeout_rate",
    ]
    for benchmark in benchmarks:
        fieldnames.extend(
            [
                f"{benchmark}_pass_rate",
                f"{benchmark}_sec_per_solve",
                f"{benchmark}_timeout_rate",
                f"{benchmark}_passed",
                f"{benchmark}_total",
                f"{benchmark}_timed_out",
                f"{benchmark}_runs",
            ]
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, score in enumerate(scores, start=1):
            row = {
                "rank": i,
                "samples": score.config.samples,
                "max_debug_iterations": score.config.max_debug_iterations,
                "timeout_s": score.config.timeout_s,
                "config": score.config.label(),
                "max_regret": f"{score.max_regret:.6f}",
                "mean_regret": f"{score.mean_regret:.6f}",
                "worst_benchmark": score.worst_benchmark,
                "mean_pass_rate": f"{score.mean_pass_rate:.6f}",
                "mean_sec_per_solve": f"{score.mean_sec_per_solve:.6f}",
                "mean_timeout_rate": f"{score.mean_timeout_rate:.6f}",
            }
            for benchmark in benchmarks:
                metric = score.per_benchmark[benchmark]
                row[f"{benchmark}_pass_rate"] = f"{metric.pass_rate:.6f}"
                row[f"{benchmark}_sec_per_solve"] = f"{metric.sec_per_solve:.6f}"
                row[f"{benchmark}_timeout_rate"] = f"{metric.timeout_rate:.6f}"
                row[f"{benchmark}_passed"] = metric.passed
                row[f"{benchmark}_total"] = metric.total
                row[f"{benchmark}_timed_out"] = metric.timed_out
                row[f"{benchmark}_runs"] = metric.runs
            writer.writerow(row)


def _write_markdown(
    *,
    scores: list[ConfigScore],
    benchmarks: list[str],
    out_md: Path,
    db_dirs: list[Path],
    w_pass: float,
    w_speed: float,
    w_timeout: float,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Transfer report")
    lines.append("")
    lines.append(
        "Ranking config robustness across benchmarks using minimax regret "
        "(lower is better)."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Benchmarks: {', '.join(benchmarks)}")
    lines.append(
        f"- Weights: pass={w_pass:.2f}, speed={w_speed:.2f}, timeout={w_timeout:.2f}"
    )
    for db_dir in db_dirs:
        lines.append(f"- DB dir: `{db_dir}`")
    lines.append("")

    if not scores:
        lines.append("No configs had complete coverage across all benchmarks.")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Robust ranking")
    lines.append("")
    lines.append(
        "| rank | config | max_regret | mean_regret | worst_benchmark | "
        "mean_pass | mean_sec/solve | mean_timeout |"
    )
    lines.append("|---:|---|---:|---:|---|---:|---:|---:|")
    for i, score in enumerate(scores, start=1):
        lines.append(
            f"| {i} | {score.config.label()} | {score.max_regret:.3f} | "
            f"{score.mean_regret:.3f} | {score.worst_benchmark} | "
            f"{score.mean_pass_rate:.1%} | {score.mean_sec_per_solve:.2f} | "
            f"{score.mean_timeout_rate:.2%} |"
        )
    lines.append("")

    top = scores[0]
    fastest = min(scores, key=lambda score: score.mean_sec_per_solve)
    most_accurate = max(scores, key=lambda score: score.mean_pass_rate)

    lines.append("## Findings")
    lines.append("")
    lines.append(
        f"- Most robust config is `{top.config.label()}` "
        f"(lowest max regret: {top.max_regret:.3f})."
    )
    lines.append(
        f"- Fastest config by mean sec/solve is `{fastest.config.label()}` "
        f"({fastest.mean_sec_per_solve:.2f})."
    )
    lines.append(
        f"- Most accurate config by mean pass rate is `{most_accurate.config.label()}` "
        f"({most_accurate.mean_pass_rate:.1%})."
    )
    lines.append(
        "- Use robust config as default mode; keep fastest and most accurate as explicit modes."
    )
    lines.append("")

    lines.append("## Per-benchmark detail")
    lines.append("")
    for benchmark in benchmarks:
        lines.append(f"### {benchmark}")
        lines.append("")
        lines.append("| config | pass_rate | sec/solve | timeout_rate | passed/total | runs |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for score in scores:
            metric = score.per_benchmark[benchmark]
            lines.append(
                f"| {score.config.label()} | {metric.pass_rate:.1%} | "
                f"{metric.sec_per_solve:.2f} | {metric.timeout_rate:.2%} | "
                f"{metric.passed}/{metric.total} | {metric.runs} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a cross-benchmark transfer report for benchmark configs."
    )
    parser.add_argument(
        "--db-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing shard DBs (repeatable).",
    )
    parser.add_argument(
        "--benchmarks",
        default="mbpp,humaneval",
        help="Comma-separated benchmark names to include.",
    )
    parser.add_argument("--model", default=None, help="Optional model filter.")
    parser.add_argument("--backend", default=None, help="Optional backend filter.")
    parser.add_argument(
        "--w-pass",
        type=float,
        default=0.65,
        help="Weight for pass-rate regret (default: 0.65).",
    )
    parser.add_argument(
        "--w-speed",
        type=float,
        default=0.30,
        help="Weight for speed regret (default: 0.30).",
    )
    parser.add_argument(
        "--w-timeout",
        type=float,
        default=0.05,
        help="Weight for timeout regret (default: 0.05).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        required=True,
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    weight_sum = args.w_pass + args.w_speed + args.w_timeout
    if abs(weight_sum - 1.0) > 1e-9:
        raise SystemExit("Weights must sum to 1.0.")

    benchmarks = [part.strip() for part in args.benchmarks.split(",") if part.strip()]
    if not benchmarks:
        raise SystemExit("No benchmarks specified.")

    db_paths = _collect_db_paths(args.db_dir)
    metrics = _load_metrics(
        db_paths=db_paths,
        benchmarks=benchmarks,
        model_id=args.model,
        backend_name=args.backend,
    )
    scores = _score_configs(
        metrics_by_config=metrics,
        benchmarks=benchmarks,
        w_pass=args.w_pass,
        w_speed=args.w_speed,
        w_timeout=args.w_timeout,
    )

    _write_csv(scores=scores, benchmarks=benchmarks, out_csv=args.out_csv)
    _write_markdown(
        scores=scores,
        benchmarks=benchmarks,
        out_md=args.out_md,
        db_dirs=args.db_dir,
        w_pass=args.w_pass,
        w_speed=args.w_speed,
        w_timeout=args.w_timeout,
    )
    print(f"Wrote transfer report: {args.out_md}")
    print(f"Wrote transfer CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
