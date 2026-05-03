from __future__ import annotations

import json
from contextlib import contextmanager
from glob import glob
from pathlib import Path

import typer

from mcode.bench.results import ResultsDB
from mcode.util import temporary_directory

DEFAULT_DB_PATH = Path("experiments/results/results.db")
DEFAULT_ARTIFACT_DIR_NAME = "artifacts"


def default_artifact_dir(db: Path) -> Path:
    return db.parent / db.stem / DEFAULT_ARTIFACT_DIR_NAME


def resolve_artifact_dir(db: Path, artifact_dir: Path | None) -> Path:
    if artifact_dir is not None:
        return artifact_dir
    return default_artifact_dir(db)


def optional_str(v: str) -> str | None:
    if v.strip().lower() in {"", "none", "null"}:
        return None
    return v


def validate_shards(
    *, shard_count: int | None, shard_index: int | None
) -> tuple[int | None, int | None]:
    if shard_index is not None and shard_count is None:
        raise typer.BadParameter("--shard-index requires --shard-count")
    if shard_count is not None and shard_index is not None and shard_index >= shard_count:
        raise typer.BadParameter("--shard-index must be < --shard-count")
    return shard_count, shard_index


def validate_shard_options(
    *,
    shards: int | None,
    shard_count: int | None,
    shard_index: int | None,
) -> tuple[int | None, int | None, int | None]:
    if shards is not None and (shard_count is not None or shard_index is not None):
        raise typer.BadParameter("--shards cannot be combined with --shard-count/--shard-index")
    shard_count, shard_index = validate_shards(shard_count=shard_count, shard_index=shard_index)
    return shards, shard_count, shard_index


def append_option(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def parse_task_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        p = Path(raw)
        exists = p.exists()
    except OSError:
        exists = False
    if exists:
        text = p.read_text()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "tasks" in data:
            ids: list[str] = []
            for v in data["tasks"].values():
                if isinstance(v, list):
                    ids.extend(v)
            return ids
        raise typer.BadParameter(f"Cannot parse task IDs from {raw}")
    return [t.strip() for t in raw.split(",") if t.strip()]


def validate_sampling(
    *,
    sampling: str,
    sampling_budget: int | None,
) -> tuple[str, int | None]:
    if sampling == "none" and sampling_budget is not None:
        raise typer.BadParameter("--sampling-budget requires --sampling != none")
    return sampling, sampling_budget


@contextmanager
def open_results_view(db_paths: tuple[Path, ...] | list[Path]):
    if not db_paths:
        db_paths = [DEFAULT_DB_PATH]

    resolved: list[Path] = []
    for p in db_paths:
        if not p.exists():
            raise typer.BadParameter(f"SQLite DB not found: {p}")
        resolved.append(p.resolve())

    if len(resolved) == 1:
        rdb = ResultsDB(resolved[0])
        try:
            yield rdb
        finally:
            rdb.close()
        return

    with temporary_directory(prefix="mcode-results-") as td:
        merged_path = Path(td) / "merged.db"
        rdb = ResultsDB(merged_path)
        try:
            rdb.merge_from(resolved)
            yield rdb
        finally:
            rdb.close()


def expand_db_paths(
    *,
    db: list[Path] | None,
    db_glob: list[str] | None,
    db_dir: list[Path] | None,
) -> list[Path]:
    paths: list[Path] = []

    for p in db or []:
        paths.append(p)

    for d in db_dir or []:
        if not d.exists() or not d.is_dir():
            raise typer.BadParameter(f"--db-dir must be a directory: {d}")
        paths.extend(sorted(d.rglob("*.db")))

    for pattern in db_glob or []:
        matches = glob(pattern, recursive=True)
        if not matches:
            raise typer.BadParameter(f"--db-glob matched no files: {pattern}")
        paths.extend([Path(m) for m in matches])

    if not paths:
        paths = [DEFAULT_DB_PATH]

    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


_default_artifact_dir = default_artifact_dir
_resolve_artifact_dir = resolve_artifact_dir
_optional_str = optional_str
_validate_shards = validate_shards
_validate_shard_options = validate_shard_options
_append_option = append_option
_parse_task_ids = parse_task_ids
_validate_sampling = validate_sampling
_open_results_view = open_results_view
_expand_db_paths = expand_db_paths
