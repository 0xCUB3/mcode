from __future__ import annotations

import importlib.resources as ir
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteEntry:
    name: str
    benchmark: str
    limit: int | None = None
    task_ids: tuple[str, ...] = ()
    split: str | None = None
    dataset: str | None = None
    language: str | None = None
    benchmark_root: str | None = None
    no_retry: bool = False


@dataclass(frozen=True)
class SuiteManifest:
    entries: tuple[SuiteEntry, ...]


_DEFAULT_SUITE_FIXTURE = "default-suite.json"


def default_suite_path() -> Path:
    resource = ir.files("mcode.bench.fixtures").joinpath(_DEFAULT_SUITE_FIXTURE)
    with ir.as_file(resource) as path:
        return path


def load_suite_manifest(path: Path | None = None) -> SuiteManifest:
    manifest_path = path if path is not None else default_suite_path()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError(f"suite manifest has no entries: {manifest_path}")
    entries = tuple(_entry_from_raw(item) for item in entries_raw)
    return SuiteManifest(entries=entries)


def task_ids_arg(entry: SuiteEntry) -> str | None:
    if not entry.task_ids:
        return None
    return ",".join(entry.task_ids)


def _entry_from_raw(raw: object) -> SuiteEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"suite entry must be an object, got {type(raw).__name__}")
    name = str(raw.get("name") or "").strip()
    benchmark = str(raw.get("benchmark") or "").strip()
    if not name or not benchmark:
        raise ValueError(f"suite entry is missing required fields: {raw}")
    task_ids_value = raw.get("task_ids") or []
    if not isinstance(task_ids_value, list):
        raise ValueError(f"suite entry task_ids must be a list: {raw}")
    return SuiteEntry(
        name=name,
        benchmark=benchmark,
        limit=_optional_int(raw.get("limit")),
        task_ids=tuple(str(item) for item in task_ids_value),
        split=_optional_str(raw.get("split")),
        dataset=_optional_str(raw.get("dataset")),
        language=_optional_str(raw.get("language")),
        benchmark_root=_optional_str(raw.get("benchmark_root")),
        no_retry=bool(raw.get("no_retry", False)),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
