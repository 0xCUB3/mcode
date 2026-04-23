from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceContextEntry:
    path: str
    kind: str
    summary: str
    priority: int


@dataclass(frozen=True)
class WorkspaceContext:
    entries: tuple[WorkspaceContextEntry, ...] = ()

    @property
    def text(self) -> str:
        if not self.entries:
            return ""
        blocks = [
            "Local workspace context:",
            "These local files look authoritative for this task. Treat them as context, "
            "and let the tests remain the final verifier.",
        ]
        for entry in self.entries:
            blocks.append(f"\n- {entry.path} ({entry.kind})\n{entry.summary}")
        return "\n".join(blocks)


_DOC_FILENAMES = {
    "AGENTS.md": (10, "agent instructions"),
    "CLAUDE.md": (20, "agent instructions"),
    "instructions.md": (30, "task instructions"),
    "instructions.append.md": (31, "task instructions"),
    "SPEC.md": (40, "specification"),
    "SPECIFICATION.md": (40, "specification"),
    "README.md": (60, "readme"),
    "CONTRIBUTING.md": (80, "contributing guide"),
}
_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    ".gradle",
}
_MAX_SCAN_DEPTH = 3
_MAX_DOC_BYTES = 64_000
_MAX_LINE_CHARS = 220


def collect_workspace_context(
    repo_root: str,
    query: str,
    *,
    max_chars: int = 3_000,
    max_entries: int = 4,
) -> WorkspaceContext:
    root = Path(repo_root)
    if max_chars <= 0 or max_entries <= 0 or not root.exists():
        return WorkspaceContext()

    tokens = _query_tokens(query)
    ranked = sorted(
        _find_doc_candidates(root),
        key=lambda item: (item[0], item[1].relative_to(root).as_posix()),
    )
    entries: list[WorkspaceContextEntry] = []
    remaining = max_chars
    for priority, path, kind in ranked:
        if len(entries) >= max_entries or remaining < 200:
            break
        rel = path.relative_to(root).as_posix()
        summary = _summarize_doc(path, tokens=tokens, max_chars=min(remaining, 1_200))
        if not summary:
            continue
        entries.append(
            WorkspaceContextEntry(
                path=rel,
                kind=kind,
                summary=summary,
                priority=priority,
            )
        )
        remaining -= len(summary) + len(rel) + 32
    return WorkspaceContext(tuple(entries))


def _find_doc_candidates(root: Path) -> list[tuple[int, Path, str]]:
    candidates: dict[Path, tuple[int, str]] = {}
    for path in _walk_docs(root):
        if not path.is_file():
            continue
        priority_kind = _classify_doc(path)
        if priority_kind is None:
            continue
        priority, kind = priority_kind
        current = candidates.get(path)
        if current is None or priority < current[0]:
            candidates[path] = (priority, kind)
    return [(priority, path, kind) for path, (priority, kind) in candidates.items()]


def _walk_docs(root: Path) -> list[Path]:
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                if child.name in _IGNORED_DIRS or depth >= _MAX_SCAN_DEPTH:
                    continue
                stack.append((child, depth + 1))
                continue
            if child.suffix.lower() == ".md":
                out.append(child)
    return out


def _classify_doc(path: Path) -> tuple[int, str] | None:
    name = path.name
    if name in _DOC_FILENAMES:
        priority, kind = _DOC_FILENAMES[name]
        if path.parent.name == ".docs" and name.startswith("instructions"):
            return priority, kind
        return priority + _depth_penalty(path), kind
    if path.parent.name in {"docs", ".docs"} and path.suffix.lower() == ".md":
        return 90 + _depth_penalty(path), "documentation"
    return None


def _depth_penalty(path: Path) -> int:
    return min(20, max(0, len(path.parts) - 1))


def _summarize_doc(path: Path, *, tokens: set[str], max_chars: int) -> str:
    try:
        if path.stat().st_size > _MAX_DOC_BYTES:
            raw = path.read_text(errors="replace")[:_MAX_DOC_BYTES]
        else:
            raw = path.read_text(errors="replace")
    except OSError:
        return ""
    lines = _clean_lines(raw)
    if not lines:
        return ""

    selected = _select_relevant_lines(lines, tokens=tokens)
    if not selected:
        selected = lines
    return _cap_lines(selected, max_chars=max_chars)


def _clean_lines(raw: str) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        if len(stripped) > _MAX_LINE_CHARS:
            stripped = stripped[: _MAX_LINE_CHARS - 3].rstrip() + "..."
        lines.append(stripped)
    return lines


def _select_relevant_lines(lines: list[str], *, tokens: set[str]) -> list[str]:
    selected: list[str] = []
    for line in lines[:12]:
        selected.append(line)
    if tokens:
        for line in lines[12:]:
            lowered = line.lower()
            if any(token in lowered for token in tokens):
                selected.append(line)
            if len(selected) >= 24:
                break
    return selected


def _cap_lines(lines: list[str], *, max_chars: int) -> str:
    out: list[str] = []
    used = 0
    for line in lines:
        line_cost = len(line) + 1
        if used + line_cost > max_chars:
            break
        out.append(line)
        used += line_cost
    return "\n".join(out).strip()


def _query_tokens(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query.lower())
        if token not in {"the", "and", "for", "with", "this", "that", "from", "into"}
    }
