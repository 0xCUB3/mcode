from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field


class FileLocalization(BaseModel):
    files: list[str] = Field(
        ...,
        description="File paths relative to repo root, most likely to need editing",
    )


_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".tox",
        ".nox",
        ".eggs",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "doc",
        "docs",
        "examples",
        "example",
        "benchmarks",
        "tests",
        "test",
        "testing",
    }
)


def collect_source_files(repo_root: str) -> list[str]:
    root = Path(repo_root)
    paths: list[str] = []
    for p in sorted(root.rglob("*.py")):
        if _EXCLUDED_DIRS.intersection(p.parts):
            continue
        paths.append(str(p.relative_to(root)))
    return paths


def build_indented_tree(paths: list[str]) -> str:
    tree: dict = {}
    for p in paths:
        parts = p.split("/")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: list[str] = []

    def _walk(node: dict, indent: int) -> None:
        for name in sorted(node, key=lambda n: (not bool(node[n]), n)):
            if node[name]:
                lines.append(" " * indent + name + "/")
                _walk(node[name], indent + 4)
            else:
                lines.append(" " * indent + name)

    _walk(tree, 0)
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z_]\w{2,}", text.lower())
    tokens = []
    for tok in raw:
        tokens.append(tok)
        # Split snake_case so "name_checker" also yields "name", "checker"
        parts = tok.split("_")
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) >= 3)
    return tokens


def rank_bm25(
    paths: list[str],
    query: str,
    repo_root: str,
    *,
    top_n: int = 30,
) -> list[str]:
    root = Path(repo_root)
    query_tokens = _tokenize(query)
    if not query_tokens or not paths:
        return paths[:top_n]

    docs: list[list[str]] = []
    valid_paths: list[str] = []
    for p in paths:
        try:
            text = (root / p).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        tokens = _tokenize(text)
        # Also tokenize the file path itself (directory and file names are signal)
        tokens.extend(_tokenize(p.replace("/", " ").replace(".py", "")))
        docs.append(tokens)
        valid_paths.append(p)

    if not docs:
        return paths[:top_n]

    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    k1, b = 1.5, 0.75

    # Document frequency
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))

    scores = []
    for i, doc in enumerate(docs):
        tf: Counter[str] = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_tokens:
            if term not in df:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            term_tf = tf[term]
            score += idf * (term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * dl / avgdl))
        scores.append((score, valid_paths[i]))

    scores.sort(key=lambda x: -x[0])
    return [p for _, p in scores[:top_n]]


def localize(
    session,
    repo_root: str,
    problem_statement: str,
    *,
    bm25_top_n: int = 30,
    max_context_chars: int = 12000,
) -> tuple[list[str], str]:
    """Localize files and build enriched hints.

    Returns (file_paths, hints_text) where hints_text contains the file contents
    ready to pass to generate_patch.
    """
    all_files = collect_source_files(repo_root)
    if not all_files:
        return [], ""

    # BM25 pre-filter
    top_files = rank_bm25(all_files, problem_statement, repo_root, top_n=bm25_top_n)
    print(f"bm25 top-{bm25_top_n}: {top_files[:10]}...", flush=True)

    # Build indented tree of top candidates for LLM
    tree = build_indented_tree(top_files)

    # LLM picks files from tree
    # Also provide a flat list so the model can copy exact paths
    system_prompt = (
        "You are an expert software engineer.\n"
        "Given a repository file tree and an issue description, "
        "identify which files most likely need to be modified to fix the issue.\n"
        "Return only file paths from the CANDIDATES list below.\n"
        "Return 1-5 files, ordered by likelihood (most likely first).\n"
        "IMPORTANT: Copy paths exactly from the candidates list. "
        "Do not invent or modify paths."
    )
    candidates = "\n".join(top_files)
    description = (
        f"Repository structure:\n{tree}\n\n"
        f"CANDIDATES (copy exact paths from this list):\n{candidates}\n\n"
        f"Issue:\n{problem_statement.strip()}"
    )
    result = session._m.instruct(
        description,
        format=FileLocalization,
        strategy=None,
        model_options=session._model_options(system_prompt=system_prompt),
    )

    import json

    try:
        data = json.loads(result.value)
        localized = data.get("files", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        localized = []

    # Filter to candidate files only (LLM often picks similar-sounding non-candidates)
    root = Path(repo_root)
    candidate_set = set(top_files)
    localized = [f for f in localized if f in candidate_set]

    if not localized:
        # Fallback: use top BM25 files
        localized = top_files[:5]

    # Read localized files and build hints
    parts = []
    chars = 0
    for rel in localized:
        fp = root / rel
        if not fp.is_file():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        budget = max_context_chars - chars
        if budget <= 0:
            break
        if len(content) > budget:
            content = content[:budget] + "\n... (truncated)"
        parts.append(f"--- {rel} ---\n{content}")
        chars += len(content) + len(rel) + 10

    hints = ""
    if parts:
        hints = "Relevant source files from the repository:\n" + "\n".join(parts)

    return localized, hints
