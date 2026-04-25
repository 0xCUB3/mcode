from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

_MAX_OUTPUT = 10_000
_DEFAULT_TIMEOUT = 30
_BLOCKED_PREFIXES = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){",
    "chmod -R 777 /",
    "sudo",
    "reboot",
    "shutdown",
    "kill -9 -1",
    "pkill",
)
_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".tox",
    ".eggs",
    ".mypy_cache",
}
_MAX_MATCHES = 30
_MAX_LINES = 200
_MAX_RESULTS = 50
_MAX_SCAN = 10_000
_TIMEOUT_S = 5
_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}


def _truncate_output(output: str) -> str:
    if len(output) <= _MAX_OUTPUT:
        return output

    half = _MAX_OUTPUT // 2
    return (
        output[:half]
        + f"\n\n[... truncated {len(output) - _MAX_OUTPUT} chars ...]\n\n"
        + output[-half:]
    )


def format_tool_result(command: str, status: str, output: str) -> str:
    body = output.rstrip() if output.strip() else "(no output)"
    return f"$ {command}\n{status}\n{body}"


def is_tool_result(output: str) -> bool:
    lines = output.splitlines()
    return len(lines) >= 2 and lines[0].startswith("$ ") and bool(lines[1].strip())


def _run_shell_command(
    command: str,
    *,
    repo_root: str,
    timeout: int | None = None,
    allowed_dirs: list[str] | None = None,
) -> tuple[str, str]:
    timeout = timeout or int(os.environ.get("MCODE_BASH_TIMEOUT", str(_DEFAULT_TIMEOUT)))

    cmd_lower = command.strip().lower()
    for blocked in _BLOCKED_PREFIXES:
        if cmd_lower.startswith(blocked):
            return "BLOCKED", f"Error: command blocked for safety: {command[:80]}"

    if allowed_dirs is not None:
        import shlex

        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            if token.startswith("/") and not any(token.startswith(d) for d in allowed_dirs):
                return (
                    "ERROR",
                    f"Error: absolute path '{token}' is outside allowed directories. "
                    "Use relative paths from the repo root.",
                )

    root = Path(repo_root).resolve()
    if not root.is_dir():
        return "ERROR", f"Error: repo root does not exist: {repo_root}"

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HOME": str(root), "LC_ALL": "C"},
        )
        output = _truncate_output(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s", ""
    except Exception as e:
        return "ERROR", f"Error: {type(e).__name__}: {e}"

    status = "PASSED" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return status, output


def execute_command(
    command: str,
    *,
    repo_root: str,
    timeout: int | None = None,
    allowed_dirs: list[str] | None = None,
) -> tuple[str, str]:
    return _run_shell_command(
        command,
        repo_root=repo_root,
        timeout=timeout,
        allowed_dirs=allowed_dirs,
    )


def search_code(query: str, *, repo_root: str) -> str:
    globs = [f"--glob=!{d}" for d in _SKIP_DIRS]
    cmd = [
        "rg",
        "--no-heading",
        "--line-number",
        "--max-columns=200",
        "--max-count=5",
        "--color=never",
        *globs,
        query,
        repo_root,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        output = result.stdout.strip()
    except FileNotFoundError:
        return _fallback_search(query, repo_root)
    except subprocess.TimeoutExpired:
        return f"Search timed out after {_TIMEOUT_S}s. Try a more specific query."

    if not output:
        return "No matches found."

    lines = output.splitlines()
    if len(lines) > _MAX_MATCHES:
        lines = lines[:_MAX_MATCHES]
        return "\n".join(lines) + f"\n... ({_MAX_MATCHES} of many matches shown)"
    return "\n".join(lines)


def _fallback_search(query: str, repo_root: str) -> str:
    pattern = re.compile(query, re.IGNORECASE)
    matches: list[str] = []
    for path in Path(repo_root).rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.stat().st_size > 500_000:
            continue
        try:
            text = path.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                rel = path.relative_to(repo_root)
                matches.append(f"{rel}:{i}: {line[:200]}")
                if len(matches) >= _MAX_MATCHES:
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches found."


def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    *,
    repo_root: str,
) -> str:
    full_path = Path(repo_root) / path if not Path(path).is_absolute() else Path(path)
    try:
        content = full_path.read_text(errors="replace")
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"

    lines = content.splitlines()
    total = len(lines)
    start = max(1, start_line) - 1
    end = min(total, end_line or total)
    selected = lines[start:end]
    truncated = False
    if len(selected) > _MAX_LINES:
        selected = selected[:_MAX_LINES]
        end = start + _MAX_LINES
        truncated = True

    numbered = [f"{start + i + 1:>4}: {line}" for i, line in enumerate(selected)]
    header = f"{path} ({total} lines total, showing {start + 1}-{end})"
    if truncated:
        header += f" [truncated to {_MAX_LINES} lines]"
    return header + "\n" + "\n".join(numbered)


def _count_errors(tree) -> int:
    count = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            count += 1
        stack.extend(node.children)
    return count

# Tree-sitter is too noisy for C-family headers and macro-heavy sources.
# The compiler-backed benchmark verifier is the reliable syntax boundary there.
_SYNTAX_GUARD_UNRELIABLE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}


def _syntax_details(path: str, content: str) -> tuple[int, str | None] | None:
    ext = Path(path).suffix.lower()
    lang = _EXTENSION_TO_LANGUAGE.get(ext)
    if lang is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(cast(Any, lang))
    except (ImportError, Exception):
        return None
    tree = parser.parse(content.encode("utf-8"))
    error_count = _count_errors(tree)
    if error_count > 0:
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "ERROR":
                row, col = node.start_point
                return error_count, f"Syntax error at line {row + 1}, column {col}"
            stack.extend(node.children)
    return 0, None


def _fuzzy_find(content: str, old_str: str) -> str | None:
    def normalize(value: str) -> str:
        return re.sub(r"[ \t]+", " ", value).strip()

    norm_old = normalize(old_str)
    old_lines = old_str.splitlines()
    content_lines = content.splitlines()
    n = len(old_lines)
    if n == 0:
        return None

    for i in range(len(content_lines) - n + 1):
        candidate = "\n".join(content_lines[i : i + n])
        if normalize(candidate) == norm_old:
            return candidate
    return None


def _format_edit_result(status: str, path: str, detail: str) -> str:
    return format_tool_result(f"edit {path}", status, detail)


def _should_skip_syntax_guard(path: str) -> bool:
    return Path(path).suffix.lower() in _SYNTAX_GUARD_UNRELIABLE_SUFFIXES


def str_replace_edit(path: str, old_str: str, new_str: str, *, repo_root: str) -> str:
    try:
        full_path = Path(repo_root) / path if not Path(path).is_absolute() else Path(path)
        content = full_path.read_text(errors="replace")
    except FileNotFoundError:
        return _format_edit_result("REJECTED", path, "Error: file not found.")
    except OSError as e:
        return _format_edit_result("REJECTED", path, f"Error reading file: {e}")

    count = content.count(old_str)
    if count == 0 and os.environ.get("MCODE_FUZZY_EDIT", "1") == "1":
        match = _fuzzy_find(content, old_str)
        if match is not None:
            old_str = match
            count = 1
    if count == 0:
        preview = "\n".join(content.splitlines()[:30])
        return _format_edit_result(
            "REJECTED",
            path,
            f"Error: old_str not found. First 30 lines:\n{preview}",
        )
    if count > 1:
        return _format_edit_result(
            "REJECTED",
            path,
            f"Error: old_str appears {count} times. Provide more context to make the match unique.",
        )

    new_content = content.replace(old_str, new_str, 1)
    if not _should_skip_syntax_guard(path):
        pre_syntax = _syntax_details(str(full_path), content)
        post_syntax = _syntax_details(str(full_path), new_content)
        pre_error_count = 0 if pre_syntax is None else pre_syntax[0]
        post_error_count = 0 if post_syntax is None else post_syntax[0]
        post_error = None if post_syntax is None else post_syntax[1]
        if post_error_count > pre_error_count and post_error is not None:
            return _format_edit_result(
                "REJECTED",
                path,
                f"Edit rejected: introduces syntax error. {post_error}. File unchanged.",
            )

    try:
        full_path.write_text(new_content)
    except OSError as e:
        return _format_edit_result("REJECTED", path, f"Error writing file: {e}")

    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    return _format_edit_result(
        "APPLIED",
        path,
        f"Successfully replaced {old_lines} lines with {new_lines} lines.",
    )


def find_file(pattern: str, *, repo_root: str) -> str:
    root = Path(repo_root)
    matches: list[str] = []
    scanned = 0
    try:
        for path in root.rglob(pattern):
            scanned += 1
            if scanned > _MAX_SCAN:
                break
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.is_file():
                matches.append(str(path.relative_to(root)))
                if len(matches) >= _MAX_RESULTS:
                    break
    except OSError as e:
        return f"Error scanning directory: {e}"
    if not matches:
        return f"No files matching '{pattern}' found."
    result = "\n".join(sorted(matches))
    if len(matches) >= _MAX_RESULTS:
        result += f"\n... (showing first {_MAX_RESULTS} matches)"
    return result


def list_dir(path: str = ".", *, repo_root: str) -> str:
    full_path = Path(repo_root) / path
    if not full_path.is_dir():
        return f"Error: not a directory: {path}"
    try:
        entries = []
        for entry in sorted(full_path.iterdir()):
            if entry.name.startswith(".") and entry.name in _SKIP_DIRS:
                continue
            kind = "dir" if entry.is_dir() else "file"
            entries.append(f"  {entry.name:<40} [{kind}]")
    except OSError as e:
        return f"Error listing {path}: {e}"
    return f"{path}/\n" + "\n".join(entries) if entries else f"{path}/ (empty)"


_TOKEN_STOPWORDS = {
    "and",
    "are",
    "bug",
    "but",
    "can",
    "error",
    "fails",
    "fix",
    "for",
    "from",
    "issue",
    "not",
    "the",
    "this",
    "when",
    "with",
}


def _query_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", query.lower()):
        if len(token) < 3 or token in _TOKEN_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _query_symbol_tokens(query: str) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
        if len(token) < 3:
            continue
        if token.lower() in _TOKEN_STOPWORDS:
            continue
        if not ("_" in token or any(ch.isupper() for ch in token[1:])):
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        symbols.append(lowered)
    return symbols


def _candidate_files(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _CODE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        candidates.append(path)
    return candidates


def _extract_issue_paths(query: str) -> set[str]:
    paths: set[str] = set()
    suffixes = "|".join(re.escape(suffix.lstrip(".")) for suffix in sorted(_CODE_SUFFIXES))
    pattern = re.compile(rf"[\w./-]+\.(?:{suffixes})")
    for match in pattern.finditer(query):
        raw = match.group(0).strip("`'\"(),:;")
        if raw:
            paths.add(raw.lstrip("./"))
    return paths


def _extract_issue_modules(query: str) -> set[str]:
    modules: set[str] = set()
    patterns = (
        r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\b",
        r"\bimport\s+([A-Za-z_][\w.]*)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, query):
            module = match.group(1).strip(".")
            if module and len(module) >= 3:
                modules.add(module.lower().replace(".", "/"))
    return modules


def _read_candidate_text(path: Path) -> str:
    try:
        if path.stat().st_size > 300_000:
            return ""
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _candidate_snippets(text: str, tokens: list[str], *, max_snippets: int = 2) -> list[str]:
    snippets: list[str] = []
    if not tokens:
        return snippets
    lowered_tokens = [token.lower() for token in tokens]
    for lineno, line in enumerate(text.splitlines(), 1):
        line_lower = line.lower()
        if not any(token in line_lower for token in lowered_tokens):
            continue
        snippet = line.strip()
        if not snippet:
            continue
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        snippets.append(f"line {lineno}: {snippet}")
        if len(snippets) >= max_snippets:
            break
    return snippets


def _score_candidate(
    root: Path,
    path: Path,
    *,
    tokens: list[str],
    symbol_tokens: list[str],
    issue_paths: set[str],
    modules: set[str],
) -> tuple[int, list[str], list[str]]:
    rel = path.relative_to(root).as_posix()
    rel_lower = rel.lower()
    name_lower = path.name.lower()
    score = 0
    reasons: list[str] = []

    path_score = sum(rel_lower.count(token) for token in tokens)
    if path_score:
        score += path_score * 6
        reasons.append("path match")

    if rel in issue_paths or any(issue_path.endswith(rel) for issue_path in issue_paths):
        score += 80
        reasons.append("issue path reference")
    elif any(Path(issue_path).name.lower() == name_lower for issue_path in issue_paths):
        score += 45
        reasons.append("issue file reference")

    for module in modules:
        if module in rel_lower or module.split("/")[-1] in name_lower:
            score += 18
            reasons.append("module reference")
            break

    text = _read_candidate_text(path)
    snippets: list[str] = []
    if text:
        text_lower = text.lower()
        content_hits = sum(min(text_lower.count(token), 3) for token in tokens)
        if content_hits:
            score += content_hits
            reasons.append("issue text match")
        symbol_hits = 0
        for symbol in symbol_tokens:
            symbol_hits += len(re.findall(rf"\b{re.escape(symbol)}\b", text_lower))
        if symbol_hits:
            score += min(symbol_hits, 5) * 8
            reasons.append("symbol match")
        snippets = _candidate_snippets(text, symbol_tokens or tokens)

    return score, reasons, snippets


def _rank_candidate_entries(root: Path, query: str) -> list[tuple[int, str, list[str], list[str]]]:
    tokens = _query_tokens(query)
    symbol_tokens = _query_symbol_tokens(query)
    issue_paths = _extract_issue_paths(query)
    modules = _extract_issue_modules(query)
    ranked: list[tuple[int, str, list[str], list[str]]] = []
    for path in _candidate_files(root):
        score, reasons, snippets = _score_candidate(
            root,
            path,
            tokens=tokens,
            symbol_tokens=symbol_tokens,
            issue_paths=issue_paths,
            modules=modules,
        )
        if score:
            ranked.append((score, path.relative_to(root).as_posix(), reasons, snippets))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _rank_candidate_paths(repo_root: Path, query: str) -> list[str]:
    return [rel for _, rel, _, _ in _rank_candidate_entries(repo_root, query)]


def build_candidate_files(repo_root: str, query: str, *, top_n: int = 6) -> str:
    root = Path(repo_root)
    if not root.exists():
        return ""

    ranked = _rank_candidate_entries(root, query)
    if not ranked:
        return ""

    lines = ["Likely files to inspect first:"]
    for _, rel, reasons, snippets in ranked[: max(1, top_n)]:
        reason_text = f" ({', '.join(dict.fromkeys(reasons))})" if reasons else ""
        lines.append(f"- {rel}{reason_text}")
        for snippet in snippets[:2]:
            lines.append(f"  - {snippet}")
    return "\n".join(lines)


def suggest_verification_commands(repo_root: str, *, max_suggestions: int = 4) -> list[str]:
    root = Path(repo_root)
    if not root.exists():
        return []

    suggestions: list[str] = []

    tox_ini = root / "tox.ini"
    if tox_ini.is_file():
        text = _read_candidate_text(tox_ini).lower()
        if "[testenv:py]" in text or re.search(r"envlist\s*=.*\bpy\b", text):
            suggestions.append("tox -e py")
        else:
            suggestions.append("tox")

    manage_py = root / "manage.py"
    if manage_py.is_file():
        text = _read_candidate_text(manage_py).lower()
        if "django" in text or "django_settings_module" in text:
            suggestions.append("python manage.py test")

    runtests_py = root / "runtests.py"
    if runtests_py.is_file():
        suggestions.append("python runtests.py")

    pytest_config = any(
        (root / name).is_file()
        for name in ("pytest.ini", "conftest.py", "setup.cfg", "pyproject.toml")
    )
    has_test_dir = any((root / name).is_dir() for name in ("test", "tests"))
    if pytest_config or has_test_dir:
        suggestions.append("python -m pytest")

    deduped: list[str] = []
    for suggestion in suggestions:
        if suggestion not in deduped:
            deduped.append(suggestion)
        if len(deduped) >= max_suggestions:
            break
    return deduped


def build_repo_map(repo_root: str, query: str, *, max_tokens: int = 4096) -> str:
    root = Path(repo_root)
    if not root.exists():
        return ""

    ranked = _rank_candidate_paths(root, query)
    if not ranked:
        ranked = [path.relative_to(root).as_posix() for path in _candidate_files(root)[:40]]
    if not ranked:
        return ""

    lines = ["Repository map:"]
    current_len = len(lines[0]) + 1
    for rel in ranked:
        line = f"- {rel}"
        if current_len + len(line) + 1 > max_tokens:
            break
        lines.append(line)
        current_len += len(line) + 1
    return "\n".join(lines)
