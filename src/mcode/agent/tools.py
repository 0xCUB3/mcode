from __future__ import annotations

import re
import signal
import subprocess
from pathlib import Path

_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".tox", "build", "dist", ".venv", "venv"}
)
_SOURCE_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".cfg",
        ".ini",
        ".txt",
        ".md",
        ".rst",
    }
)


def _safe_search(pattern: str, line: str) -> bool:
    """Regex search with a 2-second timeout to prevent catastrophic backtracking."""

    def _timeout_handler(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(2)
    try:
        return bool(re.search(pattern, line, re.IGNORECASE))
    except TimeoutError:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def make_tools(repo_root: str) -> dict[str, callable]:
    root = Path(repo_root)

    def search_code(query: str) -> str:
        """Search for code matching a pattern in the repository.

        Args:
            query: regex pattern or literal string to search for
        """
        print(f"  [tool] search_code({query!r})", flush=True)
        matches: list[str] = []
        try:
            re.compile(query)
        except re.error:
            query = re.escape(query)
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if _SKIP_DIRS.intersection(p.parts):
                continue
            if p.suffix not in _SOURCE_SUFFIXES:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if _safe_search(query, line):
                    rel = str(p.relative_to(root))
                    matches.append(f"{rel}:{i}: {line.strip()}")
                    if len(matches) >= 20:
                        break
            if len(matches) >= 20:
                break
        if not matches:
            return "No matches found."
        return "\n".join(matches)

    def read_file(path: str, start_line: int, end_line: int) -> str:
        """Read lines from a file with line numbers.

        Args:
            path: file path relative to repo root
            start_line: first line to read (1-indexed)
            end_line: last line to read (1-indexed, inclusive)
        """
        print(f"  [tool] read_file({path!r}, {start_line}, {end_line})", flush=True)
        fp = root / path
        if not fp.is_file():
            return f"Error: file not found: {path}"
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return f"Error: cannot read {path}"
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)
        if end_line - start_line + 1 > 200:
            end_line = start_line + 199
        selected = lines[start_line - 1 : end_line]
        numbered = [f"{start_line + i}: {line}" for i, line in enumerate(selected)]
        header = f"--- {path} (lines {start_line}-{end_line} of {len(lines)}) ---"
        return header + "\n" + "\n".join(numbered)

    def apply_edit(path: str, start_line: int, end_line: int, replacement: str) -> str:
        """Replace lines in a file. Validates Python syntax before accepting.

        Args:
            path: file path relative to repo root
            start_line: first line to replace (1-indexed, inclusive)
            end_line: last line to replace (1-indexed, inclusive)
            replacement: new text to replace those lines with
        """
        preview = replacement[:80] + "..." if len(replacement) > 80 else replacement
        print(
            f"  [tool] apply_edit({path!r}, {start_line}, {end_line}, {preview!r})",
            flush=True,
        )
        fp = root / path
        if not fp.is_file():
            return f"Error: file not found: {path}"
        try:
            original = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return f"Error: cannot read {path}"
        lines = original.splitlines(keepends=True)
        if start_line < 1 or end_line < start_line or start_line > len(lines):
            return (
                f"Error: invalid line range {start_line}-{end_line} (file has {len(lines)} lines)"
            )
        end_line = min(end_line, len(lines))

        replace_lines = replacement.splitlines(keepends=True)
        if replacement and not replacement.endswith("\n"):
            replace_lines[-1] += "\n"

        modified_lines = lines[: start_line - 1] + replace_lines + lines[end_line:]
        modified = "".join(modified_lines)

        if path.endswith(".py"):
            try:
                compile(modified, path, "exec")
            except SyntaxError as exc:
                return (
                    f"Error: SyntaxError in {path} line {exc.lineno}: {exc.msg}."
                    " Edit rejected, file unchanged."
                )

        fp.write_text(modified, encoding="utf-8")
        return (
            f"OK: replaced lines {start_line}-{end_line} in {path} ({len(replace_lines)} new lines)"
        )

    return {
        "search_code": search_code,
        "read_file": read_file,
        "apply_edit": apply_edit,
    }


def get_diff(repo_root: str) -> str:
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.stdout
