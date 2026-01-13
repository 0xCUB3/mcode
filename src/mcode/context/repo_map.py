from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RepoMap:
    max_files: int = 200
    max_chars: int = 50_000

    def build_map(self, repo_path: Path) -> str:
        files = [p for p in repo_path.rglob("*") if p.is_file()]
        files = [p for p in files if ".git" not in p.parts and p.suffix in {".py", ".md", ".txt"}]
        files = sorted(files)[: self.max_files]

        chunks: list[str] = []
        for path in files:
            rel = path.relative_to(repo_path)
            summary = self._summarize_file(path)
            chunks.append(f"## {rel}\n{summary}\n")
            if sum(len(c) for c in chunks) > self.max_chars:
                break
        return "\n".join(chunks).strip()

    def _summarize_file(self, path: Path) -> str:
        if path.suffix != ".py":
            try:
                return path.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                return ""
        try:
            return _summarize_python(path)
        except Exception:
            try:
                return path.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                return ""


def _summarize_python(path: Path) -> str:
    try:
        from tree_sitter import Parser
        import tree_sitter_python as tspython

        language = tspython.language()
        parser = Parser()
        parser.language = language
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node

        lines: list[str] = []
        for node in root.children:
            if node.type in {"function_definition", "class_definition"}:
                text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                first_line = text.splitlines()[0] if text else ""
                if first_line:
                    lines.append(first_line)
        return "\n".join(lines) if lines else path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")[:2000]
