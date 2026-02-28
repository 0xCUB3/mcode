from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from mcode.bench.tasks import Task


class CodeOutput(BaseModel):
    code: str = Field(..., description="Python code only, no markdown.")


class FileEdit(BaseModel):
    file: str = Field(..., description="File path relative to repo root")
    search: str = Field(..., description="Exact existing text to find")
    replace: str = Field(..., description="Replacement text")


class PatchOutput(BaseModel):
    edits: list[FileEdit] = Field(..., description="List of search/replace edits to apply")


class LineEdit(BaseModel):
    file: str = Field(..., description="File path relative to repo root")
    start_line: int = Field(..., description="First line number to replace (1-indexed, inclusive)")
    end_line: int = Field(..., description="Last line number to replace (1-indexed, inclusive)")
    replace: str = Field(
        ..., description="Replacement text (replaces lines start_line through end_line)"
    )


class LinePatchOutput(BaseModel):
    edits: list[LineEdit] = Field(..., description="List of line-range edits to apply")


def edits_to_patch(
    raw_json: str,
    repo_root: str = "/testbed",
    *,
    strict: bool = True,
) -> tuple[str, list[str]]:
    """Convert structured edits JSON to a unified diff string.

    Returns (patch_string, error_list).
    """
    import difflib
    import json
    import re
    from pathlib import Path

    try:
        data = json.loads(raw_json)
    except Exception:
        return "", []
    edits = data.get("edits", [])
    if not edits:
        return data.get("patch", ""), []  # fallback for raw diff

    root = Path(repo_root)
    file_index: dict[str, Path] | None = None
    all_paths: list[str] | None = None

    def _normalize_rel_path(rel: str) -> str:
        """Normalize model-provided relative file paths."""
        rel = (rel or "").strip()
        while rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("/"):
            rel = rel.lstrip("/")
        return rel

    def _build_index() -> None:
        nonlocal file_index, all_paths
        if file_index is not None:
            return
        file_index = {}
        all_paths = []
        for p in root.rglob("*.py"):
            if ".git" not in p.parts and "__pycache__" not in p.parts:
                file_index[p.name] = p
                all_paths.append(str(p.relative_to(root)))

    def _suggest_paths(rel: str) -> list[str]:
        """Find real paths similar to a hallucinated one."""
        _build_index()
        assert all_paths is not None
        keywords = set()
        for part in rel.replace("/", " ").replace(".py", "").replace("_", " ").split():
            if len(part) > 3:
                keywords.add(part.lower())
        scored = []
        for p in all_paths:
            p_lower = p.lower()
            hits = sum(1 for kw in keywords if kw in p_lower)
            if hits > 0:
                scored.append((hits, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:5]]

    def _resolve_path(rel: str) -> tuple[str, Path] | None:
        """Resolve a model-provided path, with fuzzy fallback."""
        full = root / rel
        if full.is_file():
            return (rel, full)
        if strict:
            return None
        parts = rel.split("/")
        for i in range(1, len(parts)):
            candidate = "/".join(parts[i:])
            full = root / candidate
            if full.is_file():
                return (candidate, full)
        _build_index()
        assert file_index is not None
        basename = parts[-1] if parts else ""
        if basename in file_index:
            matched = file_index[basename]
            return (str(matched.relative_to(root)), matched)
        return None

    def _fuzzy_find(search: str, text: str) -> tuple[int, int] | None:
        """Find the best fuzzy match for search in text. Returns (start, end)."""
        if strict:
            return None
        if search in text:
            idx = text.index(search)
            return (idx, idx + len(search))
        sm = difflib.SequenceMatcher(None, search, text, autojunk=False)
        # Find the longest contiguous matching block
        best = sm.find_longest_match(0, len(search), 0, len(text))
        if best.size == 0:
            return None
        # Expand: align to line boundaries around the match region in text
        # Use ratio of search vs the candidate region
        s_lines = search.splitlines(keepends=True)
        t_lines = text.splitlines(keepends=True)
        n = len(s_lines)
        best_ratio = 0.0
        best_span: tuple[int, int] | None = None
        # Slide a window of n lines over t_lines
        for start in range(max(0, best.b // 40 - n), min(len(t_lines), best.b // 40 + n + 1)):
            end = start + n
            if end > len(t_lines):
                break
            candidate = "".join(t_lines[start:end])
            ratio = difflib.SequenceMatcher(None, search, candidate, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, end)
        if best_span is None or best_ratio < 0.6:
            return None
        # Convert line span back to char offsets
        char_start = sum(len(ln) for ln in t_lines[: best_span[0]])
        char_end = sum(len(ln) for ln in t_lines[: best_span[1]])
        return (char_start, char_end)

    patches = []
    errors = []
    for edit in edits:
        path = _normalize_rel_path(edit.get("file", ""))
        search = edit.get("search", "")
        replace = edit.get("replace", "")
        resolved = _resolve_path(path)
        if resolved is None:
            suggestions = _suggest_paths(path)
            hint = ""
            if suggestions:
                hint = " Did you mean: " + ", ".join(suggestions)
            errors.append(f"File not found: {path}.{hint}")
            continue
        rel, full = resolved
        try:
            original = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            errors.append(f"Cannot read: {path}")
            continue
        hit_count = original.count(search)
        if hit_count == 1:
            modified = original.replace(search, replace, 1)
        elif hit_count > 1 and strict:
            errors.append(
                f"Search text must match exactly once in {rel}, but matched {hit_count} times."
            )
            continue
        else:
            span = _fuzzy_find(search, original)
            if span is None:
                # Show relevant file content so the model can fix its search text
                lines = original.splitlines()
                # Find lines matching keywords from the search text
                keywords = set(w.lower() for w in re.findall(r"[a-zA-Z_]\w{3,}", search))
                scored: list[tuple[int, int]] = []
                for li, ln in enumerate(lines):
                    hits = sum(1 for kw in keywords if kw in ln.lower())
                    if hits:
                        scored.append((hits, li))
                scored.sort(key=lambda x: -x[0])
                # Show context around top matching lines
                show: set[int] = set()
                for _, li in scored[:3]:
                    for j in range(max(0, li - 2), min(len(lines), li + 3)):
                        show.add(j)
                if not show:
                    show = set(range(min(30, len(lines))))
                snippet_lines = sorted(show)
                snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in snippet_lines)
                errors.append(
                    f"Search text not found in {rel} "
                    f"(your search started with: {search[:60]!r}). "
                    f"Relevant lines from file:\n{snippet}"
                )
                continue
            modified = original[: span[0]] + replace + original[span[1] :]
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        patches.append("".join(diff))
    return "\n".join(patches), errors


def line_edits_to_patch(raw_json: str, repo_root: str = "/testbed") -> tuple[str, list[str]]:
    """Convert line-range edits JSON to a unified diff string.

    Returns (patch_string, error_list).
    """
    import difflib
    import json
    from pathlib import Path

    try:
        data = json.loads(raw_json)
    except Exception:
        return "", []
    edits = data.get("edits", [])
    if not edits:
        return data.get("patch", ""), []

    root = Path(repo_root)

    def _normalize(rel: str) -> str:
        rel = (rel or "").strip()
        while rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("/"):
            rel = rel.lstrip("/")
        # Strip repo_root-like prefixes the model may hallucinate
        root_name = root.name
        for prefix in [f"{root_name}/", "testbed/", "work/testbed/"]:
            if rel.startswith(prefix):
                rel = rel[len(prefix) :]
                break
        return rel

    patches = []
    errors = []
    for edit in edits:
        path = _normalize(edit.get("file", ""))
        start = edit.get("start_line", 0)
        end = edit.get("end_line", 0)
        replace = edit.get("replace", "")

        full = root / path
        if not full.is_file():
            errors.append(f"File not found: {path}")
            continue
        try:
            original = full.read_text(encoding="utf-8", errors="replace")
        except Exception:
            errors.append(f"Cannot read: {path}")
            continue

        lines = original.splitlines(keepends=True)
        if start < 1 or end < start or start > len(lines):
            errors.append(
                f"Invalid line range {start}-{end} in {path} (file has {len(lines)} lines)."
            )
            continue
        end = min(end, len(lines))

        replace_lines = replace.splitlines(keepends=True)
        if replace and not replace.endswith("\n"):
            replace_lines[-1] += "\n"

        modified = lines[: start - 1] + replace_lines + lines[end:]

        # Syntax gate: reject edits that break Python syntax before running tests
        if path.endswith(".py"):
            modified_src = "".join(modified)
            try:
                compile(modified_src, path, "exec")
            except SyntaxError as exc:
                errors.append(f"SyntaxError in {path} line {exc.lineno}: {exc.msg}")
                continue

        diff = difflib.unified_diff(
            lines,
            modified,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        patches.append("".join(diff))
    return "\n".join(patches), errors


@dataclass
class LLMSession:
    model_id: str
    backend_name: str = "ollama"
    loop_budget: int = 3
    temperature: float | None = None
    seed: int | None = None
    strategy_name: str = "repair"
    s2_model_id: str | None = None
    s2_backend_name: str = "ollama"
    s2_solver_mode: str = "best_attempt"
    _m: object | None = field(default=None, repr=False)
    _s2_session: object | None = field(default=None, repr=False)

    def _backend_kwargs(self, *, backend_name: str | None = None) -> dict:
        name = backend_name or self.backend_name
        kwargs: dict = {}
        if name == "ollama":
            base_url = os.environ.get("OLLAMA_HOST")
            if base_url:
                kwargs["base_url"] = base_url
        elif name == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
        return kwargs

    def _model_options(self, *, system_prompt: str) -> dict:
        from mellea.backends import ModelOption

        opts: dict = {ModelOption.SYSTEM_PROMPT: system_prompt}
        if self.temperature is not None:
            opts[ModelOption.TEMPERATURE] = self.temperature
        if self.seed is not None:
            opts[ModelOption.SEED] = self.seed
        raw = os.environ.get("MCODE_MAX_NEW_TOKENS")
        if raw:
            opts[ModelOption.MAX_NEW_TOKENS] = int(raw)
        ctx_raw = os.environ.get("MCODE_CONTEXT_WINDOW")
        if ctx_raw:
            opts[ModelOption.CONTEXT_WINDOW] = int(ctx_raw)
        elif self.backend_name == "ollama":
            opts[ModelOption.CONTEXT_WINDOW] = 16384
        return opts

    def _strategy(self):
        from mellea.stdlib.sampling import RepairTemplateStrategy

        budget = max(1, self.loop_budget)

        if self.strategy_name == "sofai":
            from mellea.stdlib.sampling import SOFAISamplingStrategy

            if self._s2_session is None:
                raise RuntimeError(
                    "SOFAI strategy requires an active S2 session. "
                    "Make sure s2_model_id is set and open() has been called."
                )
            return SOFAISamplingStrategy(
                s1_solver_backend=self._m.backend,
                s2_solver_backend=self._s2_session.backend,
                s2_solver_mode=self.s2_solver_mode,
                loop_budget=budget,
                feedback_strategy="first_error",
            )

        return RepairTemplateStrategy(loop_budget=budget)

    def check_available(self) -> None:
        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        try:
            with mellea.start_session(
                backend_name=self.backend_name,
                model_id=self.model_id,
                **self._backend_kwargs(),
            ):
                return
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Could not start a Mellea session (backend={self.backend_name!r}, "
                f"model_id={self.model_id!r}). "
                "Ensure the backend is running and accessible (e.g. Ollama server) and retry."
            ) from e

    @contextmanager
    def open(self):
        if self._m is not None:
            yield self
            return

        try:
            import mellea
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "mellea is required for LLM interaction; "
                "install dependencies with `uv pip install -e .`"
            ) from e

        ctx = None
        if self.strategy_name == "sofai":
            from mellea.stdlib.context import ChatContext

            ctx = ChatContext()

        with mellea.start_session(
            backend_name=self.backend_name,
            model_id=self.model_id,
            ctx=ctx,
            **self._backend_kwargs(),
        ) as m:
            self._m = m
            try:
                if self.strategy_name == "sofai" and self.s2_model_id:
                    with mellea.start_session(
                        backend_name=self.s2_backend_name,
                        model_id=self.s2_model_id,
                        ctx=ChatContext(),
                        **self._backend_kwargs(backend_name=self.s2_backend_name),
                    ) as s2:
                        self._s2_session = s2
                        try:
                            yield self
                        finally:
                            self._s2_session = None
                else:
                    yield self
            finally:
                self._m = None

    def generate_code(self, *, task: Task, requirements: list | None = None):
        system_prompt = _code_system_prompt(task)
        return self._m.instruct(
            task.prompt,
            format=CodeOutput,
            strategy=self._strategy(),
            requirements=requirements or [],
            return_sampling_results=True,
            model_options=self._model_options(system_prompt=system_prompt),
        )

    def generate_patch(
        self,
        *,
        repo: str,
        problem_statement: str,
        hints_text: str = "",
        file_paths: list[str] | None = None,
        requirements: list | None = None,
    ):
        file_constraint = ""
        if file_paths:
            file_list = "\n".join(f"  - {f}" for f in file_paths)
            file_constraint = (
                f"\n\nYou may ONLY edit these files:\n{file_list}\nDo not edit any other files."
            )
        system_prompt = (
            "You are an expert software engineer.\n"
            "Given a GitHub issue, a repository, and relevant source files, "
            "produce line-range edits to fix the issue.\n"
            "Each edit has four fields:\n"
            '  - "file": path relative to repo root\n'
            '  - "start_line": first line number to replace (1-indexed, inclusive)\n'
            '  - "end_line": last line number to replace (1-indexed, inclusive)\n'
            '  - "replace": the new text that replaces those lines\n'
            "Rules:\n"
            "- Source files are shown with line numbers (e.g. '42: code here'). "
            "Use these line numbers for start_line and end_line.\n"
            "- Use file paths exactly as shown in the hints section.\n"
            "- To insert new lines without removing existing ones, "
            "set start_line and end_line to the same line and include "
            "that original line plus your new lines in replace.\n"
            "- Keep edits minimal: only change what is needed "
            "to fix the issue." + file_constraint
        )
        hints_block = f"\n\nHints:\n{hints_text.strip()}" if hints_text.strip() else ""
        description = f"Repository: {repo}\n\nIssue:\n{problem_statement.strip()}{hints_block}"
        return self._m.instruct(
            description,
            format=LinePatchOutput,
            strategy=self._strategy(),
            requirements=requirements or [],
            return_sampling_results=True,
            model_options=self._model_options(system_prompt=system_prompt),
        )


def _code_system_prompt(task: Task) -> str:
    if task.benchmark == "humaneval":
        return (
            "You are an expert Python programmer.\n"
            "Complete the function defined in the prompt.\n"
            "Keep the function name and signature exactly the same."
        )
    return "You are an expert Python programmer."
