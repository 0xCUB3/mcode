from __future__ import annotations

from pathlib import Path

from .models import PreparedPolyglotTask


def build_first_prompt(task: PreparedPolyglotTask) -> str:
    stub_line = _path_list(
        "Implementation file(s) you may edit",
        task.stub_paths,
        task.work_dir,
    )
    test_line = _path_list("Relevant test file(s)", task.test_paths, task.work_dir)
    command_line = ", ".join(task.test_commands)
    docs = _docs_excerpt(task.work_dir)
    docs_block = f"\nExercise instructions:\n{docs}\n" if docs else ""
    language_note = _language_note(task.task.language)
    reactive_note = _reactive_note(task.work_dir, task.stub_paths, task.test_paths)
    return (
        f"Please implement the '{task.task.exercise}' exercise. "
        f"The working directory is {task.work_dir}.\n"
        f"{stub_line}\n"
        f"{test_line}\n"
        f"Test command: {command_line}\n"
        f"{docs_block}\n"
        f"{language_note}"
        f"{reactive_note}"
        "Only edit the listed implementation file(s). Do not edit tests, docs, "
        "build files, dependency files, wrappers, or generated files. Read the "
        "implementation and relevant tests, make the smallest correct edit, then "
        "run the default tests."
    ).strip()


def build_retry_prompt(task: PreparedPolyglotTask, test_output: str) -> str:
    stub_line = _path_list(
        "Implementation file(s) you may edit",
        task.stub_paths,
        task.work_dir,
    )
    test_line = _path_list("Relevant test file(s)", task.test_paths, task.work_dir)
    docs = _docs_excerpt(task.work_dir)
    docs_block = f"\nExercise instructions:\n{docs}\n" if docs else ""
    language_note = _language_note(task.task.language)
    reactive_note = _reactive_note(
        task.work_dir,
        task.stub_paths,
        task.test_paths,
        test_output,
    )
    return (
        f"The tests for '{task.task.exercise}' failed. The working directory is "
        f"{task.work_dir}; paths below are relative to that directory. Here is "
        f"the test output:\n\n"
        f"```\n{_retry_output_excerpt(test_output)}\n```\n\n"
        f"{stub_line}\n"
        f"{test_line}\n"
        f"{docs_block}\n"
        f"{language_note}"
        f"{reactive_note}"
        "Fix only the listed implementation file(s). Do not edit tests, docs, "
        "build files, dependency files, wrappers, or generated files. Run the "
        "default tests after the edit."
    ).strip()


def _retry_output_excerpt(output: str, *, max_chars: int = 3000) -> str:
    if len(output) <= max_chars:
        return output
    edge = max_chars // 2
    head = output[:edge].rstrip()
    tail = output[-edge:].lstrip()
    return f"{head}\n...[test output truncated, keeping final diagnostics]...\n{tail}"


def _path_list(label: str, paths: tuple[str, ...], root: Path) -> str:
    if not paths:
        return f"{label}: (none)"
    rel_paths = []
    for path in paths:
        candidate = Path(path)
        try:
            rel_paths.append(str(candidate.relative_to(root)))
        except ValueError:
            rel_paths.append(str(candidate))
    return f"{label}: {', '.join(rel_paths)}"


def _reactive_note(
    work_dir: Path,
    stub_paths: tuple[str, ...],
    test_paths: tuple[str, ...],
    test_output: str = "",
) -> str:
    for rel_path in (*stub_paths, *test_paths):
        path = Path(rel_path)
        if not path.is_absolute():
            path = work_dir / path
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "io.reactivex.Observable" in text or "rx.Observable" in text:
            note = (
                "Reactive stream APIs can often be implemented with a simple Observable.create "
                "and ordinary local state when the tests use synchronous observables; avoid "
                "complex combinators unless they directly match the required event ordering."
            )
            if "TIMEOUT" in test_output or "Command timed out" in test_output:
                note += (
                    " A timeout in reactive tests often means the returned stream never "
                    "completed; make sure finite source completion reaches the returned stream."
                )
            return f"{note}\n"
    return ""


def _language_note(language: str) -> str:
    if language == "rust":
        return (
            "Rust Cargo.toml dependencies from the official exercise metadata are already "
            "loaded when provided; inspect Cargo.toml if the docs mention helper crates.\n"
        )
    return ""


def _docs_excerpt(work_dir: Path, *, max_chars: int = 5000) -> str:
    docs_dir = work_dir / ".docs"
    if not docs_dir.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(errors="replace").strip()
        if not text:
            continue
        parts.append(f"## {path.name}\n{text}")
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[docs truncated]"
    return text
