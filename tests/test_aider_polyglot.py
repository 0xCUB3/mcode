from __future__ import annotations

from pathlib import Path

from mcode.bench.aider_polyglot import (
    cleanup_prepared_task,
    load_aider_polyglot,
    prepare_task,
    run_single_command,
)


def _make_exercise(root: Path, language: str, exercise: str, files: dict[str, str]) -> Path:
    exercise_dir = root / language / "exercises" / "practice" / exercise
    for relative_path, content in files.items():
        path = exercise_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return exercise_dir


def test_load_aider_polyglot_filters_language_and_task_ids(tmp_path: Path) -> None:
    _make_exercise(
        tmp_path,
        "python",
        "hello-world",
        {
            "hello_world.py": "",
            "hello_world_test.py": "def test_ok():\n    assert True\n",
        },
    )
    _make_exercise(
        tmp_path,
        "go",
        "hello-world",
        {
            "hello_world.go": "package greeting\n",
            "hello_world_test.go": "package greeting\n",
            "go.mod": "module example.com/hello\n",
        },
    )

    python_tasks = load_aider_polyglot(tmp_path, language="python")
    assert [task.task_id for task in python_tasks] == ["python/hello-world"]

    filtered = load_aider_polyglot(
        tmp_path,
        task_ids=["go/hello-world"],
    )
    assert [task.task_id for task in filtered] == ["go/hello-world"]


def test_prepare_task_strips_meta_and_javascript_skip_markers(tmp_path: Path) -> None:
    _make_exercise(
        tmp_path,
        "javascript",
        "lasagna",
        {
            "lasagna.js": "export const answer = 42;\n",
            "lasagna.spec.js": "xit('works', () => expect(true).toBe(true));\n",
            ".meta/example.js": "export const cheat = true;\n",
        },
    )

    task = load_aider_polyglot(tmp_path, language="javascript")[0]
    prepared = prepare_task(task, benchmark_root=tmp_path)
    try:
        assert (prepared.work_dir / "lasagna.js").exists()
        assert not (prepared.work_dir / ".meta").exists()
        assert "it('works'" in (prepared.work_dir / "lasagna.spec.js").read_text()
        assert (prepared.work_dir / ".git").is_dir()
    finally:
        cleanup_prepared_task(prepared)


def test_run_single_command_replaces_invalid_utf8(tmp_path: Path) -> None:
    result = run_single_command(
        tmp_path,
        "python -c 'import sys; sys.stdout.buffer.write(bytes([0xff]))'",
        timeout_s=10,
    )

    assert result.passed
    assert "�" in result.output
