from __future__ import annotations

from pathlib import Path

from mcode.bench.aider_polyglot import (
    cleanup_prepared_task,
    load_aider_polyglot,
    prepare_task,
    reset_to_baseline,
    run_command_sequence,
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


def test_prepare_task_loads_rust_example_dependencies(tmp_path: Path) -> None:
    _make_exercise(
        tmp_path,
        "rust",
        "decimal",
        {
            "Cargo.toml": '[package]\nedition = "2021"\nname = "decimal"\nversion = "0.1.0"\n',
            "src/lib.rs": "pub struct Decimal;\n",
            "tests/decimal.rs": "#[test]\nfn ok() {}\n",
            ".meta/Cargo-example.toml": (
                '[package]\nedition = "2021"\nname = "decimal"\nversion = "0.1.0"\n\n'
                '[dependencies]\nnum-bigint = "0.4.4"\nnum-traits = "0.2.16"\n'
            ),
        },
    )

    task = load_aider_polyglot(tmp_path, language="rust")[0]
    prepared = prepare_task(task, benchmark_root=tmp_path)
    try:
        cargo = (prepared.work_dir / "Cargo.toml").read_text()
        assert not (prepared.work_dir / ".meta").exists()
        assert 'num-bigint = "0.4.4"' in cargo
        assert 'num-traits = "0.2.16"' in cargo
    finally:
        cleanup_prepared_task(prepared)


def test_reset_to_baseline_restores_prepared_task(tmp_path: Path) -> None:
    _make_exercise(
        tmp_path,
        "python",
        "hello-world",
        {
            "hello_world.py": "def hello():\n    return 'hello'\n",
            "hello_world_test.py": "def test_ok():\n    assert True\n",
        },
    )

    task = load_aider_polyglot(tmp_path, language="python")[0]
    prepared = prepare_task(task, benchmark_root=tmp_path)
    try:
        target = prepared.work_dir / "hello_world.py"
        target.write_text("broken\n")
        (prepared.work_dir / "scratch.txt").write_text("remove me\n")

        reset_to_baseline(prepared.work_dir)

        assert target.read_text() == "def hello():\n    return 'hello'\n"
        assert not (prepared.work_dir / "scratch.txt").exists()
    finally:
        cleanup_prepared_task(prepared)


def test_run_command_sequence_appends_failure_reports(tmp_path: Path) -> None:
    report_dir = tmp_path / "build" / "test-results" / "test"
    report_dir.mkdir(parents=True)
    (report_dir / "TEST-example.xml").write_text(
        '<testsuite><testcase name="badCase" classname="ExampleTest">'
        '<failure message="expected 1 but was 2">stack trace</failure>'
        "</testcase></testsuite>"
    )

    result = run_command_sequence(tmp_path, ("false",), timeout_s=10)

    assert not result.passed
    assert "Failure report snippets:" in result.output
    assert "ExampleTest badCase expected 1 but was 2" in result.output


def test_run_single_command_replaces_invalid_utf8(tmp_path: Path) -> None:
    result = run_single_command(
        tmp_path,
        "python -c 'import sys; sys.stdout.buffer.write(bytes([0xff]))'",
        timeout_s=10,
    )

    assert result.passed
    assert "�" in result.output
