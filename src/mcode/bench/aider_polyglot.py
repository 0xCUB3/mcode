from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from mcode.util import temporary_directory

_BENCHMARK_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"
_JS_SKIP_MARKER_RE = re.compile(r"\b(xit|xtest|xdescribe)\s*\(")
_JAVA_DISABLED_RE = re.compile(r"^[ \t]*@Disabled\b.*$", re.MULTILINE)
_LANGUAGE_ORDER = ("python", "go", "rust", "javascript", "cpp", "java")


@dataclass(frozen=True)
class AiderPolyglotTask:
    benchmark: str
    task_id: str
    language: str
    exercise: str
    source_dir: Path

    @property
    def repo(self) -> str:
        return f"aider-polyglot/{self.language}/{self.exercise}"


@dataclass(frozen=True)
class PreparedPolyglotTask:
    task: AiderPolyglotTask
    work_dir: Path
    stub_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    test_commands: tuple[str, ...]
    timeout_s: int
    tempdir: TemporaryDirectory[str] = field(repr=False, compare=False)

    def build_first_prompt(self) -> str:
        stub_line = (
            f"The stub file you need to implement is at {self.stub_paths[0]}."
            if self.stub_paths
            else ""
        )
        return (
            f"Please implement the '{self.task.exercise}' exercise. "
            f"The working directory is {self.work_dir}. "
            f"{stub_line} "
            "Explore the directory if you need more context, then implement "
            "the solution and run the tests to verify."
        ).strip()

    def build_retry_prompt(self, test_output: str) -> str:
        stub_line = f"The stub file is at {self.stub_paths[0]}." if self.stub_paths else ""
        return (
            f"The tests for '{self.task.exercise}' failed. Here is the test output:\n\n"
            f"```\n{test_output[:2000]}\n```\n\n"
            f"{stub_line} Please fix the implementation so the tests pass."
        ).strip()


@dataclass(frozen=True)
class CommandOutcome:
    passed: bool
    output: str
    exit_code: int | None
    timed_out: bool


@dataclass(frozen=True)
class _LanguageDescriptor:
    practice_dir: Path
    prepare: Callable[[Path, Path], tuple[list[str], list[str]]]
    test_commands: tuple[str, ...]
    timeout_s: int


def default_benchmark_root() -> Path:
    override = os.environ.get("MCODE_AIDER_POLYGLOT_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / "polyglot-benchmark"


def ensure_benchmark_root(path: str | Path | None = None) -> Path:
    root = Path(path).expanduser() if path is not None else default_benchmark_root()
    if root.is_dir():
        return root
    raise RuntimeError(
        f"Aider Polyglot benchmark not found at {root}. Clone it with: "
        f"git clone {_BENCHMARK_REPO} {root}"
    )


def supported_languages() -> tuple[str, ...]:
    return _LANGUAGE_ORDER


def load_aider_polyglot(
    root: str | Path | None = None,
    *,
    language: str = "all",
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[AiderPolyglotTask]:
    benchmark_root = ensure_benchmark_root(root)
    descriptors = _build_language_descriptors(benchmark_root)

    if language != "all" and language not in descriptors:
        known = ", ".join(_LANGUAGE_ORDER)
        raise ValueError(f"unknown language {language!r}. Expected one of: {known}, all")

    task_filter = set(task_ids or [])
    selected_languages = _LANGUAGE_ORDER if language == "all" else (language,)
    tasks: list[AiderPolyglotTask] = []
    for language_name in selected_languages:
        descriptor = descriptors[language_name]
        if not descriptor.practice_dir.is_dir():
            if language == "all":
                continue
            raise RuntimeError(f"practice dir not found: {descriptor.practice_dir}")
        for exercise_dir in sorted(p for p in descriptor.practice_dir.iterdir() if p.is_dir()):
            task = AiderPolyglotTask(
                benchmark="aider-polyglot",
                task_id=f"{language_name}/{exercise_dir.name}",
                language=language_name,
                exercise=exercise_dir.name,
                source_dir=exercise_dir,
            )
            if task_filter and task.task_id not in task_filter:
                continue
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                return tasks
    return tasks


def prepare_task(
    task: AiderPolyglotTask,
    *,
    benchmark_root: str | Path | None = None,
) -> PreparedPolyglotTask:
    root = ensure_benchmark_root(benchmark_root)
    descriptors = _build_language_descriptors(root)
    descriptor = descriptors[task.language]
    tempdir = temporary_directory(prefix=f"mcode-polyglot-{task.language}-")
    work_parent = Path(tempdir.name)
    work_dir = work_parent / task.exercise
    stub_paths, test_paths = descriptor.prepare(task.source_dir, work_dir)
    _init_git_repo(work_dir)
    return PreparedPolyglotTask(
        task=task,
        work_dir=work_dir,
        stub_paths=tuple(stub_paths),
        test_paths=tuple(test_paths),
        test_commands=descriptor.test_commands,
        timeout_s=descriptor.timeout_s,
        tempdir=tempdir,
    )


def cleanup_prepared_task(task: PreparedPolyglotTask) -> None:
    task.tempdir.cleanup()


def run_test_commands(task: PreparedPolyglotTask) -> CommandOutcome:
    return run_command_sequence(task.work_dir, task.test_commands, timeout_s=task.timeout_s)


def apply_patch_to_prepared_task(task: PreparedPolyglotTask, patch: str) -> CommandOutcome:
    if not patch.strip():
        return CommandOutcome(
            passed=False,
            output="No patch candidate found.",
            exit_code=None,
            timed_out=False,
        )
    patch_file = task.work_dir / ".mcode-candidate.patch"
    patch_file.write_text(patch, encoding="utf-8")
    try:
        outputs: list[str] = []
        for command in (
            f"git apply --verbose {patch_file.name}",
            f"git apply --verbose --reject {patch_file.name}",
            f"patch --batch --fuzz=5 -p1 -i {patch_file.name}",
        ):
            outcome = run_single_command(task.work_dir, command, timeout_s=task.timeout_s)
            outputs.append(f"$ {command}\n{outcome.output}".strip())
            if outcome.passed:
                return CommandOutcome(
                    passed=True,
                    output="\n\n".join(part for part in outputs if part),
                    exit_code=outcome.exit_code,
                    timed_out=False,
                )
            if outcome.timed_out:
                return CommandOutcome(
                    passed=False,
                    output="\n\n".join(part for part in outputs if part),
                    exit_code=outcome.exit_code,
                    timed_out=True,
                )
        return CommandOutcome(
            passed=False,
            output="\n\n".join(part for part in outputs if part),
            exit_code=1,
            timed_out=False,
        )
    finally:
        if patch_file.exists():
            patch_file.unlink()


def run_command_sequence(
    work_dir: Path,
    commands: tuple[str, ...],
    *,
    timeout_s: int,
) -> CommandOutcome:
    outputs: list[str] = []
    last_exit_code: int | None = None
    for command in commands:
        outcome = run_single_command(work_dir, command, timeout_s=timeout_s)
        last_exit_code = outcome.exit_code
        outputs.append(f"$ {command}\n{outcome.output}".strip())
        if not outcome.passed:
            return CommandOutcome(
                passed=False,
                output="\n\n".join(part for part in outputs if part),
                exit_code=outcome.exit_code,
                timed_out=outcome.timed_out,
            )
    return CommandOutcome(
        passed=True,
        output="\n\n".join(part for part in outputs if part),
        exit_code=last_exit_code,
        timed_out=False,
    )


def run_single_command(work_dir: Path, command: str, *, timeout_s: int) -> CommandOutcome:
    env = os.environ.copy()
    for extra_bin in (
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/opt/openjdk@21/bin"),
        Path("/opt/homebrew/opt/openjdk/bin"),
    ):
        if extra_bin.is_dir():
            env["PATH"] = f"{extra_bin}:{env.get('PATH', '')}"
    for java_home in (
        Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"),
        Path("/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"),
    ):
        if java_home.is_dir() and not env.get("JAVA_HOME"):
            env["JAVA_HOME"] = str(java_home)
            break
    if command.startswith("cargo "):
        env["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{env.get('PATH', '')}"
    command_timeout = timeout_s
    if command.startswith("npm install "):
        command_timeout = max(120, timeout_s)

    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=work_dir,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=command_timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return CommandOutcome(
            passed=False,
            output=f"Command timed out after {command_timeout}s",
            exit_code=None,
            timed_out=True,
        )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    return CommandOutcome(
        passed=result.returncode == 0,
        output=output,
        exit_code=result.returncode,
        timed_out=False,
    )


def _copy_exercise(src_dir: Path, work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name == ".meta":
            continue
        dest = work_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def _prepare_python(src: Path, work: Path) -> tuple[list[str], list[str]]:
    _copy_exercise(src, work)
    stubs = sorted(
        str(p)
        for p in work.iterdir()
        if p.is_file() and p.suffix == ".py" and "_test" not in p.name and p.name != "__init__.py"
    )
    tests = sorted(str(p) for p in work.iterdir() if p.is_file() and p.name.endswith("_test.py"))
    return stubs, tests


def _prepare_go(src: Path, work: Path) -> tuple[list[str], list[str]]:
    _copy_exercise(src, work)
    stubs = sorted(
        str(p)
        for p in work.iterdir()
        if p.is_file()
        and p.suffix == ".go"
        and not p.name.endswith("_test.go")
        and p.name != "go.mod"
    )
    tests = sorted(str(p) for p in work.iterdir() if p.name.endswith("_test.go"))
    return stubs, tests


def _prepare_rust(src: Path, work: Path) -> tuple[list[str], list[str]]:
    _copy_exercise(src, work)
    stubs = [str(work / "src" / "lib.rs")] if (work / "src" / "lib.rs").exists() else []
    tests_dir = work / "tests"
    tests = sorted(str(p) for p in tests_dir.glob("*.rs")) if tests_dir.exists() else []
    return stubs, tests


def _prepare_javascript(src: Path, work: Path) -> tuple[list[str], list[str]]:
    _copy_exercise(src, work)
    for spec in work.glob("*.spec.js"):
        text = spec.read_text()
        new_text = _JS_SKIP_MARKER_RE.sub(
            lambda match: {
                "xit": "it(",
                "xtest": "test(",
                "xdescribe": "describe(",
            }[match.group(1)],
            text,
        )
        if new_text != text:
            spec.write_text(new_text)
    stubs = sorted(
        str(p)
        for p in work.iterdir()
        if p.is_file()
        and p.suffix == ".js"
        and not p.name.endswith(".spec.js")
        and p.name != "babel.config.js"
    )
    tests = sorted(str(p) for p in work.iterdir() if p.name.endswith(".spec.js"))
    return stubs, tests


def _prepare_java(src: Path, work: Path) -> tuple[list[str], list[str]]:
    _copy_exercise(src, work)
    test_root = work / "src" / "test" / "java"
    if test_root.exists():
        for java_file in test_root.rglob("*.java"):
            text = java_file.read_text()
            new_text = _JAVA_DISABLED_RE.sub("", text)
            if new_text != text:
                java_file.write_text(new_text)
    gradlew = work / "gradlew"
    if gradlew.exists():
        gradlew.chmod(0o755)
    main_java = work / "src" / "main" / "java"
    stubs = sorted(str(p) for p in main_java.rglob("*.java")) if main_java.exists() else []
    tests = sorted(str(p) for p in test_root.rglob("*.java")) if test_root.exists() else []
    return stubs, tests


def _prepare_cpp(src: Path, work: Path) -> tuple[list[str], list[str]]:
    _copy_exercise(src, work)
    stubs = sorted(
        str(p)
        for p in work.iterdir()
        if p.is_file() and p.suffix in (".cpp", ".h") and not p.name.endswith("_test.cpp")
    )
    tests = sorted(str(p) for p in work.iterdir() if p.name.endswith("_test.cpp"))
    return stubs, tests


def _build_language_descriptors(root: Path) -> dict[str, _LanguageDescriptor]:
    return {
        "python": _LanguageDescriptor(
            practice_dir=root / "python" / "exercises" / "practice",
            prepare=_prepare_python,
            test_commands=("python -m pytest *_test.py -v --tb=short -q",),
            timeout_s=60,
        ),
        "go": _LanguageDescriptor(
            practice_dir=root / "go" / "exercises" / "practice",
            prepare=_prepare_go,
            test_commands=("go test ./...",),
            timeout_s=60,
        ),
        "rust": _LanguageDescriptor(
            practice_dir=root / "rust" / "exercises" / "practice",
            prepare=_prepare_rust,
            test_commands=("cargo test -- --include-ignored",),
            timeout_s=180,
        ),
        "javascript": _LanguageDescriptor(
            practice_dir=root / "javascript" / "exercises" / "practice",
            prepare=_prepare_javascript,
            test_commands=(
                "npm install --silent --no-audit --no-fund",
                "npm test --silent",
            ),
            timeout_s=90,
        ),
        "cpp": _LanguageDescriptor(
            practice_dir=root / "cpp" / "exercises" / "practice",
            prepare=_prepare_cpp,
            test_commands=(
                "cmake -S . -B build -DCMAKE_CXX_FLAGS=-DEXERCISM_RUN_ALL_TESTS",
                "cmake --build build",
            ),
            timeout_s=240,
        ),
        "java": _LanguageDescriptor(
            practice_dir=root / "java" / "exercises" / "practice",
            prepare=_prepare_java,
            test_commands=("./gradlew test --no-daemon -q",),
            timeout_s=300,
        ),
    }


def _init_git_repo(work_dir: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "mcode",
        "GIT_AUTHOR_EMAIL": "mcode@example.com",
        "GIT_COMMITTER_NAME": "mcode",
        "GIT_COMMITTER_EMAIL": "mcode@example.com",
    }
    subprocess.run(
        ["git", "init"],
        cwd=work_dir,
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=work_dir,
        capture_output=True,
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=work_dir,
        capture_output=True,
        check=True,
        env=env,
    )
