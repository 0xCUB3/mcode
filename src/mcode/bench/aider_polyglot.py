from __future__ import annotations

import importlib.resources
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
        stub_line = _path_list(
            "Implementation file(s) you may edit",
            self.stub_paths,
            self.work_dir,
        )
        test_line = _path_list("Relevant test file(s)", self.test_paths, self.work_dir)
        command_line = ", ".join(self.test_commands)
        docs = _docs_excerpt(self.work_dir)
        docs_block = f"\nExercise instructions:\n{docs}\n" if docs else ""
        language_note = _language_note(self.task.language)
        reactive_note = _reactive_note(self.work_dir, self.stub_paths, self.test_paths)
        return (
            f"Please implement the '{self.task.exercise}' exercise. "
            f"The working directory is {self.work_dir}.\n"
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

    def build_retry_prompt(self, test_output: str) -> str:
        stub_line = _path_list(
            "Implementation file(s) you may edit",
            self.stub_paths,
            self.work_dir,
        )
        test_line = _path_list("Relevant test file(s)", self.test_paths, self.work_dir)
        docs = _docs_excerpt(self.work_dir)
        docs_block = f"\nExercise instructions:\n{docs}\n" if docs else ""
        language_note = _language_note(self.task.language)
        reactive_note = _reactive_note(
            self.work_dir,
            self.stub_paths,
            self.test_paths,
            test_output,
        )
        return (
            f"The tests for '{self.task.exercise}' failed. The working directory is "
            f"{self.work_dir}; paths below are relative to that directory. Here is "
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


@dataclass(frozen=True)
class CommandOutcome:
    passed: bool
    output: str
    exit_code: int | None
    timed_out: bool


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


def reset_to_baseline(work_dir: Path) -> None:
    subprocess.run(
        ["git", "reset", "--hard", "HEAD"], cwd=work_dir, capture_output=True, check=True
    )
    subprocess.run(["git", "clean", "-fd"], cwd=work_dir, capture_output=True, check=True)


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
            output = "\n\n".join(part for part in outputs if part)
            if snippets := _failure_report_snippets(work_dir, output):
                output = f"{output}\n\nFailure report snippets:\n{snippets}"
            return CommandOutcome(
                passed=False,
                output=output,
                exit_code=outcome.exit_code,
                timed_out=outcome.timed_out,
            )
    return CommandOutcome(
        passed=True,
        output="\n\n".join(part for part in outputs if part),
        exit_code=last_exit_code,
        timed_out=False,
    )


def _failure_report_snippets(work_dir: Path, output: str = "") -> str:
    from mcode.agent.verification import _collect_failure_artifacts

    return _collect_failure_artifacts(work_dir, output)


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
    boost_shim = work_dir / ".mcode-boost-shim"
    if boost_shim.is_dir():
        env["BOOST_ROOT"] = str(boost_shim)
        env["BOOST_INCLUDEDIR"] = str(boost_shim / "include")
        env["BOOST_LIBRARYDIR"] = str(boost_shim / "lib")
        env["Boost_NO_SYSTEM_PATHS"] = "ON"
        if command.startswith("cmake -S ") and "BOOST_INCLUDEDIR" not in command:
            boost_config = boost_shim / "lib" / "cmake" / "Boost-1.74.0"
            command = (
                f"{command} -DCMAKE_POLICY_DEFAULT_CMP0167=NEW "
                f"-DBoost_DIR={boost_config} -DCMAKE_PREFIX_PATH={boost_shim} "
                f"-DBOOST_ROOT={boost_shim} "
                f"-DBoost_INCLUDE_DIR={boost_shim / 'include'} "
                f"-DBOOST_INCLUDEDIR={boost_shim / 'include'} "
                f"-DBoost_LIBRARY_DIR_RELEASE={boost_shim / 'lib'} "
                f"-DBoost_DATE_TIME_LIBRARY={boost_shim / 'lib' / 'libboost_date_time.a'} "
                f"-DBOOST_LIBRARYDIR={boost_shim / 'lib'}"
            )
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
        cleanup_note = _cleanup_after_timed_out_command(work_dir, command)
        output = f"Command timed out after {command_timeout}s"
        if cleanup_note:
            output = f"{output}\n{cleanup_note}"
        return CommandOutcome(
            passed=False,
            output=output,
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


def _cleanup_after_timed_out_command(work_dir: Path, command: str) -> str:
    if "npm install" not in command:
        return ""
    node_modules = work_dir / "node_modules"
    if not node_modules.exists():
        return ""
    shutil.rmtree(node_modules, ignore_errors=True)
    return "Removed partial node_modules after npm install timeout."


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
    _merge_rust_example_dependencies(src, work)
    stubs = [str(work / "src" / "lib.rs")] if (work / "src" / "lib.rs").exists() else []
    tests_dir = work / "tests"
    tests = sorted(str(p) for p in tests_dir.glob("*.rs")) if tests_dir.exists() else []
    return stubs, tests


def _merge_rust_example_dependencies(src: Path, work: Path) -> None:
    example = src / ".meta" / "Cargo-example.toml"
    cargo = work / "Cargo.toml"
    if not example.exists() or not cargo.exists():
        return
    dependency_lines = _toml_dependency_lines(example.read_text(errors="replace"))
    if not dependency_lines:
        return
    text = cargo.read_text(errors="replace")
    existing = _toml_dependency_names(_toml_dependency_lines(text))
    additions = [line for line in dependency_lines if _toml_key(line) not in existing]
    if not additions:
        return
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "[dependencies]"), None)
    if start is None:
        new_text = text.rstrip() + "\n\n[dependencies]\n" + "\n".join(additions) + "\n"
        cargo.write_text(new_text)
        return
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    new_lines = lines[:end] + additions + lines[end:]
    cargo.write_text("\n".join(new_lines) + "\n")


def _toml_dependency_lines(text: str) -> list[str]:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "[dependencies]"), None)
    if start is None:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("["):
            break
        if stripped and not stripped.startswith("#"):
            out.append(line)
    return out


def _toml_dependency_names(lines: list[str]) -> set[str]:
    return {name for line in lines if (name := _toml_key(line))}


def _toml_key(line: str) -> str:
    return line.split("=", 1)[0].strip().strip('"') if "=" in line else ""


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


def _prepare_cpp_boost_date_time_shim(work: Path) -> None:
    cmake = work / "CMakeLists.txt"
    if not cmake.is_file() or "Boost::date_time" not in cmake.read_text(errors="replace"):
        return
    root = work / ".mcode-boost-shim"
    boost_dir = root / "include" / "boost"
    include_dir = boost_dir / "date_time" / "gregorian"
    lib_dir = root / "lib"
    cmake_dir = lib_dir / "cmake" / "Boost-1.74.0"
    include_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    cmake_dir.mkdir(parents=True, exist_ok=True)
    (boost_dir / "version.hpp").write_text(
        '#pragma once\n#define BOOST_VERSION 107400\n#define BOOST_LIB_VERSION "1_74"\n',
        encoding="utf-8",
    )
    (include_dir / "gregorian.hpp").write_text(_boost_gregorian_shim(), encoding="utf-8")
    (lib_dir / "libboost_date_time.a").write_bytes(b"!<arch>\n")
    (cmake_dir / "BoostConfig.cmake").write_text(
        f"""
set(Boost_FOUND TRUE)
set(Boost_VERSION 1.74.0)
set(Boost_INCLUDE_DIRS \"{root / "include"}\")
set(Boost_LIBRARIES Boost::date_time)
if(NOT TARGET Boost::date_time)
  add_library(Boost::date_time INTERFACE IMPORTED)
  target_include_directories(Boost::date_time INTERFACE \"{root / "include"}\")
endif()
""".lstrip(),
        encoding="utf-8",
    )
    (cmake_dir / "BoostConfigVersion.cmake").write_text(
        """
set(PACKAGE_VERSION 1.74.0)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
""".lstrip(),
        encoding="utf-8",
    )


def _boost_gregorian_shim() -> str:
    return (
        importlib.resources.files("mcode.bench.resources")
        .joinpath("boost_gregorian_shim.hpp")
        .read_text(encoding="utf-8")
    )


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
    _prepare_cpp_boost_date_time_shim(work)
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
