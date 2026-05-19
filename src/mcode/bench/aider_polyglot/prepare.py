from __future__ import annotations

import importlib.resources
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mcode.util import temporary_directory

from .constants import JAVA_DISABLED_RE, JS_SKIP_MARKER_RE
from .models import AiderPolyglotTask, PreparedPolyglotTask


@dataclass(frozen=True)
class _LanguageDescriptor:
    practice_dir: Path
    prepare: Callable[[Path, Path], tuple[list[str], list[str]]]
    test_commands: tuple[str, ...]
    timeout_s: int


def prepare_task(
    task: AiderPolyglotTask,
    *,
    benchmark_root: str | Path | None = None,
) -> PreparedPolyglotTask:
    from .dataset import ensure_benchmark_root

    root = ensure_benchmark_root(benchmark_root)
    descriptors = build_language_descriptors(root)
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
        new_text = JS_SKIP_MARKER_RE.sub(
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
            new_text = JAVA_DISABLED_RE.sub("", text)
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


def build_language_descriptors(root: Path) -> dict[str, _LanguageDescriptor]:
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
