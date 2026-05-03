from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolchainCheck:
    language: str
    name: str
    ok: bool
    detail: str
    next: str


class PolyglotToolchainError(RuntimeError):
    pass


_LANGUAGES = ("python", "go", "rust", "javascript", "cpp", "java")


def supported_polyglot_toolchain_languages() -> tuple[str, ...]:
    return _LANGUAGES


def normalize_polyglot_languages(language: str | Sequence[str] | None) -> tuple[str, ...]:
    raw = [language] if isinstance(language, str) or language is None else list(language)
    values: list[str] = []
    for item in raw:
        name = (item or "all").strip().lower()
        if name == "all":
            return _LANGUAGES
        if name not in _LANGUAGES:
            known = ", ".join((*_LANGUAGES, "all"))
            raise ValueError(f"unknown language {name!r}; expected one of {known}")
        values.append(name)
    return tuple(dict.fromkeys(values))


def check_polyglot_toolchains(
    language: str | Sequence[str] | None = "all",
    *,
    run_command: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[ToolchainCheck, ...]:
    runner = run_command or subprocess.run
    checks: list[ToolchainCheck] = []
    for name in normalize_polyglot_languages(language):
        checks.extend(_checks_for_language(name, runner))
    return tuple(checks)


def missing_polyglot_toolchains(
    language: str | Sequence[str] | None = "all",
    *,
    run_command: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[ToolchainCheck, ...]:
    return tuple(
        check
        for check in check_polyglot_toolchains(language, run_command=run_command)
        if not check.ok
    )


def ensure_polyglot_toolchains(language: str | Sequence[str] | None = "all") -> None:
    missing = missing_polyglot_toolchains(language)
    if missing:
        raise PolyglotToolchainError(format_toolchain_failure(missing))


def format_toolchain_failure(checks: Sequence[ToolchainCheck]) -> str:
    lines = ["polyglot toolchain unavailable"]
    for check in checks:
        lines.append(f"- {check.language}: {check.name}: {check.detail}")
        if check.next:
            lines.append(f"  next: {check.next}")
    install = install_hint(sorted({check.language for check in checks}))
    if install:
        lines.append(f"install: {install}")
    lines.append("or run: mcode deps toolchains --benchmark aider-polyglot --install")
    return "\n".join(lines)


def install_hint(languages: Sequence[str]) -> str:
    packages = _packages_for_languages(languages)
    if not packages:
        return ""
    system = platform.system().lower()
    if system == "darwin":
        return "brew install " + " ".join(_macos_packages(packages))
    if system == "windows":
        return "winget install " + " ".join(_windows_packages(packages))
    if shutil.which("apt-get"):
        return "sudo apt-get update && sudo apt-get install -y " + " ".join(_apt_packages(packages))
    if shutil.which("dnf"):
        return "sudo dnf install -y " + " ".join(_dnf_packages(packages))
    if shutil.which("pacman"):
        return "sudo pacman -S --needed " + " ".join(_pacman_packages(packages))
    return "install " + ", ".join(packages)


def install_polyglot_toolchains(
    language: str | Sequence[str] | None = "all",
    *,
    run_command: Callable[..., subprocess.CompletedProcess] | None = None,
) -> None:
    languages = normalize_polyglot_languages(language)
    packages = _packages_for_languages(languages)
    if not packages:
        return
    runner = run_command or subprocess.run
    commands = _install_commands(packages)
    if not commands:
        raise PolyglotToolchainError("no supported installer found; " + install_hint(languages))
    for command in commands:
        runner(command, check=True)


def _checks_for_language(
    language: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> tuple[ToolchainCheck, ...]:
    if language == "python":
        return (_python_module_check(language, "pytest", runner),)
    if language == "go":
        return (_command_check(language, "go", ["go", "version"], "install Go", runner),)
    if language == "rust":
        return (
            _command_check(language, "cargo", ["cargo", "--version"], "install Rust/Cargo", runner),
            _command_check(language, "rustc", ["rustc", "--version"], "install Rust/Cargo", runner),
        )
    if language == "javascript":
        return (
            _command_check(language, "node", ["node", "--version"], "install Node.js", runner),
            _command_check(language, "npm", ["npm", "--version"], "install npm", runner),
        )
    if language == "cpp":
        return (
            _command_check(language, "cmake", ["cmake", "--version"], "install CMake", runner),
            _cpp_compiler_check(language, runner),
        )
    if language == "java":
        return (
            _command_check(language, "java", ["java", "-version"], "install a JDK", runner),
            _command_check(language, "javac", ["javac", "-version"], "install a JDK", runner),
        )
    raise ValueError(f"unknown language {language!r}")


def _command_check(
    language: str,
    name: str,
    command: list[str],
    next_step: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> ToolchainCheck:
    binary = command[0]
    candidates = _command_candidates(command)
    if not candidates:
        return ToolchainCheck(language, name, False, f"{binary} not found on PATH", next_step)
    failed_detail = f"{binary} not found on PATH"
    for candidate in candidates:
        try:
            proc = runner(candidate, capture_output=True, text=True)
        except OSError as exc:
            failed_detail = str(exc)
            continue
        output = (proc.stdout or proc.stderr or candidate[0]).strip().splitlines()
        detail = output[0] if output else candidate[0]
        if proc.returncode == 0:
            return ToolchainCheck(language, name, True, detail, "")
        failed_detail = detail
    return ToolchainCheck(language, name, False, failed_detail, next_step)


def _command_candidates(command: list[str]) -> list[list[str]]:
    binary = command[0]
    path = shutil.which(binary)
    candidates = [[path, *command[1:]]] if path else []
    if platform.system().lower() == "darwin" and binary in {"java", "javac"}:
        for root in ("/opt/homebrew/opt/openjdk/bin", "/usr/local/opt/openjdk/bin"):
            candidate = Path(root) / binary
            if candidate.exists():
                candidates.append([str(candidate), *command[1:]])
    return candidates


def _python_module_check(
    language: str,
    module: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> ToolchainCheck:
    proc = runner(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return ToolchainCheck(language, module, True, f"{module} import ok", "")
    return ToolchainCheck(
        language,
        module,
        False,
        f"python module {module!r} not importable",
        "uv run mcode deps sync",
    )


def _cpp_compiler_check(
    language: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> ToolchainCheck:
    for binary in ("c++", "g++", "clang++"):
        path = shutil.which(binary)
        if not path:
            continue
        check = _command_check(
            language, "c++", [binary, "--version"], "install a C++ compiler", runner
        )
        if check.ok:
            return check
    return ToolchainCheck(
        language,
        "c++",
        False,
        "C++ compiler not found on PATH",
        "install a C++ compiler",
    )


def _packages_for_languages(languages: Sequence[str]) -> tuple[str, ...]:
    packages: list[str] = []
    for language in languages:
        if language == "go":
            packages.append("go")
        elif language == "rust":
            packages.append("rust")
        elif language == "javascript":
            packages.append("node")
        elif language == "cpp":
            packages.extend(["cmake", "c++"])
        elif language == "java":
            packages.append("java")
    return tuple(dict.fromkeys(packages))


def _install_commands(packages: Sequence[str]) -> list[list[str]]:
    system = platform.system().lower()
    if system == "darwin" and shutil.which("brew"):
        return [["brew", "install", *_macos_packages(packages)]]
    if system == "windows" and shutil.which("winget"):
        commands: list[list[str]] = []
        for package in _windows_packages(packages):
            commands.append(
                [
                    "winget",
                    "install",
                    "--id",
                    package,
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ]
            )
        return commands
    if system == "windows" and shutil.which("choco"):
        return [["choco", "install", "-y", *_choco_packages(packages)]]
    if shutil.which("apt-get"):
        return [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", *_apt_packages(packages)],
        ]
    if shutil.which("dnf"):
        return [["sudo", "dnf", "install", "-y", *_dnf_packages(packages)]]
    if shutil.which("pacman"):
        return [["sudo", "pacman", "-S", "--needed", *_pacman_packages(packages)]]
    return []


def _macos_packages(packages: Sequence[str]) -> list[str]:
    mapping = {"rust": "rust", "node": "node", "java": "openjdk", "c++": "llvm"}
    return [mapping.get(package, package) for package in packages]


def _windows_packages(packages: Sequence[str]) -> list[str]:
    mapping = {
        "go": "GoLang.Go",
        "rust": "Rustlang.Rustup",
        "node": "OpenJS.NodeJS",
        "cmake": "Kitware.CMake",
        "java": "EclipseAdoptium.Temurin.21.JDK",
        "c++": "Microsoft.VisualStudio.2022.BuildTools",
    }
    return [mapping.get(package, package) for package in packages]


def _choco_packages(packages: Sequence[str]) -> list[str]:
    mapping = {
        "go": "golang",
        "rust": "rustup.install",
        "node": "nodejs",
        "java": "temurin21",
        "c++": "visualstudio2022buildtools",
    }
    return [mapping.get(package, package) for package in packages]


def _apt_packages(packages: Sequence[str]) -> list[str]:
    mapping = {"rust": "cargo", "node": "nodejs npm", "java": "openjdk-21-jdk", "c++": "g++"}
    expanded: list[str] = []
    for package in packages:
        expanded.extend(mapping.get(package, package).split())
    return expanded


def _dnf_packages(packages: Sequence[str]) -> list[str]:
    mapping = {
        "rust": "cargo rust",
        "node": "nodejs npm",
        "java": "java-21-openjdk-devel",
        "c++": "gcc-c++",
    }
    expanded: list[str] = []
    for package in packages:
        expanded.extend(mapping.get(package, package).split())
    return expanded


def _pacman_packages(packages: Sequence[str]) -> list[str]:
    mapping = {"rust": "rust", "node": "nodejs npm", "java": "jdk-openjdk", "c++": "gcc"}
    expanded: list[str] = []
    for package in packages:
        expanded.extend(mapping.get(package, package).split())
    return expanded
