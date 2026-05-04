from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from mellea.backends.tools import MelleaTool
from mellea.stdlib.requirements import (
    Requirement,
    simple_validate,
    tool_arg_validator,
    uses_tool,
)

from mcode.agent.tooling import execute_command, format_tool_result, is_tool_result


@dataclass(frozen=True)
class VerificationPolicy:
    test_cmds: list[str]
    test_fn: Callable[[str], str] | None
    command_fn: Callable[[str], str] | None
    allow_default_test_cmd: bool
    prompt_block: str


@dataclass
class VerificationProgress:
    edit_revision: int = 0
    last_failed_run: tuple[int, str, str] | None = None

    def note_edit_applied(self) -> None:
        self.edit_revision += 1
        self.last_failed_run = None

    def repeated_failed_run(self, test_cmd: str) -> str | None:
        if self.last_failed_run is None:
            return None
        revision, previous_cmd, previous_result = self.last_failed_run
        if revision == self.edit_revision and previous_cmd == _normalize_test_cmd_key(test_cmd):
            return previous_result
        return None

    def note_run_tests_result(self, test_cmd: str, result: str) -> None:
        if _tool_result_passed(result):
            self.last_failed_run = None
            return
        self.last_failed_run = (self.edit_revision, _normalize_test_cmd_key(test_cmd), result)


def normalize_verification_commands(source: object | None) -> list[str]:
    if source is None:
        return []
    if isinstance(source, dict):
        for key in ("test_cmds", "verification_cmds", "verification", "commands"):
            if key in source:
                return normalize_verification_commands(source[key])
        return []
    if isinstance(source, str):
        text = source.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return [text]
        return normalize_verification_commands(parsed)
    if isinstance(source, (list, tuple, set)):
        return [text for item in source if (text := str(item).strip())]
    text = str(source).strip()
    return [text] if text else []


def build_verification_prompt(test_cmds: list[str], *, allow_default: bool) -> str:
    if test_cmds:
        formatted = ", ".join(f"`{cmd}`" for cmd in test_cmds[:3])
        more = "" if len(test_cmds) <= 3 else f" and {len(test_cmds) - 3} more"
        return (
            "\n\nVerification:\n"
            "Use `run_tests` before `final_answer`. Start by calling `run_tests` with "
            f'`test_cmd="default"` to execute the task checks ({formatted}{more}). '
            "Do not pass `run_tests default` as the argument value and do not join commands "
            "with `&&`; use `default` for the declared command sequence. Keep the "
            "`final_answer` text short."
        )
    if allow_default:
        return (
            "\n\nVerification:\n"
            "Use `run_tests` before `final_answer`. Pass only the command text in `test_cmd`; "
            'use `test_cmd="default"` when a default verifier exists. Keep the `final_answer` '
            "text short."
        )
    return (
        "\n\nVerification:\n"
        "Use `run_tests` before `final_answer`. There is no default test command for "
        "this task, so pass a concrete existing test or smoke command in `test_cmd`. "
        "Do not use `default` unless the prompt lists declared task checks. Keep the "
        "`final_answer` text short."
    )


def build_verification_policy(
    *,
    test_cmds: object | None = None,
    test_fn: Callable[[str], str] | None = None,
    command_fn: Callable[[str], str] | None = None,
) -> VerificationPolicy:
    verification_cmds = normalize_verification_commands(test_cmds)
    allow_default = bool(verification_cmds or test_fn is not None)
    return VerificationPolicy(
        test_cmds=verification_cmds,
        test_fn=test_fn,
        command_fn=command_fn,
        allow_default_test_cmd=allow_default,
        prompt_block=build_verification_prompt(verification_cmds, allow_default=allow_default),
    )


def build_turn_requirements(
    *,
    verification_policy: VerificationPolicy,
    enforce_run_tests: bool,
) -> list[object]:
    if not enforce_run_tests:
        return []

    return [
        uses_tool("run_tests"),
        tool_arg_validator(
            "Set `run_tests.test_cmd` to an allowed verification command.",
            "run_tests",
            "test_cmd",
            lambda value: _valid_test_command(
                value,
                verification_policy.test_cmds,
                allow_default=verification_policy.allow_default_test_cmd,
            ),
        ),
    ]


def build_submission_requirements() -> list[object]:
    return [
        Requirement(
            "Return a concise structured submission.",
            validation_fn=simple_validate(
                lambda text: _valid_submission_text(text),
                reason="Return a concise structured submission.",
            ),
        )
    ]


def _valid_test_command(value: object, allowed_commands: list[str], *, allow_default: bool) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if text.lower() == "default":
        return allow_default
    if not allowed_commands:
        return True
    return text in allowed_commands


def _valid_submission_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 4000:
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    summary = str(parsed.get("summary", "")).strip()
    tests_ran = parsed.get("tests_ran", [])
    return bool(summary) and isinstance(tests_ran, list)


def build_run_tests_tool(
    *,
    repo_root: str,
    verification_policy: VerificationPolicy,
    progress: VerificationProgress | None = None,
):
    test_cmds = verification_policy.test_cmds
    test_fn = verification_policy.test_fn
    command_fn = verification_policy.command_fn
    if not test_cmds and test_fn is None and command_fn is None:
        return None

    def _run_tests(
        test_cmd: str = "default",
        timeout_s: int = 120,
        max_output_chars: int = 4000,
    ) -> str:
        if progress is not None and (previous_result := progress.repeated_failed_run(test_cmd)):
            previous_status = _tool_result_status(previous_result) or "FAILED"
            return format_tool_result(
                f"run_tests {test_cmd}",
                "SKIPPED",
                f"Previous run_tests already returned {previous_status} with no edit since then. "
                "Edit the code before rerunning the same tests.",
            )

        def _record(result: str) -> str:
            if progress is not None:
                progress.note_run_tests_result(test_cmd, result)
            return result

        if test_fn is not None:
            result = test_fn(test_cmd)
            if is_tool_result(result):
                return _record(result)
            return _record(format_tool_result(test_cmd, "COMPLETED", result))

        if test_cmd.strip().lower() == "default" and not test_cmds:
            return _record(
                format_tool_result(
                    test_cmd,
                    "BLOCKED",
                    "No default test command is declared. Pass a concrete test command instead.",
                )
            )
        if command_fn is not None and test_cmds and test_cmd.strip().lower() != "default":
            return _record(
                format_tool_result(
                    test_cmd,
                    "BLOCKED",
                    "Declared task checks exist for this task. Call run_tests with "
                    '`test_cmd="default"` instead of inventing alternate verification commands.',
                )
            )
        commands = test_cmds if test_cmd.strip().lower() == "default" else [test_cmd]
        outputs: list[str] = []
        for command in commands:
            if not command.strip():
                continue
            if block_reason := _blocked_verification_command_reason(command):
                outputs.append(format_tool_result(command, "BLOCKED", block_reason))
                continue
            if command_fn is not None:
                result = command_fn(command)
                if is_tool_result(result):
                    outputs.append(
                        _append_failure_artifacts(
                            result,
                            repo_root=repo_root,
                            max_output_chars=max_output_chars,
                        )
                    )
                else:
                    outputs.append(format_tool_result(command, "COMPLETED", result))
                continue
            status, output = execute_command(
                command,
                repo_root=repo_root,
                timeout=timeout_s,
            )
            result = format_tool_result(command, status, output)
            result = _append_failure_artifacts(
                result, repo_root=repo_root, max_output_chars=max_output_chars
            )
            outputs.append(_truncate_tool_result(result, max_output_chars))
        if outputs:
            return _record("\n---\n".join(outputs))
        return _record(format_tool_result(test_cmd, "SKIPPED", "No test commands available."))

    tool = MelleaTool.from_callable(_run_tests, name="run_tests")
    return MelleaTool(
        name=tool.name,
        tool_call=_run_tests,
        as_json_tool=_patch_run_tests_schema(tool.as_json_tool),
    )


def _patch_run_tests_schema(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    patched = deepcopy(schema)
    function_schema = patched.get("function")
    if not isinstance(function_schema, dict):
        return patched
    parameters = function_schema.get("parameters")
    if not isinstance(parameters, dict):
        return patched
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        if isinstance(properties.get("timeout_s"), dict):
            properties["timeout_s"]["type"] = "integer"
        if isinstance(properties.get("max_output_chars"), dict):
            properties["max_output_chars"]["type"] = "integer"
    parameters["required"] = ["test_cmd"]
    return patched


def _blocked_verification_command_reason(command: str) -> str | None:
    normalized = f" {command.lower()} "
    evasive_markers = (
        " -k not ",
        " -k 'not ",
        ' -k "not ',
        " --ignore=",
        " --ignore ",
        " || true",
        " | head",
        " | tail",
        " | grep",
    )
    if any(marker in normalized for marker in evasive_markers):
        return (
            "Verification commands must run the relevant failing tests without skipping, "
            "excluding, or masking their exit status. Run the direct failing test command "
            "instead."
        )
    return None


def _append_failure_artifacts(
    tool_result: str,
    *,
    repo_root: str,
    max_output_chars: int,
) -> str:
    if _tool_result_passed(tool_result):
        return _truncate_tool_result(tool_result, max_output_chars)
    artifact_text = _collect_failure_artifacts(Path(repo_root), tool_result)
    if not artifact_text:
        return _truncate_tool_result(tool_result, max_output_chars)
    combined = f"{tool_result.rstrip()}\n\nFailure report snippets:\n{artifact_text}"
    return _truncate_tool_result(combined, max_output_chars)


def _normalize_test_cmd_key(test_cmd: str) -> str:
    return test_cmd.strip().lower() or "default"


def _tool_result_status(tool_result: str) -> str | None:
    lines = tool_result.splitlines()
    if len(lines) < 2:
        return None
    status = lines[1].strip()
    return status or None


def _tool_result_passed(tool_result: str) -> bool:
    lines = tool_result.splitlines()
    if len(lines) < 2:
        return False
    return lines[1].strip() == "PASSED"


def _truncate_tool_result(value: str, max_output_chars: int) -> str:
    if len(value) <= max_output_chars:
        return value
    lines = value.splitlines()
    header = "\n".join(lines[:2]).strip()
    marker = "\n...[tool output truncated, keeping final diagnostics]...\n"
    keep = max_output_chars - len(header) - len(marker) - 1
    if keep <= 0:
        return value[:max_output_chars]
    return f"{header}{marker}{value[-keep:]}"


def _collect_failure_artifacts(repo_root: Path, output: str = "") -> str:
    if not repo_root.is_dir():
        return ""
    candidates = (
        [] if _looks_like_compile_failure(output) else _failure_artifact_candidates(repo_root)
    )
    failed_cases: list[tuple[str, str, str]] = []
    for path in candidates[:4]:
        if path.suffix.lower() == ".xml":
            failed_cases.extend(_failed_test_cases(path))
    snippets: list[str] = []
    source_blocks = [
        block
        for block in (
            _failure_source_snippets(repo_root, failed_cases),
            _failure_source_snippets_from_test_locations(repo_root, output),
            _go_failed_subtest_source_snippets(repo_root, output),
            _failure_source_snippets_from_output(repo_root, output),
        )
        if block
    ]
    if source_blocks:
        source_text = "\n\n".join(source_blocks)
        snippets.append(f"Failing test source snippets:\n{source_text}")
    for path in candidates[:4]:
        snippet = _failure_artifact_snippet(path)
        if snippet:
            rel = path.relative_to(repo_root).as_posix()
            snippets.append(f"{rel}:\n{snippet}")
    return "\n\n".join(snippets)


def _looks_like_compile_failure(output: str) -> bool:
    return bool(
        re.search(
            r"\b(?:compilation failed|compilejava|compileerror|could not compile|error\[E\d+\])\b",
            output,
            re.IGNORECASE,
        )
    )


def _failure_artifact_candidates(repo_root: Path) -> list[Path]:
    patterns = (
        "build/test-results/test/*.xml",
        "target/surefire-reports/*.txt",
        "target/surefire-reports/*.xml",
        "test-results/**/*.xml",
        "reports/**/*.xml",
        "build/reports/tests/test/classes/*.html",
    )
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


def _failed_test_cases(path: Path) -> list[tuple[str, str, str]]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    cases: list[tuple[str, str, str]] = []
    for match in re.finditer(r"<testcase\b([^>]*)>(.*?)</testcase>", text, re.DOTALL):
        body = match.group(2)
        if "<failure" not in body and "<error" not in body:
            continue
        attrs = match.group(1)
        name = _xml_attr(attrs, "name")
        classname = _xml_attr(attrs, "classname")
        message = _xml_attr(body, "message")
        cases.append((classname, name, message))
    return cases


def _failure_source_snippets(repo_root: Path, failed_cases: list[tuple[str, str, str]]) -> str:
    snippets: list[str] = []
    seen: set[tuple[Path, str]] = set()
    for classname, test_name, _message in failed_cases[:4]:
        path = _find_test_source(repo_root, classname)
        method = _test_method_name(test_name)
        if path is None or not method or (path, method) in seen:
            continue
        seen.add((path, method))
        snippet = _source_method_snippet(path, method)
        if snippet:
            rel = path.relative_to(repo_root).as_posix()
            snippets.append(f"{rel}::{method}\n{snippet}")
    return "\n\n".join(snippets)


def _failure_source_snippets_from_test_locations(repo_root: Path, output: str) -> str:
    snippets: list[str] = []
    seen: set[tuple[Path, int]] = set()
    for rel_path, line_number in _failed_test_locations_from_output(output)[:4]:
        path = repo_root / rel_path
        if not path.is_file() or (path, line_number) in seen:
            continue
        seen.add((path, line_number))
        snippet = _source_block_snippet(path, max(0, line_number - 1))
        if snippet:
            snippets.append(f"{rel_path}:{line_number}\n{snippet}")
    return "\n\n".join(snippets)


def _failed_test_locations_from_output(output: str) -> list[tuple[str, int]]:
    locations: list[tuple[str, int]] = []
    for line in output.splitlines():
        match = re.match(
            r"\s*(?P<path>[\w./\\-]*(?:test|spec)[\w./\\-]*\.[A-Za-z0-9_]+):(?P<line>\d+)(?::\d+)?:",
            line,
        )
        if not match:
            continue
        path = match.group("path").replace("\\", "/")
        try:
            line_number = int(match.group("line"))
        except ValueError:
            continue
        locations.append((path, line_number))
    return locations


def _go_failed_subtest_source_snippets(repo_root: Path, output: str) -> str:
    names = _go_failed_subtest_names(output)
    if not names:
        return ""
    snippets: list[str] = []
    seen: set[tuple[Path, int]] = set()
    wanted = {_normalized_go_test_name(name): name for name in names[:4]}
    for path in _test_source_candidates(repo_root):
        if path.suffix != ".go" or "test" not in path.name.lower():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines):
            for literal in re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', line):
                normalized = _normalized_go_test_name(literal)
                if normalized not in wanted or (path, index) in seen:
                    continue
                seen.add((path, index))
                snippet = _source_block_from_lines(lines, index, max_lines=30)
                if snippet:
                    rel = path.relative_to(repo_root).as_posix()
                    snippets.append(f"{rel}::{wanted[normalized]}\n{snippet}")
                if len(snippets) >= 4:
                    return "\n\n".join(snippets)
    return "\n\n".join(snippets)


def _go_failed_subtest_names(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        match = re.match(r"\s*--- FAIL: [^\s/]+/(?P<name>[^\s(]+)", line)
        if match:
            names.append(match.group("name").strip())
    return names


def _normalized_go_test_name(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip())


def _failure_source_snippets_from_output(repo_root: Path, output: str) -> str:
    failed_names = _failed_test_names_from_output(output)
    if not failed_names:
        return ""
    snippets: list[str] = []
    seen: set[tuple[Path, int]] = set()
    for name in failed_names[:4]:
        for path, index in _find_named_test_sources(repo_root, name):
            if (path, index) in seen:
                continue
            seen.add((path, index))
            snippet = _source_block_snippet(path, index)
            if snippet:
                rel = path.relative_to(repo_root).as_posix()
                snippets.append(f"{rel}::{name}\n{snippet}")
            if len(snippets) >= 4:
                return "\n\n".join(snippets)
    return "\n\n".join(snippets)


def _failed_test_names_from_output(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        match = re.match(r"\s*[✕×x]\s+(.+?)(?:\s+\(\d+\s*ms\))?\s*$", line)
        if match:
            names.append(match.group(1).strip())
    return names


def _find_named_test_sources(repo_root: Path, test_name: str) -> list[tuple[Path, int]]:
    matches: list[tuple[Path, int]] = []
    escaped = re.escape(test_name)
    patterns = (
        re.compile(rf"\b(?:test|it)\s*\(\s*['\"]{escaped}['\"]"),
        re.compile(rf"\b(?:xtest|xit)\s*\(\s*['\"]{escaped}['\"]"),
    )
    for path in _test_source_candidates(repo_root):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines):
            if any(pattern.search(line) for pattern in patterns):
                matches.append((path, index))
    return matches


def _test_source_candidates(repo_root: Path) -> list[Path]:
    suffixes = (".java", ".kt", ".py", ".js", ".ts", ".rs", ".go", ".cpp", ".cc", ".cxx")
    candidates = [
        path for path in repo_root.rglob("*") if path.is_file() and path.suffix in suffixes
    ]
    candidates.sort(
        key=lambda path: (
            "test" not in path.name.lower() and "spec" not in path.name.lower(),
            str(path),
        )
    )
    return candidates


def _find_test_source(repo_root: Path, classname: str) -> Path | None:
    simple = classname.rsplit(".", 1)[-1].split("$", 1)[0].strip()
    if not simple:
        return None
    suffixes = (".java", ".kt", ".py", ".js", ".ts", ".rs", ".go", ".cpp", ".cc", ".cxx")
    expected = {f"{simple}{suffix}" for suffix in suffixes}
    for path in repo_root.rglob("*"):
        if path.is_file() and path.name in expected:
            return path
    return None


def _test_method_name(test_name: str) -> str:
    return test_name.split("(", 1)[0].strip()


def _source_method_snippet(path: Path, method: str, *, max_lines: int = 45) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    target = re.compile(rf"\b{re.escape(method)}\b")
    index = next((i for i, line in enumerate(lines) if target.search(line)), None)
    if index is None:
        return ""
    return _source_block_from_lines(lines, index, max_lines=max_lines)


def _source_block_snippet(path: Path, index: int, *, max_lines: int = 45) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    clamped_index = min(max(index, 0), len(lines) - 1)
    return _source_block_from_lines(lines, clamped_index, max_lines=max_lines)


def _source_block_from_lines(lines: list[str], index: int, *, max_lines: int) -> str:
    start = _enclosing_test_block_start(lines, index)
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1
    end = _method_end(lines, index, max_lines=max_lines)
    compact = []
    for i in range(start, end):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("//"):
            compact.append(f"{i + 1}: {stripped}")
    return "\n".join(compact)[:1800]


def _enclosing_test_block_start(lines: list[str], index: int) -> int:
    for i in range(min(index, len(lines) - 1), -1, -1):
        stripped = lines[i].strip()
        if re.match(r"(?:def|func)\s+test[_A-Za-z0-9]*\b", stripped):
            return i
        if re.match(r"(?:public\s+)?void\s+\w+\s*\(", stripped):
            return i
        if re.match(r"TEST_CASE\s*\(", stripped):
            return i
    return index


def _method_end(lines: list[str], index: int, *, max_lines: int) -> int:
    balance = 0
    seen_open = False
    for i in range(index, min(len(lines), index + max_lines)):
        balance += lines[i].count("{") - lines[i].count("}")
        seen_open = seen_open or "{" in lines[i]
        if seen_open and balance <= 0:
            return i + 1
    return min(len(lines), index + max_lines)


def _failure_artifact_snippet(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    if path.suffix.lower() == ".xml":
        failed_cases = []
        for match in re.finditer(r"<testcase\b([^>]*)>(.*?)</testcase>", text, re.DOTALL):
            body = match.group(2)
            if "<failure" not in body and "<error" not in body:
                continue
            attrs = match.group(1)
            name = _xml_attr(attrs, "name")
            classname = _xml_attr(attrs, "classname")
            message = _xml_attr(body, "message")
            failed_cases.append(" ".join(p for p in (classname, name, message) if p))
        if failed_cases:
            text = "\n".join(failed_cases)
        else:
            attributes = re.findall(r'\b(?:message|type|name|classname)="([^"]+)"', text)
            text = "\n".join(attributes + [re.sub(r"<[^>]+>", " ", text)])
        text = html.unescape(text)
    elif path.suffix.lower() == ".html":
        attributes = re.findall(r'\b(?:message|type|name|classname)="([^"]+)"', text)
        text = "\n".join(attributes + [re.sub(r"<[^>]+>", " ", text)])
        text = html.unescape(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if re.search(
            r"fail|error|expected|actual|assert|exception|traceback",
            line,
            re.IGNORECASE,
        )
    ]
    selected = interesting or lines
    snippet = "\n".join(selected[:30])
    return snippet[:2000]


def _xml_attr(text: str, name: str) -> str:
    match = re.search(rf'\b{re.escape(name)}="([^"]*)"', text)
    return html.unescape(match.group(1)) if match else ""
