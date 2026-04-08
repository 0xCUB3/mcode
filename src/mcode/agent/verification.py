from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationPolicy:
    test_cmds: list[str]
    test_fn: Callable[[str], str] | None
    prompt_block: str

    def tool_kwargs(self) -> dict[str, object]:
        return {
            "test_cmds": self.test_cmds,
            "test_fn": self.test_fn,
        }


@dataclass(frozen=True)
class VerificationState:
    used_run_tests: bool = False
    successful_run_tests: bool = False
    used_default_run_tests: bool = False
    successful_default_run_tests: bool = False
    successful_submit_eligible_run_tests: bool = False
    successful_submit_eligible_after_edit: bool = False
    post_edit_probe_calls: int = 0
    post_edit_run_tests_calls: int = 0
    blocked_verification_commands: int = 0


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
        cmds: list[str] = []
        for item in source:
            text = str(item).strip()
            if text:
                cmds.append(text)
        return cmds

    text = str(source).strip()
    return [text] if text else []


_VERIFICATION_DISCOVERY_TOOLS = frozenset({"search_code", "read_file", "find_file", "list_dir"})
_FOCUSED_DISCOVERY_TOOLS = frozenset({"read_file"})
_DISALLOWED_VERIFICATION_PREFIXES = frozenset(
    {"bash", "sh", "zsh", "fish", "cd", "source", "conda", "export"}
)
_ALLOWED_VERIFICATION_PREFIXES = frozenset(
    {
        "pytest",
        "python",
        "python3",
        "uv",
        "tox",
        "nox",
        "cargo",
        "go",
        "npm",
        "pnpm",
        "yarn",
        "make",
    }
)


def classify_verification_command(command: str) -> str:
    text = command.strip()
    if not text:
        return "unknown"
    try:
        tokens = shlex.split(text)
    except ValueError:
        return "unknown"
    if not tokens:
        return "unknown"

    launcher = tokens[0].split("/")[-1].lower()
    if launcher in {"pytest", "tox", "nox", "cargo", "go", "npm", "pnpm", "yarn", "make"}:
        return "runner"
    if launcher == "uv":
        return "runner"
    if launcher in {"python", "python3"}:
        if len(tokens) > 1 and tokens[1] == "-c":
            return "probe"
        if len(tokens) > 1 and tokens[1] == "-m":
            return "runner"
        return "script"
    return "unknown"


def _default_verification_is_submit_eligible(test_cmds: list[str]) -> bool:
    return any(classify_verification_command(command) == "runner" for command in test_cmds)


def _contains_unquoted_shell_control(command: str) -> bool:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if char in {"|", ";", ">", "<"}:
            return True
        if char == "&" and index + 1 < len(command) and command[index + 1] == "&":
            return True
    return False


def validate_verification_command(command: str) -> str | None:
    text = command.strip()
    if not text:
        return "Run a plain verification command. The current command is empty."
    if _contains_unquoted_shell_control(text):
        return (
            "Run a plain verification command inside `run_tests`. Do not use pipes, "
            "redirection, or command chaining."
        )
    try:
        tokens = shlex.split(text)
    except ValueError:
        return (
            "Run a plain verification command inside `run_tests`. The current "
            "command could not be parsed safely."
        )
    if not tokens:
        return "Run a plain verification command. The current command is empty."

    launcher = tokens[0].split("/")[-1].lower()
    if launcher in _DISALLOWED_VERIFICATION_PREFIXES:
        return (
            "Use `run_tests` with the test command itself, not shell setup like `cd`, "
            "`conda`, `source`, or `bash`."
        )
    if launcher not in _ALLOWED_VERIFICATION_PREFIXES:
        return (
            "Use `run_tests` with a direct test command such as `pytest`, `python -m "
            "pytest`, `python -m unittest`, `cargo test`, or `npm test`."
        )
    return None


def build_verification_prompt(test_cmds: list[str]) -> str:
    if test_cmds:
        formatted = ", ".join(f"`{cmd}`" for cmd in test_cmds[:3])
        more = "" if len(test_cmds) <= 3 else f" and {len(test_cmds) - 3} more"
        return (
            "\n\nVerification:\n"
            "You have a `run_tests` tool. Start with `run_tests default` "
            f"to run the task-default checks ({formatted}{more}). Keep "
            "verification cheap, if you need a narrower check, use a "
            "targeted command instead of a broad suite. Use `probe_python` for tiny Python "
            "repros instead of writing a scratch script. Pass a plain test command only, do "
            "not wrap it in `cd`, `conda`, `source`, pipes, redirection, or `bash`. Do not "
            "run tests through `bash` when `run_tests` is available. `python -c` style probes "
            "can guide debugging, but they do not unlock final submission. Do not call "
            "`final_answer` after editing until you have attempted verification with "
            "`run_tests`."
        )

    return (
        "\n\nVerification:\n"
        "You still have a `run_tests` tool even though there are no "
        "task-default commands. Use it with the cheapest plain command "
        "that exercises your change, such as `pytest -q path/to/test.py -k "
        "name` or `python -m pytest -q path/to/test.py -k name`. Use "
        "`probe_python` for tiny Python repros. Do not wrap test commands in `cd`, `conda`, "
        "`source`, pipes, redirection, or `bash`. Avoid full-suite runs unless necessary, "
        "and do not run tests through `bash` when `run_tests` is available. `python -c` "
        "style probes can guide debugging, but they do not unlock final submission. Do not "
        "call `final_answer` after editing until you have attempted verification with "
        "`run_tests`."
    )


def build_budget_warning(
    *,
    has_changes: bool,
    has_run_tests_tool: bool,
    used_run_tests: bool,
) -> str:
    if not has_changes:
        return (
            "WARNING: You have 2 turns left and your working tree "
            "still has no code changes. Make at least one edit now. "
            "Do not call `final_answer` yet."
        )

    if has_run_tests_tool and not used_run_tests:
        return (
            "WARNING: You have 2 turns left and you have not run "
            "verification yet. Use `run_tests default` now, or a "
            "narrower `run_tests <command>` if needed. Do not call "
            "`final_answer` yet."
        )

    return (
        "WARNING: You have 2 turns left. If your edit is already "
        "verified, call `final_answer` now. If verification is still "
        "failing or missing, do not call `final_answer` yet."
    )


def verification_state_from_event_log(
    event_log: object | None,
    *,
    default_test_cmds: list[str] | None = None,
) -> VerificationState:
    if event_log is None:
        return VerificationState()

    to_dicts = getattr(event_log, "to_dicts", None)
    if not callable(to_dicts):
        return VerificationState()

    pending_test_cmd: str | None = None
    pending_submit_eligible = False
    pending_edit_generation = 0
    used_run_tests = False
    successful_run_tests = False
    used_default_run_tests = False
    successful_default_run_tests = False
    successful_submit_eligible_run_tests = False
    successful_submit_eligible_after_edit = False
    post_edit_probe_calls = 0
    post_edit_run_tests_calls = 0
    blocked_verification_commands = 0
    edit_generation = 0

    for event in to_dicts():
        if not isinstance(event, dict):
            continue
        if event.get("kind") == "summary":
            metadata = event.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("kind") == "edit_started":
                edit_generation += 1
                post_edit_probe_calls = 0
                post_edit_run_tests_calls = 0
            continue
        if event.get("tool_name") == "probe_python" and event.get("kind") == "tool_call":
            if edit_generation > 0:
                post_edit_probe_calls += 1
            continue
        if event.get("tool_name") != "run_tests":
            continue

        if event.get("kind") == "tool_call":
            arguments = event.get("arguments", {})
            if isinstance(arguments, dict):
                raw_test_cmd = str(arguments.get("test_cmd", "")).strip()
                pending_test_cmd = raw_test_cmd.lower()
                used_run_tests = True
                if edit_generation > 0:
                    post_edit_run_tests_calls += 1
                if pending_test_cmd == "default":
                    used_default_run_tests = True
                pending_submit_eligible = pending_test_cmd == "default" and (
                    _default_verification_is_submit_eligible(default_test_cmds or [])
                )
                pending_edit_generation = edit_generation
                if raw_test_cmd and pending_test_cmd != "default":
                    pending_submit_eligible = (
                        classify_verification_command(raw_test_cmd) == "runner"
                    )
            continue

        if event.get("kind") != "tool_result":
            continue

        output = str(event.get("output", ""))
        if "\nBLOCKED\n" in output:
            blocked_verification_commands += 1
            pending_test_cmd = None
            pending_submit_eligible = False
            pending_edit_generation = 0
            continue

        success = (
            "PASSED" in output
            and "FAILED" not in output
            and "TIMEOUT" not in output
            and "ERROR" not in output
        )
        if success:
            successful_run_tests = True
            if pending_test_cmd == "default":
                successful_default_run_tests = True
            if pending_submit_eligible:
                successful_submit_eligible_run_tests = True
                if pending_edit_generation == edit_generation and edit_generation > 0:
                    successful_submit_eligible_after_edit = True
        pending_test_cmd = None
        pending_submit_eligible = False
        pending_edit_generation = 0

    return VerificationState(
        used_run_tests=used_run_tests,
        successful_run_tests=successful_run_tests,
        used_default_run_tests=used_default_run_tests,
        successful_default_run_tests=successful_default_run_tests,
        successful_submit_eligible_run_tests=successful_submit_eligible_run_tests,
        successful_submit_eligible_after_edit=successful_submit_eligible_after_edit,
        post_edit_probe_calls=post_edit_probe_calls,
        post_edit_run_tests_calls=post_edit_run_tests_calls,
        blocked_verification_commands=blocked_verification_commands,
    )


@dataclass(frozen=True)
class _LocalToolInvocation:
    name: str
    status: str = "completed"


@dataclass(frozen=True)
class _LocalToolPhaseState:
    turn: int
    budget: int
    invocations: tuple[_LocalToolInvocation, ...] = ()
    malformed_tool_calls: int = 0
    final_answer_blocks: int = 0

    @property
    def has_edit(self) -> bool:
        return any(call.name == "edit" for call in self.invocations)

    @property
    def progress(self) -> float:
        return self.turn / max(1, self.budget)

    @property
    def last_tool_name(self) -> str | None:
        if not self.invocations:
            return None
        return self.invocations[-1].name

    @property
    def repeated_tool_streak(self) -> int:
        if not self.invocations:
            return 0
        last_name = self.invocations[-1].name
        streak = 0
        for call in reversed(self.invocations):
            if call.name != last_name:
                break
            streak += 1
        return streak


def tool_phase_state_from_event_log(
    event_log: object | None,
    *,
    turn: int,
    budget: int,
):
    try:
        from mellea.agent.strategy import ToolInvocation, ToolPhaseState
    except ImportError:
        ToolInvocation = _LocalToolInvocation
        ToolPhaseState = _LocalToolPhaseState

    if event_log is None:
        return ToolPhaseState(turn=turn, budget=budget)

    to_dicts = getattr(event_log, "to_dicts", None)
    if not callable(to_dicts):
        return ToolPhaseState(turn=turn, budget=budget)

    invocations: list[ToolInvocation] = []
    malformed_tool_calls = 0
    final_answer_blocks = 0
    for event in to_dicts():
        if not isinstance(event, dict):
            continue
        if event.get("kind") == "summary":
            metadata = event.get("metadata", {})
            if isinstance(metadata, dict):
                summary_kind = metadata.get("kind")
                if summary_kind == "malformed_tool_call":
                    malformed_tool_calls += 1
                if summary_kind == "submission_blocked":
                    final_answer_blocks += 1
            continue
        if event.get("kind") != "tool_result":
            continue
        tool_name = event.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        status = str(event.get("status", "completed") or "completed")
        invocations.append(ToolInvocation(tool_name, status=status))

    return ToolPhaseState(
        turn=turn,
        budget=budget,
        invocations=tuple(invocations),
        malformed_tool_calls=malformed_tool_calls,
        final_answer_blocks=final_answer_blocks,
    )


def tighten_available_tools(
    available_tools: list[str],
    *,
    phase_state,
    has_changes: bool,
    verification_state: VerificationState,
    has_run_tests_tool: bool,
) -> list[str]:
    allowed = set(available_tools)
    discovery_tools = allowed & _VERIFICATION_DISCOVERY_TOOLS
    focused_discovery = allowed & _FOCUSED_DISCOVERY_TOOLS
    last_tool_name = getattr(phase_state, "last_tool_name", None)
    repeated_tool_streak = int(getattr(phase_state, "repeated_tool_streak", 0) or 0)

    if not has_changes and not phase_state.has_edit:
        if phase_state.progress >= 0.35:
            allowed.discard("bash")
        if repeated_tool_streak >= 2 and last_tool_name in discovery_tools:
            allowed -= discovery_tools - focused_discovery
            allowed.add("edit")
        if phase_state.progress >= 0.5:
            allowed -= discovery_tools - focused_discovery
            allowed.add("edit")
    elif has_run_tests_tool:
        if has_changes:
            allowed.discard("probe_python")
        if not verification_state.used_run_tests and phase_state.progress >= 0.5:
            allowed.discard("bash")
            allowed -= discovery_tools - focused_discovery
            allowed.add("run_tests")
        if has_changes and not verification_state.successful_run_tests:
            allowed.discard("bash")
            allowed -= discovery_tools - focused_discovery
            allowed.add("edit")
            allowed.add("run_tests")
        if (
            verification_state.used_run_tests
            and not verification_state.successful_run_tests
            and phase_state.progress >= 0.7
        ):
            allowed.discard("bash")
            allowed -= discovery_tools - focused_discovery
            allowed.add("edit")
            allowed.add("run_tests")

    return [tool_name for tool_name in available_tools if tool_name in allowed]


def build_phase_guidance(
    *,
    has_changes: bool,
    has_run_tests_tool: bool,
    verification_state: VerificationState,
    phase_state,
    require_default_verification: bool,
) -> str | None:
    if not has_changes and not phase_state.has_edit:
        if phase_state.progress >= 0.5:
            return (
                "Phase: edit now. You have spent enough turns diagnosing. Stop broad search, "
                "pick the likeliest file, and make one concrete edit this turn."
            )
        return (
            "Phase: diagnose, then edit. Stop widening the search. Pick the single most likely "
            "file and make one concrete edit this turn."
        )

    if not has_run_tests_tool or not has_changes:
        return None

    if verification_state.blocked_verification_commands:
        return (
            "Phase: verify. Run `run_tests` with a plain command only. Do not wrap it in "
            "`cd`, `conda`, `source`, pipes, redirection, or `bash`."
        )

    if require_default_verification and not verification_state.used_default_run_tests:
        return (
            "Phase: verify. Run `run_tests default` now. Do not keep widening the search or "
            "call `final_answer` before the task-default verification runs."
        )

    if has_changes:
        return (
            "Phase: verify. You already have a patch. Stop probing and run "
            "`run_tests` now with the cheapest plain command that exercises the edited path."
        )
    if not require_default_verification and not verification_state.used_run_tests:
        return (
            "Phase: verify. Run `run_tests` with the cheapest plain command that exercises "
            "your change before you keep editing or submit."
        )
    if not verification_state.successful_run_tests:
        return (
            "Phase: repair and re-verify. Your current verification has not passed yet. Keep "
            "iterating until `run_tests` succeeds."
        )
    if not verification_state.successful_submit_eligible_after_edit:
        return (
            "Phase: verify with a real test runner. Cheap probes can guide debugging, but "
            "before submit you still need a runner-style `run_tests` check after your latest edit."
        )

    return None


def build_tool_gate_message(
    tool_name: str,
    *,
    available_tools: list[str],
    has_changes: bool,
    has_run_tests_tool: bool,
    verification_state: VerificationState,
    require_default_verification: bool,
    requested_family: object | None = None,
    route_mode: object | None = None,
    requested_test_cmd: str | None = None,
) -> str | None:
    if tool_name == "final_answer":
        return build_submit_block_message(
            has_changes=has_changes,
            has_run_tests_tool=has_run_tests_tool,
            verification_state=verification_state,
            require_default_verification=require_default_verification,
        )

    if tool_name in available_tools:
        return None

    family_text = getattr(requested_family, "value", requested_family)
    route_text = getattr(route_mode, "value", route_mode)
    family_prefix = ""
    if isinstance(family_text, str) and family_text:
        family_prefix = f"Stay in the `{family_text}` capability family for this phase. "

    if tool_name == "bash":
        fallback_suffix = ""
        if route_text == "bundled_tool_fallback":
            fallback_suffix = (
                " This runtime is routing that family through the bundled tools, not an "
                "adapter-specific path."
            )
        return (
            family_prefix + "Bash is an escape hatch here. Use `read_file`, `search_code`, "
            "`find_file`, `list_dir`, `edit`, `probe_python`, and `run_tests` "
            "for the main loop unless you truly need a shell-only command." + fallback_suffix
        )
    if tool_name == "probe_python" and has_changes and has_run_tests_tool:
        return (
            family_prefix
            + "You already have a patch. `probe_python` is no longer available after edits. "
            "Run `run_tests` now with the cheapest plain runner command that exercises "
            "the edited path."
        )
    if tool_name == "run_tests" and has_changes and has_run_tests_tool:
        command_kind = classify_verification_command(requested_test_cmd or "")
        if command_kind in {"probe", "script"}:
            return (
                family_prefix
                + "You already have a patch. Post-edit `run_tests` probes must be runner-style. "
                "Use `pytest`, `python -m pytest`, or another real test runner command that "
                "exercises the edited path."
            )

    return None


def build_submit_block_message(
    *,
    has_changes: bool,
    has_run_tests_tool: bool,
    verification_state: VerificationState,
    require_default_verification: bool,
) -> str | None:
    if not has_changes:
        return (
            "You still have no code changes. Make at least one edit before calling `final_answer`."
        )

    if not has_run_tests_tool:
        return None

    if require_default_verification:
        if not verification_state.used_default_run_tests:
            return (
                "Before calling `final_answer`, run `run_tests default` and use that "
                "result to verify your patch."
            )
        if not verification_state.successful_default_run_tests:
            return (
                "Do not call `final_answer` yet. `run_tests default` has not passed. "
                "Fix the code until the task-default verification succeeds."
            )
        if verification_state.successful_submit_eligible_after_edit:
            return None
        return (
            "Before calling `final_answer`, run a runner-style `run_tests` check after your "
            "latest edit. Cheap probes like `python -c` do not unlock submission."
        )

    if not verification_state.used_run_tests:
        return (
            "Before calling `final_answer`, run `run_tests` with the cheapest command "
            "that exercises your change."
        )
    if not verification_state.successful_run_tests:
        return (
            "Do not call `final_answer` yet. Your verification has not passed. Fix the "
            "code until `run_tests` succeeds."
        )
    if not verification_state.successful_submit_eligible_after_edit:
        return (
            "Before calling `final_answer`, run a runner-style `run_tests` check after your "
            "latest edit. Cheap probes like `python -c` do not unlock submission."
        )
    return None


def build_run_tests_tool(
    *,
    repo_root: str,
    test_cmds: list[str],
    command_fn: Callable[[str], str] | None = None,
    workspace=None,
):
    import subprocess

    from mellea.agent.runtime.workspace import format_workspace_state
    from mellea.agent.tools.bash import execute_command, format_tool_result, is_tool_result
    from mellea.backends.tools import MelleaTool

    workspace_state = format_workspace_state(workspace)

    def _annotate(result: str) -> str:
        if workspace_state is None or workspace_state in result:
            return result
        return f"{result}\n\n{workspace_state}"

    def _run_tests(
        test_cmd: str = "default",
        timeout_s: int = 120,
        max_output_chars: int = 4000,
    ) -> str:
        commands = test_cmds if test_cmd.strip().lower() == "default" else [test_cmd]
        outputs: list[str] = []
        for command in commands:
            if not command.strip():
                continue
            validation_error = validate_verification_command(command)
            if validation_error is not None:
                outputs.append(format_tool_result(command, "BLOCKED", validation_error))
                continue
            if command_fn is not None:
                result = command_fn(command)
                if is_tool_result(result):
                    outputs.append(result)
                else:
                    outputs.append(format_tool_result(command, "COMPLETED", result))
                continue
            try:
                status, output = execute_command(
                    command,
                    repo_root=repo_root,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                status, output = f"TIMEOUT after {timeout_s}s", ""
            except OSError as exc:
                status, output = "ERROR", f"Error: {type(exc).__name__}: {exc}"
            if len(output) > max_output_chars:
                output = output[-max_output_chars:]
            outputs.append(format_tool_result(command, status, output))
        if outputs:
            return _annotate("\n---\n".join(outputs))
        return _annotate(format_tool_result(test_cmd, "SKIPPED", "No test commands available."))

    return MelleaTool.from_callable(_run_tests, name="run_tests")


def build_probe_python_tool(
    *,
    repo_root: str,
    command_fn: Callable[[str], str] | None = None,
    workspace=None,
):
    import shlex
    import subprocess

    from mellea.agent.runtime.workspace import format_workspace_state
    from mellea.agent.tools.bash import execute_command, format_tool_result, is_tool_result
    from mellea.backends.tools import MelleaTool

    workspace_state = format_workspace_state(workspace)

    def _annotate(result: str) -> str:
        if workspace_state is None or workspace_state in result:
            return result
        return f"{result}\n\n{workspace_state}"

    def _probe_python(
        code: str,
        timeout_s: int = 30,
        max_output_chars: int = 4000,
    ) -> str:
        source = str(code or "")
        if not source.strip():
            return _annotate(
                format_tool_result(
                    "python -c",
                    "BLOCKED",
                    "Provide Python code to run in `probe_python`.",
                )
            )
        command = f"python -c {shlex.quote(source)}"
        if command_fn is not None:
            result = command_fn(command)
            if is_tool_result(result):
                return _annotate(result)
            return _annotate(format_tool_result(command, "COMPLETED", result))
        try:
            status, output = execute_command(
                command,
                repo_root=repo_root,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            status, output = f"TIMEOUT after {timeout_s}s", ""
        except OSError as exc:
            status, output = "ERROR", f"Error: {type(exc).__name__}: {exc}"
        if len(output) > max_output_chars:
            output = output[-max_output_chars:]
        return _annotate(format_tool_result(command, status, output))

    return MelleaTool.from_callable(_probe_python, name="probe_python")


def build_verification_policy(
    *,
    repo_root: str,
    test_cmds: object | None = None,
    test_fn: Callable[[str], str] | None = None,
    timeout_s: int | None = None,
) -> VerificationPolicy:
    verification_cmds = normalize_verification_commands(test_cmds)
    del repo_root, timeout_s

    return VerificationPolicy(
        test_cmds=verification_cmds,
        test_fn=test_fn,
        prompt_block=build_verification_prompt(verification_cmds),
    )
