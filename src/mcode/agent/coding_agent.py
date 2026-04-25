from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from mcode.agent.coding_policy import CodingPolicy, build_coding_policy
from mcode.agent.repo_customization import load_repo_customization
from mcode.agent.tooling import (
    build_candidate_files,
    build_repo_map,
    find_file,
    list_dir,
    read_file,
    search_code,
    str_replace_edit,
    suggest_verification_commands,
)
from mcode.agent.verification import (
    VerificationPolicy,
    VerificationProgress,
    build_run_tests_tool,
    build_verification_policy,
)
from mcode.agent.workspace_context import collect_workspace_context
from mcode.llm.harness_experiments import active_harness_experiments
from mcode.mellea_compat import build_tool_from_callable


@dataclass(frozen=True)
class CodingAgentAssembly:
    repo: str
    repo_root: str
    coding_policy: CodingPolicy
    verification_policy: VerificationPolicy
    tools: list
    model_options: dict
    loop_budget: int
    timeout_s: int
    harness_experiments: tuple[str, ...]

    @property
    def system_prompt(self) -> str:
        return self.coding_policy.system_prompt

    @property
    def goal(self) -> str:
        return self.coding_policy.goal

    @property
    def verification_cmds(self) -> list[str]:
        return self.verification_policy.test_cmds


def build_coding_agent(
    *,
    session,
    repo: str,
    problem_statement: str,
    hints_text: str = "",
    repo_root: str,
    visible_repo_root: str | None = None,
    test_cmds: object | None = None,
    test_fn=None,
    command_fn: Callable[[str], str] | None = None,
) -> CodingAgentAssembly:
    del visible_repo_root

    budget = max(1, session.loop_budget)
    timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))
    verification_command_suggestions = []
    try:
        verification_command_suggestions = suggest_verification_commands(repo_root)
    except Exception as e:
        print(f"  [verification_suggestions] failed: {e}", flush=True)

    verification_policy = build_verification_policy(
        test_cmds=test_cmds,
        test_fn=test_fn,
        command_fn=command_fn,
        suggested_test_cmds=verification_command_suggestions,
    )

    repo_map_text = ""
    try:
        repo_map_text = build_repo_map(repo_root, problem_statement, max_tokens=4096)
    except Exception as e:
        print(f"  [repo_map] failed: {e}", flush=True)

    candidate_files_text = ""
    try:
        candidate_files_text = build_candidate_files(repo_root, problem_statement, top_n=6)
    except Exception as e:
        print(f"  [localization] failed: {e}", flush=True)

    workspace_context_text = ""
    try:
        workspace_context_text = collect_workspace_context(repo_root, problem_statement).text
    except Exception as e:
        print(f"  [workspace_context] failed: {e}", flush=True)

    repo_customization = load_repo_customization(repo_root)
    coding_policy = build_coding_policy(
        repo=repo,
        problem_statement=problem_statement,
        hints_text=hints_text,
        repo_map_text=repo_map_text,
        candidate_files_text=candidate_files_text,
        repo_customization_text=repo_customization.text,
        workspace_context_text=workspace_context_text,
        verification_prompt=verification_policy.prompt_block,
    )
    harness_experiments = active_harness_experiments()
    tools = _build_tools_for_experiments(
        repo_root,
        verification_policy=verification_policy,
        harness_experiments=harness_experiments,
    )

    return CodingAgentAssembly(
        repo=repo,
        repo_root=repo_root,
        coding_policy=coding_policy,
        verification_policy=verification_policy,
        tools=tools,
        model_options=session._model_options(system_prompt=coding_policy.system_prompt),
        loop_budget=budget,
        timeout_s=timeout_s,
        harness_experiments=harness_experiments,
    )


def _build_tools_for_experiments(
    repo_root: str,
    *,
    verification_policy: VerificationPolicy,
    harness_experiments: tuple[str, ...],
):
    del harness_experiments
    return make_agent_tools(repo_root, verification_policy=verification_policy)


def make_agent_tools(
    repo_root: str,
    *,
    verification_policy: VerificationPolicy,
):
    progress = VerificationProgress()

    def _search(query: str) -> str:
        return search_code(query, repo_root=repo_root)

    def _edit(path: str, old_str: str, new_str: str) -> str:
        result = str_replace_edit(path, old_str, new_str, repo_root=repo_root)
        if "APPLIED" in result:
            progress.note_edit_applied()
        return result

    def _read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        return read_file(path, start_line, end_line, repo_root=repo_root)

    def _find(pattern: str) -> str:
        return find_file(pattern, repo_root=repo_root)

    def _list(path: str = ".") -> str:
        return list_dir(path, repo_root=repo_root)

    tools = [
        build_tool_from_callable(_search, name="search_code"),
        build_tool_from_callable(_edit, name="edit"),
        build_tool_from_callable(_read, name="read_file"),
        build_tool_from_callable(_find, name="find_file"),
        build_tool_from_callable(_list, name="list_dir"),
    ]
    run_tests_tool = build_run_tests_tool(
        repo_root=repo_root,
        verification_policy=verification_policy,
        progress=progress,
    )
    if run_tests_tool is not None:
        tools.append(run_tests_tool)
    return tools
