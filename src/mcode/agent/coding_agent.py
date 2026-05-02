from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from mellea.backends.tools import MelleaTool

from mcode.agent.coding_policy import CodingPolicy, build_coding_policy
from mcode.agent.repo_customization import load_repo_customization
from mcode.agent.tooling import find_file, list_dir, read_file, search_code, str_replace_edit
from mcode.agent.verification import (
    VerificationPolicy,
    VerificationProgress,
    build_run_tests_tool,
    build_verification_policy,
)


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

    budget = max(1, session.loop_budget)
    timeout_s = int(os.environ.get("MCODE_REACT_TIMEOUT", str(budget * 30)))
    verification_policy = build_verification_policy(
        test_cmds=test_cmds,
        test_fn=test_fn,
        command_fn=command_fn,
    )

    repo_customization = load_repo_customization(repo_root)
    coding_policy = build_coding_policy(
        repo=repo,
        problem_statement=problem_statement,
        hints_text=hints_text,
        repo_customization_text=repo_customization.text,
        verification_prompt=verification_policy.prompt_block,
    )
    tools = make_agent_tools(
        repo_root,
        verification_policy=verification_policy,
        visible_repo_root=visible_repo_root,
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
    )


_VISIBLE_REPO_ALIASES = ("/testbed", "/home/user/repo", "c:/users/user/tmp/repo")


def _normalize_tool_path(path: str, *, visible_repo_root: str | None) -> str:
    text = path.strip()
    aliases = [alias for alias in (visible_repo_root, *_VISIBLE_REPO_ALIASES) if alias]
    lowered = text.lower()
    for alias in aliases:
        normalized_alias = alias.rstrip("/")
        alias_lower = normalized_alias.lower()
        if lowered == alias_lower:
            return "."
        if lowered.startswith(alias_lower + "/"):
            return text[len(normalized_alias) + 1 :]
    return path


def make_agent_tools(
    repo_root: str,
    *,
    verification_policy: VerificationPolicy,
    visible_repo_root: str | None = None,
):
    progress = VerificationProgress()

    def _search(query: str) -> str:
        return search_code(query, repo_root=repo_root)

    def _edit(path: str, old_str: str, new_str: str) -> str:
        result = str_replace_edit(
            _normalize_tool_path(path, visible_repo_root=visible_repo_root),
            old_str,
            new_str,
            repo_root=repo_root,
        )
        if "APPLIED" in result:
            progress.note_edit_applied()
        return result

    def _read(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        return read_file(
            _normalize_tool_path(path, visible_repo_root=visible_repo_root),
            start_line or 1,
            end_line,
            repo_root=repo_root,
        )

    def _find(pattern: str) -> str:
        return find_file(pattern, repo_root=repo_root)

    def _list(path: str | None = None) -> str:
        return list_dir(
            _normalize_tool_path(path or ".", visible_repo_root=visible_repo_root),
            repo_root=repo_root,
        )

    tools = [
        MelleaTool.from_callable(_search, name="search_code"),
        MelleaTool.from_callable(_edit, name="edit"),
        MelleaTool.from_callable(_read, name="read_file"),
        MelleaTool.from_callable(_find, name="find_file"),
        MelleaTool.from_callable(_list, name="list_dir"),
    ]
    run_tests_tool = build_run_tests_tool(
        repo_root=repo_root,
        verification_policy=verification_policy,
        progress=progress,
    )
    if run_tests_tool is not None:
        tools.append(run_tests_tool)
    return tools
