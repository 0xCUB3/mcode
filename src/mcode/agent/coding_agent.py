from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from mellea.backends.tools import MelleaTool

from mcode.agent.repo_customization import load_repo_customization
from mcode.agent.tooling import find_file, list_dir, read_file, search_code, str_replace_edit
from mcode.agent.verification import (
    VerificationPolicy,
    VerificationProgress,
    build_run_tests_tool,
    build_verification_policy,
)

_BASE_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in an open-source repository. "
    "You MUST edit existing source files to fix the bug. Do NOT create new files. "
    "Do NOT write test scripts. Only modify the existing code that contains the bug.\n\n"
    "Use the structured code tools to search, read, edit, and verify. Start narrow, make one "
    "concrete edit once you have a target, then use `run_tests` before `final_answer`. "
    "When you call `final_answer`, keep the answer short."
)


def _build_coding_prompt(
    *,
    repo: str,
    problem_statement: str,
    hints_text: str = "",
    repo_customization_text: str = "",
    verification_prompt: str = "",
 ) -> tuple[str, str]:
    hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""
    customization_block = (
        f"\n\nRepository-specific guidance:\n{repo_customization_text.strip()}"
        if repo_customization_text.strip()
        else ""
    )
    goal = (
        f"Fix this bug in {repo} by editing the existing source code.\n\n"
        f"Issue:\n{problem_statement.strip()}"
        f"{hints_block}{customization_block}{verification_prompt}\n\n"
        "Do not open a second solving path. Diagnose, edit, verify, then submit."
    )
    return _BASE_SYSTEM_PROMPT, goal


@dataclass(frozen=True)
class CodingAgentAssembly:
    repo: str
    repo_root: str
    system_prompt: str
    goal: str
    verification_policy: VerificationPolicy
    tools: list
    model_options: dict
    loop_budget: int
    timeout_s: int

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
    system_prompt, goal = _build_coding_prompt(
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
        system_prompt=system_prompt,
        goal=goal,
        verification_policy=verification_policy,
        tools=tools,
        model_options=session._model_options(system_prompt=system_prompt),
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
