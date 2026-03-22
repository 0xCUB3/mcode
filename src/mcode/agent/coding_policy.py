from __future__ import annotations

from dataclasses import dataclass

_BASE_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in an open-source repository. "
    "You MUST edit existing source files to fix the bug. Do NOT create new files. "
    "Do NOT write test scripts. Only modify the existing code that contains the bug.\n\n"
    "Strategy:\n"
    "1. Read the issue carefully\n"
    "2. Search the codebase to find the relevant code\n"
    "3. Identify the root cause\n"
    "4. Make the minimal edit to fix it\n"
    "5. Call final_answer when done"
)

_EXPLORATION_SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in an open-source repository. "
    "You MUST edit existing source files to fix the bug. Do NOT create new files. "
    "Do NOT write test scripts. Only modify the existing code that contains the bug.\n\n"
    "Strategy:\n"
    "1. EXPLORE: Read the issue carefully. Search the codebase to find the relevant code. "
    "Read multiple files to understand the context. Do NOT edit anything yet.\n"
    "2. DIAGNOSE: Before making any edit, explain the root cause in your reasoning. "
    "If you cannot explain exactly why the current code is wrong, keep reading.\n"
    "3. EDIT: Make the minimal fix. Change the fewest lines possible. Prefer fixing the root "
    "cause over adding workarounds.\n"
    "4. VERIFY: Review your edit by reading the changed file. Make sure "
    "you didn't break anything.\n"
    "5. Call final_answer when done.\n\n"
    "Do NOT jump to editing after reading just one file. Understand the problem fully first."
)


@dataclass(frozen=True)
class CodingPolicy:
    system_prompt: str
    goal: str


def build_system_prompt(*, explore_prompt: bool) -> str:
    return _EXPLORATION_SYSTEM_PROMPT if explore_prompt else _BASE_SYSTEM_PROMPT


def build_goal(
    *,
    repo: str,
    problem_statement: str,
    repo_map_text: str = "",
    hints_text: str = "",
    verification_prompt: str = "",
) -> str:
    repo_map_block = f"\n\nRepository structure:\n{repo_map_text}" if repo_map_text else ""
    hints_block = f"\n\nAdditional context:\n{hints_text.strip()}" if hints_text.strip() else ""
    return (
        f"Fix this bug in {repo} by editing the existing source code.\n\n"
        f"Issue:\n{problem_statement.strip()}"
        f"{repo_map_block}{hints_block}{verification_prompt}\n\n"
        "Only edit existing files. Do not create new files or test scripts."
    )


def build_coding_policy(
    *,
    repo: str,
    problem_statement: str,
    hints_text: str = "",
    repo_map_text: str = "",
    verification_prompt: str = "",
    explore_prompt: bool = False,
) -> CodingPolicy:
    return CodingPolicy(
        system_prompt=build_system_prompt(explore_prompt=explore_prompt),
        goal=build_goal(
            repo=repo,
            problem_statement=problem_statement,
            hints_text=hints_text,
            repo_map_text=repo_map_text,
            verification_prompt=verification_prompt,
        ),
    )
