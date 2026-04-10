from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from mcode.launch import service as service_module
from mcode.launch.config import LaunchConfig, load_launch_config
from mcode.launch.models import SyncMode
from mcode.launch.profiles import resolve_serving_profile
from mcode.launch.state import LauncherState, RunHandle, ServerHandle, load_state, update_state
from mcode.launch.sync import (
    SyncPlan,
    WorkspaceSignatureInput,
    build_sync_plan,
    build_workspace_signature,
    list_untracked_files,
)


def test_resolve_serving_profile_for_gemma4() -> None:
    profile = resolve_serving_profile("google/gemma-4-31B-it")

    assert profile.name == "gemma4"
    assert "--tool-call-parser" in profile.flags
    assert "gemma4" in profile.flags


def test_resolve_serving_profile_for_qwen3() -> None:
    profile = resolve_serving_profile("Qwen/Qwen3.5-27B")

    assert profile.name == "qwen3"
    assert "qwen3_coder" in profile.flags


def test_resolve_serving_profile_defaults_to_openai_compatible() -> None:
    profile = resolve_serving_profile("meta-llama/Llama-3.1-70B-Instruct")

    assert profile.name == "default"
    assert profile.flags == []


def test_workspace_signature_changes_with_overlay_hash() -> None:
    left = build_workspace_signature(
        WorkspaceSignatureInput(
            repo_url="https://github.com/0xCUB3/mcode.git",
            ref_sha="abc123",
            overlay_patch_sha="one",
            bootstrap_key="extras:swebench,datasets",
        )
    )
    right = build_workspace_signature(
        WorkspaceSignatureInput(
            repo_url="https://github.com/0xCUB3/mcode.git",
            ref_sha="abc123",
            overlay_patch_sha="two",
            bootstrap_key="extras:swebench,datasets",
        )
    )

    assert left != right


def test_load_launch_config_reads_bluevela_defaults(tmp_path: Path) -> None:
    path = tmp_path / "launch.toml"
    path.write_text(
        """
[bluevela]
login = "user@login3.example.com"
workspace_root = "/u/user/mcode-launch"
queue = "normal"
group = "grp_runtime"
"""
    )

    config = load_launch_config(path)

    assert isinstance(config, LaunchConfig)
    assert config.bluevela.login == "user@login3.example.com"
    assert config.bluevela.workspace_root == "/u/user/mcode-launch"


def test_load_launch_config_derives_podman_roots_from_shared_root(tmp_path: Path) -> None:
    path = tmp_path / "launch.toml"
    path.write_text(
        """
[bluevela]
shared_root = "/proj/shared/user"
"""
    )

    config = load_launch_config(path)

    assert config.bluevela.podman_graphroot == "/proj/shared/user/podman/graphroot"
    assert config.bluevela.podman_runroot == "/proj/shared/user/podman/runroot"


def _build_launch_spec(**overrides):
    kwargs = {
        "config": load_launch_config(Path("/does/not/exist")),
        "target": "bluevela",
        "model": "Qwen/Qwen3-1.7B",
        "benchmark": "swebench-lite",
        "backend": None,
        "split": None,
        "loop_budget": 1,
        "timeout": 60,
        "parallelism": 1,
        "limit": 1,
        "task_ids": None,
        "reuse": "prefer",
        "sync_mode": "git-overlay",
        "ref": "HEAD",
        "json_mode": False,
        "yes": False,
        "follow": False,
        "tp": 1,
        "dp": 1,
        "api_server_count": 1,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.9,
        "port": None,
        "serving_profile": None,
        "no_auto_profile": False,
        "keep_alive": None,
        "ollama_num_parallel": None,
        "ollama_max_queue": None,
        "openai_base_url": None,
        "dataset": None,
    }
    kwargs.update(overrides)
    return service_module.build_launch_spec(**kwargs)


def test_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "launch-state.json"
    state = LauncherState(
        servers=[
            ServerHandle(
                id="srv-1",
                target="bluevela",
                reuse_key="bluevela:model",
                endpoint="http://host:8321/v1",
                status="healthy",
                metadata={"job_id": "123"},
            )
        ],
        runs=[
            RunHandle(
                id="run-1",
                target="bluevela",
                benchmark="swebench-live",
                status="running",
                metadata={"job_id": "456"},
            )
        ],
    )

    update_state(
        path,
        lambda current: (
            setattr(current, "servers", state.servers),
            setattr(current, "runs", state.runs),
            setattr(current, "workspaces", state.workspaces),
        ),
    )
    loaded = load_state(path)

    assert loaded == state
    assert json.loads(path.read_text())["servers"][0]["id"] == "srv-1"


def test_sync_mode_enum_matches_cli_contract() -> None:
    assert SyncMode.GIT_OVERLAY.value == "git-overlay"
    assert SyncMode.GIT_REF.value == "git-ref"
    assert SyncMode.WORKING_TREE.value == "working-tree"


def test_build_sync_plan_noops_when_existing_workspace_matches(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "tracked.txt").write_text("hello\n")
    (repo_root / ".git").mkdir()

    def fake_git_output(path: Path, *args: str) -> str:
        assert path == repo_root
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc123\n"
        if args[:3] == ("remote", "get-url", "origin"):
            return "https://github.com/0xCUB3/mcode.git\n"
        if args[:4] == ("diff", "--binary", "HEAD", "--"):
            return ""
        raise AssertionError(args)

    from mcode.launch import sync as sync_module

    original = sync_module._git_output
    sync_module._git_output = fake_git_output
    try:
        plan1 = build_sync_plan(
            repo_root,
            sync=type(
                "S", (), {"mode": SyncMode.GIT_OVERLAY, "ref": "HEAD", "bootstrap_key": "uv"}
            )(),
            workspace_root="/u/user/mcode-launch",
        )
        plan2 = build_sync_plan(
            repo_root,
            sync=type(
                "S", (), {"mode": SyncMode.GIT_OVERLAY, "ref": "HEAD", "bootstrap_key": "uv"}
            )(),
            workspace_root="/u/user/mcode-launch",
            existing=type("W", (), {"signature": plan1.signature, "path": plan1.remote_path})(),
        )
    finally:
        sync_module._git_output = original

    assert isinstance(plan1, SyncPlan)
    assert plan2.is_noop is True


def test_list_untracked_files_reads_git_output(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    def fake_git_output(path: Path, *args: str) -> str:
        assert path == repo_root
        assert args == ("ls-files", "--others", "--exclude-standard")
        return "src/new_file.py\nnotes.txt\n"

    from mcode.launch import sync as sync_module

    original = sync_module._git_output
    sync_module._git_output = fake_git_output
    try:
        files = list_untracked_files(repo_root)
    finally:
        sync_module._git_output = original

    assert files == ["src/new_file.py", "notes.txt"]


def test_endpoint_health_tolerates_transient_socket_errors() -> None:
    calls = 0
    original = service_module.urlopen

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            raise ConnectionResetError("reset")
        return _Response()

    try:
        service_module.urlopen = fake_urlopen
        assert service_module._endpoint_is_healthy("http://example.com/v1") is True
    finally:
        service_module.urlopen = original


def test_build_launch_spec_defaults_split_for_swebench_lite() -> None:
    spec = _build_launch_spec()

    assert spec.benchmark.split == "test"
    assert spec.benchmark.dataset == "SWE-bench/SWE-bench_Lite"


def test_build_launch_spec_defaults_split_for_swebench_live() -> None:
    spec = _build_launch_spec(benchmark="swebench-live")

    assert spec.benchmark.split == "verified"


def test_build_launch_spec_normalizes_task_id_file_to_inline_ids(tmp_path: Path) -> None:
    task_file = tmp_path / "task_ids.txt"
    task_file.write_text("sympy__sympy-1\nsympy__sympy-2\n")
    original = service_module._find_missing_task_ids
    service_module._find_missing_task_ids = lambda *args, **kwargs: []

    try:
        spec = _build_launch_spec(
            model="google/gemma-4-31B-it",
            limit=None,
            task_ids=str(task_file),
        )
    finally:
        service_module._find_missing_task_ids = original

    assert spec.benchmark.task_ids == "sympy__sympy-1,sympy__sympy-2"


def test_build_launch_spec_warns_on_partial_task_id_overlap(tmp_path: Path) -> None:
    task_file = tmp_path / "task_ids.txt"
    task_file.write_text("sympy__sympy-1\nsympy__sympy-2\n")
    original = service_module._find_missing_task_ids
    service_module._find_missing_task_ids = lambda *args, **kwargs: ["sympy__sympy-2"]

    try:
        with pytest.warns(UserWarning, match=r"matched 1/2 tasks.*sympy__sympy-2"):
            spec = _build_launch_spec(
                model="google/gemma-4-31B-it",
                limit=None,
                task_ids=str(task_file),
                dataset="princeton-nlp/SWE-bench_Verified",
            )
    finally:
        service_module._find_missing_task_ids = original

    assert spec.benchmark.dataset == "princeton-nlp/SWE-bench_Verified"


def test_build_launch_spec_rejects_zero_task_id_overlap(tmp_path: Path) -> None:
    task_file = tmp_path / "task_ids.txt"
    task_file.write_text("sympy__sympy-1\nsympy__sympy-2\n")
    original = service_module._find_missing_task_ids
    service_module._find_missing_task_ids = lambda *args, **kwargs: [
        "sympy__sympy-1",
        "sympy__sympy-2",
    ]

    try:
        with pytest.raises(ValueError, match=r"matched 0/2 tasks"):
            _build_launch_spec(
                model="google/gemma-4-31B-it",
                limit=None,
                task_ids=str(task_file),
            )
    finally:
        service_module._find_missing_task_ids = original


def test_remote_endpoint_health_uses_ssh_result() -> None:
    original = service_module._run_ssh_result

    try:
        service_module._run_ssh_result = lambda *args, **kwargs: CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )
        assert service_module._remote_endpoint_is_healthy(
            "user@login3.example.com", "http://host:8321/v1"
        )
    finally:
        service_module._run_ssh_result = original


def test_launch_run_rejects_follow_without_yes(tmp_path: Path) -> None:
    config = load_launch_config(Path("/does/not/exist"))
    spec = service_module.build_launch_spec(
        config=config,
        target="local-vllm",
        model="Qwen/Qwen3.5-27B",
        benchmark="swebench-live",
        backend=None,
        split=None,
        loop_budget=1,
        timeout=60,
        parallelism=1,
        limit=1,
        task_ids=None,
        reuse="prefer",
        sync_mode="git-overlay",
        ref="HEAD",
        json_mode=False,
        yes=False,
        follow=True,
        tp=1,
        dp=1,
        api_server_count=1,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        port=None,
        serving_profile=None,
        no_auto_profile=False,
        keep_alive=None,
        ollama_num_parallel=None,
        ollama_max_queue=None,
        openai_base_url=None,
    )

    result = service_module.launch_run(spec, repo_root=tmp_path)

    assert result.ok is False
    assert "--follow requires --yes" in result.message
