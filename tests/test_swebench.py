from __future__ import annotations

import types

import pytest

import mcode.execution.sandbox as sandbox_module
from mcode.execution.sandbox import DockerUnavailableError
from mcode.execution.swebench import (
    SWEbenchSandbox,
    _build_agent_setup_script,
    _build_agent_shell_command,
)


def test_build_agent_setup_script_keeps_eval_setup_and_drops_patch_steps():
    script = _build_agent_setup_script(
        [
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "git config --global --add safe.directory /testbed",
            "git status",
            "git show",
            "git -c core.fileMode=false diff abc123",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "python -m pip install -e .[test] --verbose",
            "python setup.py build_ext --inplace",
            "git checkout abc123 path/to/test_file.py",
            "git apply -v - <<'EOF'",
            ": '>>>>> Start Test Output'",
        ]
    )

    assert "python -m pip install -e .[test] --verbose" in script
    assert "python setup.py build_ext --inplace" in script
    assert "git status" not in script
    assert "git show" not in script
    assert "git -c core.fileMode=false diff abc123" not in script
    assert "git checkout abc123 path/to/test_file.py" not in script
    assert "git apply -v - <<'EOF'" not in script
    assert ">>>>> Start Test Output" not in script


def test_build_agent_shell_command_activates_testbed_and_rewrites_repo_root():
    command = "cd /tmp/mcode-testbed-999/testbed && python -m pytest -q"
    wrapped = _build_agent_shell_command(
        command,
        host_repo_root="/tmp/mcode-testbed-999/testbed",
    )

    assert "source /opt/miniconda3/bin/activate" in wrapped
    assert "conda activate testbed" in wrapped
    assert "git config --global --add safe.directory /testbed" in wrapped
    assert "cd /testbed && python -m pytest -q" in wrapped
    assert "/tmp/mcode-testbed-999/testbed" not in wrapped


def test_swebench_get_client_retries_after_stale_client(monkeypatch):
    class FakeDockerException(Exception):
        pass

    class FakeClient:
        def __init__(self, *, fail_ping: bool = False) -> None:
            self.fail_ping = fail_ping
            self.closed = False

        def ping(self) -> None:
            if self.fail_ping:
                raise FakeDockerException("socket missing")

        def close(self) -> None:
            self.closed = True

    stale = FakeClient(fail_ping=True)
    fresh = FakeClient()
    calls: list[str] = []

    def fake_from_env():
        calls.append("from_env")
        return fresh

    fake_docker = types.SimpleNamespace(
        from_env=fake_from_env,
        errors=types.SimpleNamespace(DockerException=FakeDockerException),
    )
    monkeypatch.setattr(sandbox_module, "docker", fake_docker)
    monkeypatch.setenv("MCODE_DOCKER_CONNECT_RETRIES", "2")
    monkeypatch.setenv("MCODE_DOCKER_RETRY_DELAY", "0")

    sandbox = SWEbenchSandbox()
    sandbox._client = stale

    client = sandbox._get_client()

    assert client is fresh
    assert stale.closed is True
    assert calls == ["from_env"]


def test_swebench_get_client_raises_docker_unavailable_after_retries(monkeypatch):
    class FakeDockerException(Exception):
        pass

    def fake_from_env():
        raise FakeDockerException("socket missing")

    fake_docker = types.SimpleNamespace(
        from_env=fake_from_env,
        errors=types.SimpleNamespace(DockerException=FakeDockerException),
    )
    monkeypatch.setattr(sandbox_module, "docker", fake_docker)
    monkeypatch.setenv("MCODE_DOCKER_CONNECT_RETRIES", "2")
    monkeypatch.setenv("MCODE_DOCKER_RETRY_DELAY", "0")

    sandbox = SWEbenchSandbox()

    with pytest.raises(DockerUnavailableError, match="SWE-bench Lite"):
        sandbox._get_client()
