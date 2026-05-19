from __future__ import annotations

import io
import sys
import tarfile
import types

import pytest

import mcode.bench.swebench.docker as sandbox_module
from mcode.bench.swebench.docker import DockerUnavailableError, is_docker_unavailable_error
from mcode.bench.swebench.lite import (
    RetryablePodmanImageError,
    SWEbenchSandbox,
    _build_agent_setup_script,
    _build_agent_shell_command,
    _ensure_image,
    _extract_agent_test_commands,
    _is_retryable_podman_image_error,
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


def test_extract_agent_test_commands_from_eval_script():
    commands = _extract_agent_test_commands(
        [
            "python -m pip install -e .",
            ": '>>>>> Start Test Output'",
            "./tests/runtests.py --verbosity 2 auth_tests.test_validators",
            ": '>>>>> End Test Output'",
            "git checkout abc tests/foo.py",
        ]
    )

    assert commands == ["./tests/runtests.py --verbosity 2 auth_tests.test_validators"]


def test_build_agent_shell_command_activates_testbed_and_rewrites_repo_root(tmp_path):
    repo_root = tmp_path / "mcode-testbed" / "testbed"
    command = f"cd {repo_root} && python -m pytest -q"
    wrapped = _build_agent_shell_command(command, host_repo_root=str(repo_root))

    assert "source /opt/miniconda3/bin/activate" in wrapped
    assert "conda activate testbed" in wrapped
    assert "git config --global --add safe.directory /testbed" in wrapped
    assert "cd /testbed && python -m pytest -q" in wrapped
    assert str(repo_root) not in wrapped


def test_build_agent_shell_command_rewrites_common_repo_aliases():
    command = "cd /home/user/repo && python -m pytest -q"
    wrapped = _build_agent_shell_command(command)

    assert "cd /testbed && python -m pytest -q" in wrapped
    assert "/home/user/repo" not in wrapped


def test_build_agent_shell_command_does_not_corrupt_repos_alias():
    command = "cd /home/user/repos/django && python -m pytest -q"
    wrapped = _build_agent_shell_command(command)

    assert "cd /testbed && python -m pytest -q" in wrapped
    assert "/testbeds/django" not in wrapped


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


def test_is_docker_unavailable_error_matches_podman_socket_timeouts():
    exc = RuntimeError(
        "ReadTimeout: UnixHTTPConnectionPool(host='localhost', port=None): "
        "Read timed out. (read timeout=60)"
    )

    assert is_docker_unavailable_error(exc) is True


def test_retryable_podman_image_error_matches_observed_unpack_failure():
    err = (
        "writing blob: adding layer with blob "
        "sha256:abc: unpacking failed: Chown error detected. "
        "potentially insufficient UIDs or GIDs available in user namespace"
    )

    assert _is_retryable_podman_image_error(err) is True


class _FakeImageNotFound(Exception):
    pass


class _FakeDigestImage:
    def __init__(self, repo_digests: list[str] | None = None) -> None:
        self.attrs = {"RepoDigests": repo_digests or []}


class _FakeDigestImages:
    def __init__(self, image: object | None) -> None:
        self.image = image
        self.calls: list[str] = []

    def get(self, name: str) -> object:
        self.calls.append(name)
        if self.image is None:
            raise _FakeImageNotFound(name)
        return self.image


class _FakeDigestApi:
    def __init__(
        self,
        *,
        registry_digest: str = "sha256:abc",
        inspect_error: Exception | None = None,
    ) -> None:
        self.registry_digest = registry_digest
        self.inspect_error = inspect_error
        self.inspect_calls: list[str] = []
        self.pull_calls: list[str] = []

    def inspect_distribution(self, fq: str) -> dict[str, dict[str, str]]:
        self.inspect_calls.append(fq)
        if self.inspect_error is not None:
            raise self.inspect_error
        return {"Descriptor": {"digest": self.registry_digest}}

    def pull(self, fq: str, *, stream: bool, decode: bool):
        assert stream is True
        assert decode is True
        self.pull_calls.append(fq)
        yield {"status": "pulled"}


class _FakeDigestClient:
    def __init__(self, image: object | None, api: _FakeDigestApi) -> None:
        self.images = _FakeDigestImages(image)
        self.api = api


def _install_fake_docker(monkeypatch) -> None:
    fake_docker = types.SimpleNamespace(
        errors=types.SimpleNamespace(ImageNotFound=_FakeImageNotFound)
    )
    monkeypatch.setitem(sys.modules, "docker", fake_docker)


def test_ensure_image_reuses_cached_image_when_digest_matches(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    client = _FakeDigestClient(
        _FakeDigestImage(["docker.io/swebench/example:latest@sha256:abc"]),
        _FakeDigestApi(registry_digest="sha256:abc"),
    )

    action = _ensure_image(
        client,
        "swebench/example:latest",
        check_image_digests=True,
    )

    assert action == "cached"
    assert client.api.inspect_calls == ["docker.io/swebench/example:latest"]
    assert client.api.pull_calls == []


def test_ensure_image_retries_digest_check_with_unqualified_name(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    class FallbackApi(_FakeDigestApi):
        def inspect_distribution(self, fq: str) -> dict[str, dict[str, str]]:
            self.inspect_calls.append(fq)
            if fq == "docker.io/swebench/example:latest":
                raise RuntimeError("404 not found")
            return {"Descriptor": {"digest": self.registry_digest}}

    client = _FakeDigestClient(
        _FakeDigestImage(["docker.io/swebench/example:latest@sha256:abc"]),
        FallbackApi(registry_digest="sha256:abc"),
    )

    action = _ensure_image(
        client,
        "swebench/example:latest",
        check_image_digests=True,
    )

    assert action == "cached"
    assert client.api.inspect_calls == [
        "docker.io/swebench/example:latest",
        "swebench/example:latest",
    ]


def test_ensure_image_refreshes_cached_image_when_tag_moves(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    client = _FakeDigestClient(
        _FakeDigestImage(["docker.io/swebench/example:latest@sha256:old"]),
        _FakeDigestApi(registry_digest="sha256:new"),
    )

    action = _ensure_image(
        client,
        "swebench/example:latest",
        check_image_digests=True,
    )

    assert action == "refreshed"
    assert client.api.pull_calls == ["docker.io/swebench/example:latest"]


def test_ensure_image_pulls_missing_image(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    api = _FakeDigestApi(registry_digest="sha256:unused")
    client = _FakeDigestClient(None, api)

    action = _ensure_image(
        client,
        "swebench/example:latest",
        check_image_digests=True,
    )

    assert action == "pulled"
    assert client.api.inspect_calls == []
    assert client.api.pull_calls == ["docker.io/swebench/example:latest"]


def test_ensure_image_raises_retryable_error_when_digest_check_fails(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    client = _FakeDigestClient(
        _FakeDigestImage(["docker.io/swebench/example:latest@sha256:abc"]),
        _FakeDigestApi(inspect_error=RuntimeError("registry unavailable")),
    )

    with pytest.raises(RetryablePodmanImageError, match="digest check failed"):
        _ensure_image(
            client,
            "swebench/example:latest",
            check_image_digests=True,
        )

    assert client.api.pull_calls == []


def test_ensure_image_skips_digest_check_when_disabled(tmp_path, monkeypatch):
    _install_fake_docker(monkeypatch)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    client = _FakeDigestClient(
        _FakeDigestImage(["docker.io/swebench/example:latest@sha256:old"]),
        _FakeDigestApi(inspect_error=RuntimeError("registry unavailable")),
    )

    action = _ensure_image(
        client,
        "swebench/example:latest",
        check_image_digests=False,
    )

    assert action == "cached"
    assert client.api.inspect_calls == []
    assert client.api.pull_calls == []


def test_ensure_image_retries_retryable_pull_failure(tmp_path, monkeypatch):
    class ImageNotFound(Exception):
        pass

    class FakeImages:
        def get(self, name):
            raise ImageNotFound(name)

    class FakeApi:
        def __init__(self) -> None:
            self.calls = 0

        def pull(self, fq, *, stream, decode):
            assert fq == "docker.io/swebench/example:latest"
            assert stream is True
            assert decode is True
            self.calls += 1
            yield {"error": "unpacking failed: Chown error detected"}

    class FakeClient:
        images = FakeImages()

        def __init__(self) -> None:
            self.api = FakeApi()

    fake_docker = types.SimpleNamespace(errors=types.SimpleNamespace(ImageNotFound=ImageNotFound))
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    client = FakeClient()
    with pytest.raises(RetryablePodmanImageError):
        _ensure_image(client, "swebench/example:latest")

    assert client.api.calls == 2


def test_ensure_image_retries_retryable_inspect_failure(tmp_path, monkeypatch):
    class ImageNotFound(Exception):
        pass

    class FakeImages:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, name):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "ReadTimeout: UnixHTTPConnectionPool(host='localhost', port=None): "
                    "Read timed out. (read timeout=60)"
                )
            return object()

    class FakeApi:
        def pull(self, fq, *, stream, decode):
            del fq, stream, decode
            raise AssertionError("pull should not run when inspect retry succeeds")

    class FakeClient:
        def __init__(self) -> None:
            self.images = FakeImages()
            self.api = FakeApi()

    fake_docker = types.SimpleNamespace(errors=types.SimpleNamespace(ImageNotFound=ImageNotFound))
    monkeypatch.setitem(sys.modules, "docker", fake_docker)
    monkeypatch.setenv("MCODE_PODMAN_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("MCODE_PODMAN_PULL_RETRY_DELAY", "0")

    client = FakeClient()
    _ensure_image(client, "swebench/example:latest", check_image_digests=False)

    assert client.images.calls == 2


def test_repo_context_disables_network_for_source_container(monkeypatch):
    class FakeSourceContainer:
        def get_archive(self, path):
            assert path == "/testbed"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo("testbed")
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            buf.seek(0)
            return [buf.getvalue()], {}

        def remove(self, force=False):
            assert force is True

    class FakeExecContainer:
        def start(self):
            return None

        def remove(self, force=False):
            assert force is True

    create_calls: list[dict] = []
    containers = [FakeSourceContainer(), FakeExecContainer()]

    class FakeContainerManager:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            return containers.pop(0)

    class FakeClient:
        containers = FakeContainerManager()

    fake_test_spec = types.SimpleNamespace(
        instance_image_key="docker.io/example/image:latest",
        platform="linux/amd64",
        eval_script_list=[],
    )

    fake_test_spec_module = types.ModuleType("swebench.harness.test_spec.test_spec")
    fake_test_spec_module.make_test_spec = lambda *args, **kwargs: fake_test_spec
    monkeypatch.setitem(sys.modules, "swebench", types.ModuleType("swebench"))
    monkeypatch.setitem(sys.modules, "swebench.harness", types.ModuleType("swebench.harness"))
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.test_spec",
        types.ModuleType("swebench.harness.test_spec"),
    )
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec.test_spec", fake_test_spec_module)

    monkeypatch.setattr(
        "mcode.bench.swebench.lite._ensure_image",
        lambda client, name, **kwargs: None,
    )
    monkeypatch.setattr(
        "mcode.bench.swebench.lite._exec_agent_command_in_container",
        lambda *args, **kwargs: ("", 0, False),
    )

    sandbox = SWEbenchSandbox()
    monkeypatch.setattr(sandbox, "_get_client", lambda: FakeClient())

    with sandbox.repo_context({"instance_id": "astropy__astropy-12907"}):
        pass

    assert create_calls[0]["network_disabled"] is True
    # cpu_limit defaults to None → no cpu_quota / cpu_period kwargs leaked
    assert "cpu_quota" not in create_calls[1]
    assert "cpu_period" not in create_calls[1]


def test_repo_context_caps_exec_container_cpu_when_cpu_limit_set(monkeypatch):
    class FakeSourceContainer:
        def get_archive(self, path):
            assert path == "/testbed"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo("testbed")
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            buf.seek(0)
            return [buf.getvalue()], {}

        def remove(self, force=False):
            assert force is True

    class FakeExecContainer:
        def start(self):
            return None

        def remove(self, force=False):
            assert force is True

    create_calls: list[dict] = []
    containers = [FakeSourceContainer(), FakeExecContainer()]

    class FakeContainerManager:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            return containers.pop(0)

    class FakeClient:
        containers = FakeContainerManager()

    fake_test_spec = types.SimpleNamespace(
        instance_image_key="docker.io/example/image:latest",
        platform="linux/amd64",
        eval_script_list=[],
    )
    fake_test_spec_module = types.ModuleType("swebench.harness.test_spec.test_spec")
    fake_test_spec_module.make_test_spec = lambda *args, **kwargs: fake_test_spec
    monkeypatch.setitem(sys.modules, "swebench", types.ModuleType("swebench"))
    monkeypatch.setitem(sys.modules, "swebench.harness", types.ModuleType("swebench.harness"))
    monkeypatch.setitem(
        sys.modules,
        "swebench.harness.test_spec",
        types.ModuleType("swebench.harness.test_spec"),
    )
    monkeypatch.setitem(sys.modules, "swebench.harness.test_spec.test_spec", fake_test_spec_module)
    monkeypatch.setattr(
        "mcode.bench.swebench.lite._ensure_image",
        lambda client, name, **kwargs: None,
    )
    monkeypatch.setattr(
        "mcode.bench.swebench.lite._exec_agent_command_in_container",
        lambda *args, **kwargs: ("", 0, False),
    )

    sandbox = SWEbenchSandbox(cpu_limit=2.0)
    monkeypatch.setattr(sandbox, "_get_client", lambda: FakeClient())

    with sandbox.repo_context({"instance_id": "astropy__astropy-12907"}):
        pass

    # exec_container is the second create call. Cap = 2 cores → quota 200_000 / period 100_000.
    exec_kwargs = create_calls[1]
    assert exec_kwargs["cpu_period"] == 100_000
    assert exec_kwargs["cpu_quota"] == 200_000
    # Library-level OMP/BLAS thread caps must also be set so cgroup-v1
    # rootless podman silently dropping cpu_quota doesn't unleash 110-thread
    # pytest spikes.
    env = exec_kwargs["environment"]
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"


def test_sandbox_cpu_limit_zero_or_negative_treated_as_unlimited():
    s = SWEbenchSandbox(cpu_limit=0)
    assert s.cpu_limit is None
    assert s._cpu_kwargs() == {}
    s = SWEbenchSandbox(cpu_limit=-1.5)
    assert s.cpu_limit is None
    assert s._cpu_kwargs() == {}
    s = SWEbenchSandbox(cpu_limit=4.0)
    assert s.cpu_limit == 4.0
    assert s._cpu_kwargs() == {"cpu_period": 100_000, "cpu_quota": 400_000}
    # Near-zero positive: quota would round to <1ms, so we return {} rather
    # than send a half-set HostConfig.
    s = SWEbenchSandbox(cpu_limit=0.0001)
    assert s._cpu_kwargs() == {}
