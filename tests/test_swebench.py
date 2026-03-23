from __future__ import annotations

from mcode.execution.swebench import (
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
