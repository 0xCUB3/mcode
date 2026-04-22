from __future__ import annotations

import subprocess
import sys


def test_installed_mellea_exposes_native_solver_surface():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from mellea import start_session; "
                "from mellea.stdlib.frameworks.react import react; "
                "from mellea.backends.tools import MelleaTool; "
                "from mellea.plugins.pluginset import PluginSet; "
                "from mcode.bench.compare import compare_runs; "
                "from mcode.mellea_compat import import_requirements, import_sampling; "
                "assert callable(start_session); "
                "assert callable(react); "
                "assert callable(MelleaTool.from_callable); "
                "assert PluginSet is not None; "
                "assert callable(compare_runs); "
                "assert import_requirements() is not None; "
                "assert import_sampling() is not None"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
