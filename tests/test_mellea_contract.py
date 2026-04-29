from __future__ import annotations

import subprocess
import sys


def test_installed_mellea_exposes_native_solver_surface():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util; "
                "from mellea import start_session; "
                "from mellea.stdlib.frameworks.react import react; "
                "from mellea.backends.tools import MelleaTool; "
                "from mellea.plugins.pluginset import PluginSet; "
                "from mellea.stdlib.requirements import Requirement, simple_validate, uses_tool; "
                "from mellea.stdlib.sampling import MultiTurnStrategy; "
                "from mcode.bench.compare import compare_runs; "
                "from mcode.mellea_compat import apply_provider_compatibility_patches; "
                "assert callable(start_session); "
                "assert callable(react); "
                "assert callable(MelleaTool.from_callable); "
                "assert PluginSet is not None; "
                "assert Requirement is not None; "
                "assert callable(simple_validate); "
                "assert callable(uses_tool); "
                "assert MultiTurnStrategy is not None; "
                "assert callable(compare_runs); "
                "assert callable(apply_provider_compatibility_patches); "
                "assert importlib.util.find_spec('mellea.agent') is None"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
