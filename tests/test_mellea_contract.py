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
                "assert callable(start_session); "
                "assert callable(react); "
                "assert callable(MelleaTool.from_callable)"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
