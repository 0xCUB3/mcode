"""Pin selection logic so the two reporter stacks (mcode.ui.task_reporter and
mcode.launch.progress) don't drift.

Both pick reporter type from the same env signals (json_mode + isatty(stream))
and both fall back to Plain when Rich construction fails. We don't compare
class identities (they're separate) — we compare the abstract category
(json/rich/plain) each chooser picks for an identical input.
"""

from __future__ import annotations

import io

from mcode.launch import progress as launch_progress
from mcode.launch.models import Phase
from mcode.ui import task_reporter as ui_reporter


def _ui_category(r: object) -> str:
    if isinstance(r, ui_reporter.JsonReporter):
        return "json"
    if isinstance(r, ui_reporter.RichReporter):
        return "rich"
    if isinstance(r, ui_reporter.PlainReporter):
        return "plain"
    return type(r).__name__


def _launch_category(r: object) -> str:
    if isinstance(r, launch_progress.JsonReporter):
        return "json"
    if isinstance(r, launch_progress.RichReporter):
        return "rich"
    if isinstance(r, launch_progress.PlainReporter):
        return "plain"
    return type(r).__name__


PHASES = [Phase(key="x", label="X"), Phase(key="y", label="Y")]


def test_drift_json_mode_both_pick_json():
    buf = io.StringIO()
    a = ui_reporter.choose(json_mode=True, stream=buf)
    b = launch_progress.choose(PHASES, json_mode=True, stream=buf)
    assert _ui_category(a) == _launch_category(b) == "json"


def test_drift_non_tty_both_pick_plain():
    buf = io.StringIO()
    a = ui_reporter.choose(json_mode=False, stream=buf)
    b = launch_progress.choose(PHASES, json_mode=False, stream=buf)
    assert _ui_category(a) == _launch_category(b) == "plain"
