"""TaskReporter API + selection."""

from __future__ import annotations

import io
import json

from mcode.ui.task_reporter import (
    JsonReporter,
    NullReporter,
    PlainReporter,
    choose,
)


def test_null_reporter_drops_everything():
    r = NullReporter()
    with r:
        r.total(10)
        r.advance(detail="x")
        r.event("ok", "yay")
        r.finish(ok=True, summary="done")


def test_json_reporter_emits_one_line_per_event_with_monotonic_seq():
    buf = io.StringIO()
    r = JsonReporter(stream=buf)
    with r:
        r.total(3)
        r.advance(detail="task-1")
        r.advance(detail="task-2")
        r.event("warn", "slow")
        r.finish(ok=True, summary="done")

    seqs: list[int] = []
    for line in buf.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        seqs.append(obj["seq"])
        assert "kind" in obj and "ts" in obj
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_plain_reporter_renders_human_lines():
    buf = io.StringIO()
    r = PlainReporter(stream=buf)
    with r:
        r.total(2)
        r.advance(detail="a")
        r.advance(detail="b")
        r.event("ok", "first done")
        r.finish(ok=True, summary="2 passed")

    text = buf.getvalue()
    assert "total: 2" in text
    assert "[1/2]" in text
    assert "[2/2]" in text
    assert "first done" in text
    assert "2 passed" in text


def test_choose_picks_json_when_json_mode_set():
    buf = io.StringIO()
    r = choose(json_mode=True, stream=buf)
    assert isinstance(r, JsonReporter)


def test_choose_picks_plain_for_non_tty_stream():
    buf = io.StringIO()
    r = choose(json_mode=False, stream=buf)
    assert isinstance(r, PlainReporter)
