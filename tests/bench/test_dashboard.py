"""Dashboard writer thread + event ordering."""

from __future__ import annotations

import io
import json
import threading
import time

from mcode.ui.dashboard import Dashboard


def _drain(dashboard: Dashboard) -> None:
    """Wait until the writer thread has finished consuming."""
    dashboard.close()


def test_json_mode_emits_one_event_per_line_with_monotonic_seq():
    buf = io.StringIO()
    d = Dashboard(mode="json", total_shards=2, benchmark="smoke", model="test", stream=buf)
    with d:
        d.post("shard_start", shard=0, db="/tmp/a.db", log="/tmp/a.log")
        d.post("shard_start", shard=1, db="/tmp/b.db", log="/tmp/b.log")
        d.post("shard_stdout", shard=0, line="task-1 ok")
        d.post("shard_stdout", shard=1, line="task-1 fail")
        d.post("shard_done", shard=0)
        d.post("shard_failed", shard=1, rc=1, log="/tmp/b.log")

    seqs: list[int] = []
    kinds: list[str] = []
    for line in buf.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        seqs.append(obj["seq"])
        kinds.append(obj["kind"])
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert "run_start" in kinds
    assert "shard_start" in kinds
    assert "shard_done" in kinds
    assert "shard_failed" in kinds


def test_plain_mode_preserves_expected_line_shapes():
    buf = io.StringIO()
    d = Dashboard(mode="plain", total_shards=2, benchmark="smoke", model="t", stream=buf)
    with d:
        d.post("shard_start", shard=0, db="/tmp/a.db", log="/tmp/a.log")
        d.post("shard_stdout", shard=0, line="hello")
        d.post("shard_done", shard=0)
        d.post("shard_failed", shard=1, rc=2, log="/tmp/b.log")
        d.post("merged", db="/tmp/out.db")

    text = buf.getvalue()
    # Format-equivalent (not byte-equivalent) line shapes. Plain mode skips
    # run_start because the orchestrator emits the equivalent "▶ sharded run"
    # info line itself for byte-compatibility with pre-Wave-2 output.
    assert "▶ run benchmark=" not in text
    assert "▶ shard 1/2 db=/tmp/a.db log=/tmp/a.log" in text
    assert "[shard 0] hello" in text
    assert "✓ shard 1/2 finished" in text
    assert "✗ shard 2/2 failed exit=2 log=/tmp/b.log" in text
    assert "✓ merged shards into /tmp/out.db" in text


def test_close_drains_pending_events():
    buf = io.StringIO()
    d = Dashboard(mode="json", total_shards=1, benchmark="x", model="m", stream=buf)
    d.__enter__()
    for i in range(50):
        d.post("shard_stdout", shard=0, line=f"line-{i}")
    d.close()

    seen = [
        json.loads(line)["data"].get("line", "")
        for line in buf.getvalue().splitlines()
        if line.strip() and "shard_stdout" in line
    ]
    assert "line-49" in seen, "writer thread should drain pending events on close"


def test_post_is_safe_from_many_threads():
    buf = io.StringIO()
    d = Dashboard(mode="json", total_shards=4, benchmark="x", model="m", stream=buf)
    with d:

        def producer(idx: int) -> None:
            for i in range(20):
                d.post("shard_stdout", shard=idx, line=f"thread-{idx}-line-{i}")

        ts = [threading.Thread(target=producer, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # Give writer a moment to drain.
        time.sleep(0.1)

    # Strictly monotonic seq across threads.
    seqs = [json.loads(line)["seq"] for line in buf.getvalue().splitlines() if line.strip()]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_writer_thread_is_daemon_and_joins():
    buf = io.StringIO()
    d = Dashboard(mode="plain", total_shards=1, benchmark="x", model="m", stream=buf)
    with d:
        d.post("shard_start", shard=0, db="x", log="y")
    # After close, writer thread should not be alive.
    assert d._writer is not None
    assert not d._writer.is_alive()
