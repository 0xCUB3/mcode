from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

from mcode.ui.dashboard import Dashboard


def test_json_mode_emits_one_event_per_line_with_monotonic_seq(tmp_path: Path):
    buf = io.StringIO()
    db_a = tmp_path / "a.db"
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    d = Dashboard(mode="json", total_shards=2, benchmark="smoke", model="test", stream=buf)
    with d:
        d.post("shard_start", shard=0, db=str(db_a), log=str(log_a))
        d.post("shard_start", shard=1, db=str(tmp_path / "b.db"), log=str(log_b))
        d.post("shard_stdout", shard=0, line="task-1 ok")
        d.post("shard_stdout", shard=1, line="task-1 fail")
        d.post("shard_done", shard=0)
        d.post("shard_failed", shard=1, rc=1, log=str(log_b))

    seqs: list[int] = []
    kinds: list[str] = []
    for line in buf.getvalue().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        seqs.append(obj["seq"])
        kinds.append(obj["kind"])
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert {"run_start", "shard_start", "shard_done", "shard_failed"} <= set(kinds)


def test_plain_mode_preserves_expected_line_shapes(tmp_path: Path):
    buf = io.StringIO()
    db_a = tmp_path / "a.db"
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    out_db = tmp_path / "out.db"
    d = Dashboard(mode="plain", total_shards=2, benchmark="smoke", model="t", stream=buf)
    with d:
        d.post("shard_start", shard=0, db=str(db_a), log=str(log_a))
        d.post("shard_stdout", shard=0, line="hello")
        d.post("shard_done", shard=0)
        d.post("shard_failed", shard=1, rc=2, log=str(log_b))
        d.post("merged", db=str(out_db))

    text = buf.getvalue()
    assert "▶ run benchmark=" not in text
    assert f"▶ shard 1/2 db={db_a} log={log_a}" in text
    assert "[shard 0] hello" in text
    assert "✓ shard 1/2 finished" in text
    assert f"✗ shard 2/2 failed exit=2 log={log_b}" in text
    assert f"✓ merged shards into {out_db}" in text


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
    assert "line-49" in seen


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
        time.sleep(0.1)

    seqs = [json.loads(line)["seq"] for line in buf.getvalue().splitlines() if line.strip()]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_writer_thread_is_daemon_and_joins():
    buf = io.StringIO()
    d = Dashboard(mode="plain", total_shards=1, benchmark="x", model="m", stream=buf)
    with d:
        d.post("shard_start", shard=0, db="x", log="y")
    assert d._writer is not None
    assert not d._writer.is_alive()
