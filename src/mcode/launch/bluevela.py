"""Blue Vela LSF target.

Public surface:

    launch(spec: LaunchSpec, reporter: Reporter) -> RunRecord
    doctor(cfg: BlueVelaConfig, *, deep: bool = False) -> list[Check]
    doctor_init(cfg_path: Path) -> BlueVelaConfig
    stop(record_id: str) -> bool
    fetch(record_id: str, dest: Path, *, snapshot: bool = False) -> Path
    refresh(record: RunRecord) -> RunRecord

Phases (see plan "Phase-based progress UI"):

    submit   → Submit vLLM job to LSF
    queued   → Waiting in queue (bjobs state is PEND)
    starting → Container pull + model load + warmup (single phase, per M1)
    ready    → Server healthy (vllm_host.txt present + HTTP /v1/models = 200)
    shards   → Benchmark shards submitted

Non-negotiables (see plan "Blue Vela LSF — operational rules"):

- Login nodes only for bsub/bjobs/bpeek/bkill. Never ssh to compute nodes.
- `-G <group>` on every bsub. Empty group is a hard error.
- `stop --all` scopes to the caller's recorded runs; never bkill 0 / bkill -u.
- vLLM runs only in podman, never on the bare node.

Non-negotiables (see plan Codex revisions):

- No automatic server reuse in v1. `--reuse-server <id>` required, with full
  config-hash match (B1).
- Every mutating CLI command runs refresh(record) first unless --offline (B3).
- Queue "auto" requires config-declared queue_order + bsub -H hold validation (B4).
- Graphroot per host, runroot per-job-index — derived in the shell script, not
  in Python (B5).
- Adaptive backoff on polling: 2 Hz -> 0.5 Hz -> 0.1 Hz (B6).
- Env delivery via env.json + jq @sh. No export KEY=VALUE string assembly (B7).
- Absolute startup deadline (default 1800 s) in addition to stall timeout
  (don't-regress note from pass 3).

Failure catalog: list[tuple[re.Pattern, list[Hint]]] — regex-matched; multiple
hints per pattern labeled "try first / if that fails / last resort". No
destructive hints.
"""

from __future__ import annotations
