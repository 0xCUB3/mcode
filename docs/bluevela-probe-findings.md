# Blue Vela Phase 0.5 probe findings (2026-04-12)

Probed from `login3.bluevela.rmf.ibm.com` as user `skula` in group `grp_runtime`. Full sanitized output: `tests/launch/fixtures/bluevela_probe.json`.

These findings override the original plan's hypotheses.

## GPU reservation mode

| Mode | bsub response |
|-|-|
| `num=N:mode=exclusive_process` | Accepted cleanly. `"GPU mode=exclusive_process specified. No change"` |
| `num=N:mode=shared` | Accepted but **deprecated**. `"GPU mode=shared. This is allowed but deprecated"` |

**Decision:** default `[bluevela.gpu_mode] = "exclusive_process"`. The current `deploy/bluevela/start-vllm.sh` uses `shared` and still works but is on the deprecated path. The rewrite should lead users away from it.

## Queues visible to skula / grp_runtime

| Queue | Status | Priority | Notes |
|-|-|-|-|
| `owners` | Closed:Inact | 43 | Highest priority but inactive |
| `night` | Closed:Inact | 40 | Inactive |
| `short` | Closed:Inact | 35 | Inactive |
| `normal` | Open:Active | 30 | **Default.** 933 PEND / 824 RUN at probe time. Batch OK. |
| `interactive` | Open:Active | 30 | **Interactive only.** Rejects batch `bsub`. |
| `preemptable` | Open:Active | 20 | 54 PEND / 946 RUN. **Requires `-G grp_preemptable`** — skula doesn't have access. |
| `idle` | Closed:Inact | 20 | Inactive |
| `preemptable_tes` | Open:Active | 20 | Requires different group presumably. |

**Default `queue_order`:** `["normal"]` for `grp_runtime` users. The plan's B4 queue-auto logic (config preference list + `bsub -H` hold validation) is correct — the hold-validate round-trip is fast (< 1 s per queue). Users with access to `grp_preemptable` can add `preemptable` to their queue_order.

## spjb front-end tools

Not installed on login3:

- `jbmon`: **no**
- `jbsub`: **no**
- `jbinfo`: **no**
- `bugroup`: yes
- `bsub`, `bjobs`, `bkill`, `bqueues`, `bhosts`, `lsid`: yes

**Decision:** the launcher must not depend on spjb tools. `doctor --deep`'s queue-headroom hint ("queue X has N free GPU slots") has to come from parsing `bqueues -l <q>` + `bhosts` directly, not `jbmon`.

## Submission mechanics

- `bsub -H` (hold) + `bkill` is a fast, free validation round-trip. PSUSP'd jobs consume no resources. Confirms plan B4's approach.
- `bsub -interactive` from an SSH session without a TTY likely fails — need to allocate with `ssh -t` or skip interactive-mode connectivity tests.
- `normal` queue accepts `-W <minutes>` walltime limits.

## Login host

Real hostname is `login3.bluevela.rmf.ibm.com`, not a generic `login.rmf...` round-robin alias.

## Not yet verified (deferred)

- Podman rootless works on compute nodes — working `deploy/bluevela/start-vllm.sh` is evidence it does, but probe didn't re-test.
- HF cache / `hf-env.sh` location on skula's account — defer to `doctor --init`.
- Port 8321 vs 8000 availability on compute nodes — `deploy/bluevela/env.sh` uses 8321 as default; not re-probed.
- Full end-to-end `bsub -H -q normal -gpu ...` with HF_TOKEN+podman is out of scope for the probe — goes in the matrix smoke test once `bluevela.py` is built.
