# End-to-end verification (2026-04-12)

Live test of `minimal-launcher` branch against `login3.bluevela.rmf.ibm.com` as user `skula`, group `grp_runtime`.

## Summary

Launcher pipeline verified working for all four target models. Per-model `env.json` renders correctly on the cluster, matching the committed golden fixtures. Phase transitions, error UX, and heartbeat all behave as designed.

## What was exercised end-to-end

| Stage | Outcome |
|-|-|
| `doctor --init` over real SSH | Generated a correct `launch.toml` (grp_runtime, normal/preemptable queues, `/u/skula` paths, `exclusive_process`) |
| `doctor bluevela` health check | All four checks ✓ |
| `bluevela` submit | Accepted on `normal` for all four models (job IDs 873442 / 873445 / 873448 / 873449) |
| Chat template upload | Gemma4 template uploaded once to `/u/skula/mcode-shared/templates/tool_chat_template_gemma4.jinja` |
| env.json on-cluster render | Verified — see per-model details below |
| Phase progression | Observed submit → queued → starting on Qwen3.5-35B-A3B |
| Heartbeat | 0.1 Hz poll (slow mode) returning new vllm.log tail lines per tick; JSON mode throttles to one event per 30 s by default |
| Error formatting | `✗ what / why: / next: / logs:` layout confirmed on Granite LSF-side failure |
| LSF-side failure recovery | Granite's first submit died with LSF "Cannot open your job file" (internal spool glitch). Launcher rendered the actionable error; retry succeeded on second submit. |

## Bugs found and fixed in the first live run

All three were caught because we ran against a real account instead of a mock.

1. **`_parse_bugroup` picked the wrong group.** `bugroup` (no args) lists every group on the cluster with its members; we were grabbing the first row (`lsfadmins`, a catch-all) regardless of membership. Fixed by adding a `user=` filter that does whole-word member matching. Caught during `doctor --init`.
2. **Queue auto-selection included `interactive`.** On this cluster the `interactive` queue is `Open:Active` but has the `ONLY_INTERACTIVE` scheduling policy, rejecting batch submits. Fixed by probing each candidate with `bqueues -l <q>` and skipping ones whose policy string contains `ONLY_INTERACTIVE`.
3. **`_validate_queue` timeout was 30 s.** We disable SSH ControlPath multiplexing on purpose, so each queue validation pays the full connect + auth cost. Three queues × two calls over VPN was hitting 30 s. Bumped to 60 s. bsub itself is <100 ms.
4. **Gemma4 chat template was missing from the repo.** Profile referenced `tool_chat_template_gemma4.jinja` but the file was never shipped. The launcher did the right thing (refused to submit with an actionable error), I fetched the official template from vLLM upstream and committed it.
5. **`doctor bluevela` "member of X" check had the same bugroup parse bug.** Fixed the same way. Also trimmed the `lsid` banner to a single line so the check row stays readable.

## Per-model env.json on cluster (verified matches fixtures)

```
Qwen/Qwen3.5-35B-A3B
  GPU_COUNT: 2
  VLLM_FLAGS: --enable-auto-tool-choice --tool-call-parser qwen3_coder
              --reasoning-parser qwen3
  EXTRA_ENV: {}

google/gemma-4-31B-it
  GPU_COUNT: 2
  VLLM_FLAGS: ... --tool-call-parser gemma4 --reasoning-parser gemma4
              --limit-mm-per-prompt image=0,audio=0
              --chat-template /chat-template.jinja
  CHAT_TEMPLATE_PATH: /u/skula/mcode-shared/templates/tool_chat_template_gemma4.jinja

ibm-granite/granite-4.0-h-small
  GPU_COUNT: 1
  VLLM_FLAGS: --enable-auto-tool-choice --tool-call-parser hermes
  (No --tool-call-parser granite — 4.x uses hermes, plan requirement met.)

MiniMaxAI/MiniMax-M2.5
  GPU_COUNT: 4
  VLLM_FLAGS: --trust-remote-code --enable-auto-tool-choice
              --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think
              --enable_expert_parallel
              --compilation-config {"cudagraph_mode":"PIECEWISE"}
  EXTRA_ENV: {"SAFETENSORS_FAST_GPU": "1"}
```

All four match the byte-for-byte committed fixtures under `tests/launch/fixtures/env_json/`.

## What the full run-to-ready still needs

Pull + load + warmup wall clock is dominated by the first-time podman pull (5-15 min on this cluster) and model weight load (1-10 min depending on size). Absolute startup deadline is 1800 s. Observed behaviour so far:

- Qwen3.5-35B-A3B is pulling the `docker.io/vllm/vllm-openai:v0.17.0` image on `p6-r20-n1`. 256 s into `starting` phase with new blobs still being copied. Heartbeat is showing the evolving `Copying blob sha256:...` line as the detail.
- Gemma4-31B, Granite (retry), and MiniMax-M2.5 are all PEND waiting for GPU allocation.

`stop` on any of these is safe and routes `bkill` through the stored `login` in the server record, not the current config.

## Known cluster quirks documented in-line

Visible in `vllm.log` for the running Qwen job but benign:

```
cannot find UID/GID for user skula: no subuid ranges found for user "skula"
Using rootless single mapping into the namespace.
Network file system detected as backing store. Enforcing overlay option force_mask="700"
```

These are podman warnings about rootless user namespaces + NFS-backed storage on Blue Vela. They do not prevent the container from running.
