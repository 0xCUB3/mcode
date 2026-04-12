# End-to-end verification (2026-04-12)

Live test of `minimal-launcher` branch against `login3.bluevela.rmf.ibm.com` as user `skula`, group `grp_runtime`.

## Summary

All four target models (Qwen3.5-35B-A3B, Granite 4.0 h-small, MiniMax-M2.5, Gemma-4-31B-it) booted to healthy HTTP endpoints on real Blue Vela compute nodes. Each then ran a four-task SWE-bench Live smoke test against its endpoint through `mcode bench`, exercising the full pipeline: LSF submit → podman pull → vLLM serve → OpenAI-compatible API → bench task sandbox → verification. Phase transitions, error UX, heartbeat backoff, and golden env.json fixtures all behaved as designed.

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

## Bugs found and fixed in live runs

Every one of these was caught only because we ran against a real account. All are now codified in the launcher, shell scripts, profiles, or `doctor --init`, so the next user does not re-encounter them.

1. **`_parse_bugroup` picked the wrong group.** `bugroup` (no args) lists every group on the cluster with its members; we were grabbing the first row (`lsfadmins`, a catch-all) regardless of membership. Fixed by adding a `user=` filter that does whole-word member matching.
2. **Queue auto-selection included `interactive`.** On this cluster the `interactive` queue is `Open:Active` but has the `ONLY_INTERACTIVE` scheduling policy, rejecting batch submits. Fixed by probing each candidate with `bqueues -l <q>` and filtering. `doctor --init` now fails closed if every policy probe errored, rather than silently falling back.
3. **`_validate_queue` timeout was 30 s.** We disable SSH ControlPath multiplexing, so each validation pays the full connect + auth cost. Three queues × two calls over VPN hit 30 s. Bumped to 60 s.
4. **Gemma4 chat template was missing from the repo.** Fixed by fetching the official template from vLLM upstream and committing it under `src/mcode/launch/resources/`.
5. **Home quota blew up.** Per-host podman graphroots on `/u/<user>` accumulated ~20 GB each; a few runs blew the 100 GB home quota and broke later pulls mid-unpack. Fixed by moving podman graphroots in `bluevela_vllm.sh` to per-job `/tmp` (node-local), so each launch starts with a clean isolated store.
6. **Per-host graphroot caused podman "database configuration mismatch".** The change above also fixed this: podman's persistent DB inside the old shared graphroot remembered the previous job's runroot and refused a second launch on the same host.
7. **Gemma4 vLLM flag churn.** The `gemma4` tool parser doesn't exist in vLLM 0.17 — `functiongemma` is the right name. `--limit-mm-per-prompt` changed to JSON syntax in 0.17. vLLM 0.17's bundled Transformers predates the `gemma4` arch entirely; the dedicated `docker.io/vllm/vllm-openai:gemma4` tag (2026-04-10) has both the arch and a new-enough Transformers. Head dim 512 is not supported by `FLASH_ATTN`, `FLASHINFER`, or `TORCH_SDPA` on this image; `FLEX_ATTENTION` works. All locked in the Gemma4 profile.
8. **Startup deadline tuning.** Per-job graphroots guarantee cold image pulls every launch. Bumped the absolute startup deadline from 1800 s to 2400 s to keep headroom.
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
