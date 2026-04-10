# Blue Vela Deployment

Use the unified launcher. The shell scripts in this directory are fallback/debug tools, not the primary workflow.

## Quickstart

```bash
uv run mcode launch
```

Check access and sync state:

```bash
uv run mcode launch doctor --target bluevela
uv run mcode launch sync --target bluevela --check
```

Launch a run:

```bash
uv run mcode launch \
  --target bluevela \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-live \
  --parallelism 4 \
  --yes
```

Launch and follow logs:

```bash
uv run mcode launch \
  --target bluevela \
  --model Qwen/Qwen3.5-27B \
  --benchmark swebench-live \
  --parallelism 4 \
  --yes \
  --follow
```

If you run `swebench-lite` against a non-default Hugging Face dataset such as `princeton-nlp/SWE-bench_Verified`, pass `--dataset ...` on `launch` so task-id files are checked against the right slice.

Manage runs:

```bash
uv run mcode launch status --json
uv run mcode launch attach run-12345678
uv run mcode launch fetch run-12345678 --destination results
uv run mcode launch stop run-12345678
uv run mcode launch stop server-12345678
```

## Prerequisites

- SSH access to `login3.bluevela.rmf.ibm.com`
- Connected to IBM VPN

## Config

Optional per-user defaults:

```toml
[bluevela]
login = "your-user@login3.bluevela.rmf.ibm.com"
workspace_root = "/u/your-user/mcode-launch"
shared_root = "/proj/dmfexp/your-user"
hf_env = "/u/your-user/.config/mcode/hf-env.sh"
```

The launcher resolves defaults from `$USER`, including `/u/$USER/mcode-launch`, `/proj/dmfexp/$USER`, and `/u/$USER/.config/mcode/hf-env.sh`.

## Legacy Scripts

Keep the scripts in this directory for debugging or compatibility only:

- `setup.sh`
- `start-vllm.sh`
- `run-swebench-live.sh`
- `stop-vllm.sh`
- `fetch-results.sh`

For the full operational command set, use [`docs/COMMANDS.md`](../../docs/COMMANDS.md).
