# Blue Vela Deployment

For new work, **use `uv run mcode launch` from the repo root** instead of these scripts. See the root `README.md`. The scripts here are the legacy path kept as a fallback.

## Prerequisites

- SSH access to a Blue Vela login host (e.g. `login3.bluevela.rmf.ibm.com`)
- Connected to IBM VPN
- LSF group membership (e.g. `grp_runtime`)

## Preferred workflow: `uv run mcode launch`

```bash
# One-time: write ~/.config/mcode/launch.toml
uv run mcode launch doctor bluevela --init --login <user>@<login-host>

# Launch a server + run bench against it
uv run mcode launch bluevela --model Qwen/Qwen3.6-35B-A3B
uv run mcode bench swebench-live --backend openai --model Qwen/Qwen3.6-35B-A3B --limit 10
MCODE_CONTEXT_WINDOW=262144 uv run mcode bench swebench-lite --backend openai --model Qwen/Qwen3.6-35B-A3B --on bluevela --limit 10
uv run mcode bench swebench-live --backend openai --model Qwen/Qwen3.6-35B-A3B --sampling multiturn --n-samples 3 --limit 10
MCODE_CONTEXT_WINDOW=262144 uv run mcode bench smoke --backend openai --model Qwen/Qwen3.6-35B-A3B --on bluevela --shards 4

# Stop when done
uv run mcode launch stop --all
```

## Legacy shell-script workflow

## Config

Optional per-user defaults:

```toml
[bluevela]
login = "your-user@login3.bluevela.rmf.ibm.com"
workspace_root = "/u/your-user/mcode-launch"
queue = "auto"
shared_root = "/proj/dmfexp/your-user"
hf_env = "/u/your-user/.config/mcode/hf-env.sh"
```

The launcher resolves defaults from `$USER`, including `/u/$USER/mcode-launch`, `/proj/dmfexp/$USER`, and `/u/$USER/.config/mcode/hf-env.sh`. With `queue = "auto"`, it probes Blue Vela and picks the least loaded open, batch-capable queue among the highest-priority queues you can use. Set a concrete queue name if you need to pin one.

## Legacy Scripts

Keep the scripts in this directory for debugging or compatibility only:

- `setup.sh`
- `start-vllm.sh`
- `run-swebench-live.sh`
- `stop-vllm.sh`
- `fetch-results.sh`

For the current operational command set, use the root [README](../../README.md).
