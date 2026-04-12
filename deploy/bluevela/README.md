# Blue Vela Deployment

For new work, **use `mcode launch` from the repo root** instead of these scripts — see the root `README.md`. The scripts here are the legacy path kept as a fallback.

## Prerequisites

- SSH access to a Blue Vela login host (e.g. `login3.bluevela.rmf.ibm.com`)
- Connected to IBM VPN
- LSF group membership (e.g. `grp_runtime`)

## Preferred workflow: `mcode launch`

```bash
# One-time: write ~/.config/mcode/launch.toml
mcode launch doctor bluevela --init --login <user>@<login-host>

# Launch a server + run bench against it
mcode launch bluevela --model Qwen/Qwen3.5-35B-A3B
OPENAI_BASE_URL=$(mcode launch status --json | jq -r '.servers[0].endpoint') \
OPENAI_API_KEY=dummy \
mcode bench swebench-live --backend openai --model Qwen/Qwen3.5-35B-A3B --limit 10

# Stop when done
mcode launch stop --all
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

For the full operational command set, use [`docs/COMMANDS.md`](../../docs/COMMANDS.md).
