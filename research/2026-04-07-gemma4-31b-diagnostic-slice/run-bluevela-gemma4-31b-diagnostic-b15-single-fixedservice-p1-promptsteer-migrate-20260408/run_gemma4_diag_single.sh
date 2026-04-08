#!/usr/bin/env bash
set -euo pipefail
ROOT=/u/skula/mcode-hashline-20260406
RUN_DIR=/u/skula/mcode-hashline-20260406/research/2026-04-07-gemma4-31b-diagnostic-slice/run-bluevela-gemma4-31b-diagnostic-b15-single-fixedservice-p1-promptsteer-migrate-20260408
GRAPHROOT=/proj/dmfexp/skula/podman/graphroot
RUNROOT=/tmp/podman-mcode-700438-b15-refresh/runroot
export XDG_RUNTIME_DIR=/tmp/podman-gemma4-diag1-${LSB_JOBID}/xdg
SOCK=$XDG_RUNTIME_DIR/podman.sock
mkdir -p "$GRAPHROOT" "$RUNROOT" "$XDG_RUNTIME_DIR"
rm -rf "$RUNROOT/networks/rootless-netns"
rm -f "$SOCK"
podman --cgroup-manager=cgroupfs --storage-driver=overlay --root="$GRAPHROOT" --runroot="$RUNROOT" system migrate >/dev/null 2>&1 || true
podman --cgroup-manager=cgroupfs --storage-driver=overlay --root="$GRAPHROOT" --runroot="$RUNROOT" rm -af >/dev/null 2>&1 || true
podman --cgroup-manager=cgroupfs --storage-driver=overlay --root="$GRAPHROOT" --runroot="$RUNROOT" system service --time=0 unix://"$SOCK" >"$RUN_DIR/podman.log" 2>&1 &
PODMAN_PID=$!
export DOCKER_HOST=unix://$SOCK
trap 'kill "$PODMAN_PID" 2>/dev/null; wait "$PODMAN_PID" 2>/dev/null || true' EXIT
cd "$ROOT"
source /u/skula/.config/mcode/hf-env.sh
for attempt in $(seq 1 60); do
  if uv run python - <<'PYEOF' >/dev/null 2>&1
import docker
client = docker.from_env()
client.ping()
PYEOF
  then
    break
  fi
  sleep 1
  if [[ $attempt -eq 60 ]]; then
    echo "Docker socket did not become ready" >&2
    exit 1
  fi
done
export OPENAI_BASE_URL=http://p2-r28-n2.bluevela.rmf.ibm.com:8331/v1
export OPENAI_API_KEY=dummy
export MCODE_MAX_NEW_TOKENS=4096
export MCODE_CONTEXT_WINDOW=32768
export MCODE_REACT_TIMEOUT=1800
export MCODE_KEEP_IMAGES=1
export MELLEA_BASH_TOOL=1
uv run mcode bench swebench-lite   --backend openai   --model google/gemma-4-31B-it   --dataset princeton-nlp/SWE-bench_Verified   --loop-budget 15   --timeout 300   --mem-limit 4g   --pids-limit 512   --n-samples 1   --task-ids research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt   --db "$RUN_DIR/diagnostic.db"
