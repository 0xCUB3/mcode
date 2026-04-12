#!/usr/bin/env bash
# Launch the vLLM server on a Blue Vela compute node.
#
# Invoked via bsub from bluevela.py. Reads env.json (next to this script) via
# `jq @sh` — never receives config via Python-interpolated env or args.
# Writes the compute-node hostname to $RUN_DIR/vllm_host.txt so the launcher
# can discover the endpoint. Writes vLLM stdout/stderr to $RUN_DIR/vllm.log.
#
# Derivations (per plan B5): graphroot per host (shared across jobs),
# runroot per-job-index, both inside this script.

set -euo pipefail

# env.json contract — see plan "Shell scripts over string assembly":
#   MODEL, VLLM_IMAGE, VLLM_FLAGS[], EXTRA_ENV{}, CHAT_TEMPLATE_PATH,
#   VLLM_PORT, GPU_COUNT, RUN_DIR, BV_SHARED_DIR
#
# The builder is responsible for injecting `--chat-template` into VLLM_FLAGS
# when CHAT_TEMPLATE_PATH is set — this script only mounts the file.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_JSON="${SCRIPT_DIR}/env.json"

if [[ ! -f "$ENV_JSON" ]]; then
  echo "ERROR: env.json not found next to $0" >&2
  exit 1
fi

# Load scalar string keys as exports.
eval "$(jq -r 'to_entries[] | select(.value|type=="string") | "export \(.key)=\(.value | @sh)"' "$ENV_JSON")"
# Load EXTRA_ENV map (string -> string).
eval "$(jq -r '(.EXTRA_ENV // {}) | to_entries[] | "export \(.key)=\(.value | @sh)"' "$ENV_JSON")"
# Load VLLM_FLAGS array.
readarray -t VLLM_FLAGS_ARR < <(jq -r '.VLLM_FLAGS[]' "$ENV_JSON")

mkdir -p "$RUN_DIR"
hostname > "$RUN_DIR/vllm_host.txt"
HOST_TAG="$(hostname -s)"

# Podman storage roots — BOTH per-job.
#
# Earlier iterations used a per-host shared graphroot under $BV_SHARED_DIR
# to benefit from image layer caching across runs. In practice, podman's
# persistent DB inside the graphroot remembers the last job's runroot and
# every subsequent job on the same host fails with:
#
#    Error: database run root "/tmp/.../runroot-A" does not match our run
#    root "/tmp/.../runroot-B": database configuration mismatch
#
# Node-local per-job graphroot fully isolates each launch. Cost: image
# re-pull on every run (~5-10 min on this cluster). Benefit: reliability.
# This matches the smoke-test script's approach.
export XDG_RUNTIME_DIR="/tmp/podman-run-$(id -u)-${LSB_JOBID:-0}"
mkdir -p "$XDG_RUNTIME_DIR"
GRAPHROOT="${XDG_RUNTIME_DIR}/graphroot"
RUNROOT="${XDG_RUNTIME_DIR}/runroot"
mkdir -p "$GRAPHROOT" "$RUNROOT"

# Optional HF env file (HF_TOKEN etc.).
if [[ -n "${BV_HF_ENV:-}" && -f "$BV_HF_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$BV_HF_ENV"
fi

# Build podman --env pass-through for EXTRA_ENV keys.
PODMAN_ENV_ARGS=()
while IFS= read -r key; do
  PODMAN_ENV_ARGS+=("--env" "$key")
done < <(jq -r '(.EXTRA_ENV // {}) | keys[]' "$ENV_JSON")

# Chat template mount (Gemma4 etc.). The builder also appends
# --chat-template /chat-template.jinja to VLLM_FLAGS.
CHAT_TEMPLATE_ARGS=()
if [[ -n "${CHAT_TEMPLATE_PATH:-}" && -f "$CHAT_TEMPLATE_PATH" ]]; then
  CHAT_TEMPLATE_ARGS+=("--mount" "type=bind,src=${CHAT_TEMPLATE_PATH},dst=/chat-template.jinja,ro")
fi

echo "[bluevela_vllm] host=$HOST_TAG model=$MODEL port=$VLLM_PORT gpu=$GPU_COUNT"
nvidia-smi -L || true

exec podman --cgroup-manager=cgroupfs --storage-driver=overlay \
  --root "$GRAPHROOT" --runroot "$RUNROOT" \
  run --rm \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --ipc=host \
    --net=host \
    --storage-opt ignore_chown_errors=true \
    "${PODMAN_ENV_ARGS[@]}" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    "$VLLM_IMAGE" \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$GPU_COUNT" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
    --trust-remote-code \
    "${VLLM_FLAGS_ARR[@]}"
