#!/usr/bin/env bash
# Run one benchmark shard on a Blue Vela compute node.
#
# Invoked via bsub job array. Reads env.json next to this script.
# See plan "Shell scripts over string assembly" for the env contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_JSON="${SCRIPT_DIR}/env.json"

eval "$(jq -r 'to_entries[] | select(.value|type=="string") | "export \(.key)=\(.value | @sh)"' "$ENV_JSON")"
eval "$(jq -r '(.EXTRA_ENV // {}) | to_entries[] | "export \(.key)=\(.value | @sh)"' "$ENV_JSON")"

# TODO: podman setup (docker-compat socket), cd into $WORKSPACE,
# uv run mcode bench ... --shard "$LSB_JOBINDEX" --of "$SHARD_COUNT".
echo "[bluevela_shard] TODO shard=$LSB_JOBINDEX of=$SHARD_COUNT"
