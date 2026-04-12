#!/usr/bin/env bash
# Fetch results for a completed run.
#
# Per plan M3: refuses unless run is in a terminal state, or --snapshot is
# requested (in which case this script copies rundir to a sibling snapshot
# dir via `cp --reflink=auto` first).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_JSON="${SCRIPT_DIR}/env.json"
eval "$(jq -r 'to_entries[] | select(.value|type=="string") | "export \(.key)=\(.value | @sh)"' "$ENV_JSON")"

# TODO: if SNAPSHOT=1, cp --reflink=auto "$RUN_DIR" "$RUN_DIR.snapshot-$(date +%s)"
# then rsync that to $DEST. Else rsync $RUN_DIR directly.
echo "[bluevela_fetch] TODO RUN_DIR=$RUN_DIR DEST=$DEST SNAPSHOT=${SNAPSHOT:-0}"
