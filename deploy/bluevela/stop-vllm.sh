#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Killing vLLM server job..."
bkill -J "vllm-server" 2>/dev/null && echo "Done." || echo "No running vLLM job found."
rm -f "${BV_MCODE_DIR}/vllm_host.txt"
