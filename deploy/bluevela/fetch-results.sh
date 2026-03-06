#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

LOCAL_RESULTS="$(cd "${SCRIPT_DIR}/../.." && pwd)/results"
mkdir -p "${LOCAL_RESULTS}"

echo "Fetching results from Blue Vela..."
rsync -avz --progress \
  "${BV_LOGIN}:${BV_RESULTS_DIR}/*.db" \
  "${LOCAL_RESULTS}/"

echo "Results saved to: ${LOCAL_RESULTS}/"
ls -la "${LOCAL_RESULTS}"/*.db 2>/dev/null || echo "No .db files found."
