#!/usr/bin/env bash
set -euo pipefail

MCODE_DIR="${BV_MCODE_DIR:-/u/skula/mcode}"
REPO_URL="${MCODE_REPO:-https://github.com/0xCUB3/mcode.git}"

echo "=== Installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv $(uv --version)"

echo "=== Setting up mcode ==="
if [[ ! -d "${MCODE_DIR}" ]]; then
  git clone "${REPO_URL}" "${MCODE_DIR}"
fi
cd "${MCODE_DIR}"

echo "=== Creating virtualenv with Python 3.11 ==="
uv venv --python 3.11 venv
source venv/bin/activate

echo "=== Installing mcode ==="
uv pip install -e ".[evalplus,datasets]"

echo "=== Creating results directory ==="
mkdir -p results

echo "=== Done ==="
echo "Activate with: source ${MCODE_DIR}/venv/bin/activate"
