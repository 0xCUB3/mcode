#!/usr/bin/env bash
# 16-task SWE-bench Verified diagnostic slice, reusable across models.
# Task list: astropy smoke (10) + django/matplotlib/pylint/sklearn/sphinx/sympy (6).
# Originally from research/2026-04-03-adapter-aware-orchestrator-contract/.
#
# Usage:
#   scripts/smoke-16.sh <hf-model-id> [db-path]
#   scripts/smoke-16.sh ibm-granite/granite-4.0-h-small
#   scripts/smoke-16.sh Qwen/Qwen3.5-35B-A3B /tmp/qwen.db
#
# Endpoint resolution:
#   Uses OPENAI_BASE_URL / OPENAI_API_KEY if set, otherwise auto-resolves
#   from `mcode launch status` for a healthy server matching the model.

set -euo pipefail

MODEL="${1:?usage: scripts/smoke-16.sh <model> [db]}"
DB="${2:-experiments/results/smoke-16-$(echo "$MODEL" | tr '/' '-').db}"
TASK_IDS="research/2026-04-03-adapter-aware-orchestrator-contract/medium-diagnostic-task-ids.txt"

if [[ ! -f "$TASK_IDS" ]]; then
  echo "task-id file not found: $TASK_IDS" >&2
  exit 1
fi

mkdir -p "$(dirname "$DB")"
rm -f "$DB"

exec uv run mcode bench swebench-lite \
  --backend openai \
  --model "$MODEL" \
  --dataset princeton-nlp/SWE-bench_Verified \
  --task-ids "$TASK_IDS" \
  --loop-budget 15 \
  --timeout 300 \
  --db "$DB"
