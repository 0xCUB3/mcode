#!/usr/bin/env bash
set -euo pipefail

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: 'oc' is required (OpenShift CLI)." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is required." >&2
  exit 2
fi

usage() {
  cat <<'USAGE' >&2
Usage:
  ./deploy/k8s/run-suite.sh

This runs an automated benchmark suite on OpenShift:
- HumanEval + MBPP as sharded Jobs (run-bench.sh)
- SWE-bench Lite as one-Pod-per-instance batch (run-swebench-lite.sh)

Env vars (optional):
  MODEL=...                Default: granite4
  OLLAMA_HOST=...          Default: http://ollama:11434
  PARALLELISM=...          Default: 2 (shards in flight for HumanEval/MBPP)

  # Sharding defaults (can be overridden per run in the script below)
  HUMANEVAL_SHARDS=...     Default: 10
  MBPP_SHARDS=...          Default: 20

  # Smoke mode (recommended first)
  SMOKE=1|0                Default: 1
                           SMOKE=1 uses small LIMITs to validate the pipeline end-to-end.

  OUT_DIR=...              Default: experiments/results/suite-<timestamp>

  # Ollama convenience
  AUTO_PULL=1|0            Default: 1
                           If MODEL is missing, try to pull it via Ollama's /api/pull.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! oc whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in to OpenShift. Run 'oc login' first." >&2
  exit 2
fi

ns="$(oc project -q)"
run_tag="$(date +%Y%m%d-%H%M%S)"
suite_dir="${OUT_DIR:-experiments/results/suite-${run_tag}}"
mkdir -p "${suite_dir}"

model="${MODEL:-granite4}"
ollama_host="${OLLAMA_HOST:-http://ollama:11434}"
parallelism="${PARALLELISM:-2}"
auto_pull="${AUTO_PULL:-1}"

humaneval_shards="${HUMANEVAL_SHARDS:-10}"
mbpp_shards="${MBPP_SHARDS:-20}"

smoke="${SMOKE:-1}"

echo "Namespace:   ${ns}"
echo "Suite dir:   ${suite_dir}"
echo "Backend:     ollama"
echo "Model:       ${model}"
echo "Ollama host: ${ollama_host}"
echo "Parallelism: ${parallelism}"
echo "SMOKE:       ${smoke}"
echo "AUTO_PULL:   ${auto_pull}"

echo "--- checking ollama model tags (in-cluster) ---"
raw_tags="${suite_dir}/ollama-tags.raw.txt"
tags_json="${suite_dir}/ollama-tags.json"
if ! oc run --rm -i curl --image=curlimages/curl --restart=Never --command -- \
  curl -s "${ollama_host}/api/tags" >"${raw_tags}"; then
  echo "ERROR: could not reach ${ollama_host} from the namespace." >&2
  exit 2
fi

extract_json() {
  local in_path="$1"
  local out_path="$2"
  uv run python - <<PY
import json
from pathlib import Path

raw = Path(${in_path@Q}).read_text(encoding="utf-8", errors="replace")
start = raw.find("{")
end = raw.rfind("}")
if start < 0 or end < 0 or end <= start:
    raise SystemExit("Could not locate JSON object in output")
obj = json.loads(raw[start : end + 1])
Path(${out_path@Q}).write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")
PY
}

extract_json "${raw_tags}" "${tags_json}"

resolve_model() {
  uv run python - <<PY
import json
from pathlib import Path

tags = json.loads(Path(${tags_json@Q}).read_text(encoding="utf-8"))
names = sorted({m.get("name", "") for m in tags.get("models", []) if m.get("name", "")})
want = ${model@Q}
if want in names:
    print(want)
    raise SystemExit(0)

# If the user asked for an untagged name (e.g. "granite4") but Ollama stores it as "granite4:latest",
# pick a sensible match.
if ":" not in want:
    pref = want + ":"
    matches = [n for n in names if n.startswith(pref)]
    if len(matches) == 1:
        print(matches[0])
        raise SystemExit(0)
    if f"{want}:latest" in matches:
        print(f"{want}:latest")
        raise SystemExit(0)
    if matches:
        print(sorted(matches)[0])
        raise SystemExit(0)

available = ", ".join(names)
raise SystemExit(f"missing:{want} available:{available}")
PY
}

resolved=""
if ! resolved="$(resolve_model 2>/dev/null)"; then
  msg="$(resolve_model 2>&1 || true)"
  echo "NOTE: Ollama model not present (${msg})."
  if [[ "${auto_pull}" != "1" ]]; then
    echo "ERROR: set AUTO_PULL=1 to auto-pull or manually pull the model in the cluster." >&2
    exit 2
  fi
  echo "--- pulling model via ollama /api/pull (this can take a while) ---"
  pull_log="${suite_dir}/ollama-pull-${model}.log"
  # /api/pull streams JSON lines; we save them for debugging.
  oc run --rm -i curl --image=curlimages/curl --restart=Never --command -- \
    sh -lc "curl -s -X POST '${ollama_host}/api/pull' -H 'Content-Type: application/json' -d '{\"name\":\"${model}\"}'" \
    >"${pull_log}" || true

  # Re-check tags after pulling.
  oc run --rm -i curl --image=curlimages/curl --restart=Never --command -- \
    curl -s "${ollama_host}/api/tags" >"${raw_tags}"
  extract_json "${raw_tags}" "${tags_json}"
  if ! resolved="$(resolve_model 2>/dev/null)"; then
    echo "ERROR: model ${model@Q} still not present after /api/pull. See:" >&2
    echo "  - ${tags_json}" >&2
    echo "  - ${pull_log}" >&2
    exit 2
  fi
fi

if [[ "${resolved}" != "${model}" ]]; then
  echo "NOTE: using resolved Ollama tag ${resolved@Q} (requested ${model@Q})."
  model="${resolved}"
fi

run_job() {
  local bench="$1"
  local budget="$2"
  local timeout="$3"
  local shards="$4"
  local limit="$5"

  local job_name="mcode-${bench}-b${budget}-t${timeout}-${run_tag}"
  job_name="$(echo "${job_name}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -c 'a-z0-9.-' '-')"
  # Trim to 63 chars and ensure it starts/ends with [a-z0-9] (k8s name + label value safe).
  job_name="${job_name:0:63}"
  job_name="$(echo "${job_name}" | sed -E 's/^[^a-z0-9]+//; s/[^a-z0-9]+$//')"
  if [[ -z "${job_name}" ]]; then
    echo "ERROR: could not derive a valid job name" >&2
    return 2
  fi

  local raw_dir="${suite_dir}/raw/${job_name}"
  mkdir -p "${raw_dir}"

  echo "=== ${bench} budget=${budget} timeout=${timeout} shards=${shards} limit=${limit:-none} ==="

  JOB_NAME="${job_name}" \
  OUT_DIR="${raw_dir}" \
  FETCH_RESULTS=1 \
  WAIT_TIMEOUT=12h \
  PARALLELISM="${parallelism}" \
  OVERRIDE_BENCHMARK="${bench}" \
  OVERRIDE_BACKEND="ollama" \
  OVERRIDE_MODEL="${model}" \
  OVERRIDE_OLLAMA_HOST="${ollama_host}" \
  OVERRIDE_LOOP_BUDGET="${budget}" \
  OVERRIDE_TIMEOUT_S="${timeout}" \
  OVERRIDE_SHARD_COUNT="${shards}" \
  OVERRIDE_LIMIT="${limit}" \
  ./deploy/k8s/run-bench.sh

  # Merge shard DBs into one DB per run.
  local merged_db="${suite_dir}/${bench}-b${budget}-t${timeout}.db"
  python_shards=()
  while IFS= read -r p; do
    [[ -n "${p}" ]] || continue
    python_shards+=( "${p}" )
  done < <(find "${raw_dir}" -name "${bench}-shard-*.db" -type f | sort)

  if (( ${#python_shards[@]} == 0 )); then
    echo "ERROR: no shard DBs found under ${raw_dir}" >&2
    return 1
  fi

  uv run mcode merge-shards --out "${merged_db}" --force "${python_shards[@]}"
  uv run mcode results --db "${merged_db}" --benchmark "${bench}" >"${suite_dir}/${bench}-b${budget}-t${timeout}.results.txt"
  echo "Merged: ${merged_db}"
}

echo "--- suite plan ---"
if [[ "${smoke}" == "1" ]]; then
  echo "HumanEval: 2 configs (LIMIT=10)"
  echo "MBPP:     1 config (LIMIT=10)"
  echo "SWE-lite: gold LIMIT=2"
else
  echo "HumanEval: 4 budgets (full)"
  echo "MBPP:     3 budgets (full)"
  echo "SWE-lite: gold LIMIT=10 + model LIMIT=3"
fi

mkdir -p "${suite_dir}/raw"

if [[ "${smoke}" == "1" ]]; then
  run_job humaneval 1 90 "${humaneval_shards}" 10
  run_job humaneval 5 90 "${humaneval_shards}" 10
  run_job mbpp 1 90 "${mbpp_shards}" 10

  MODE=gold SPLIT=test LIMIT=2 PARALLELISM=1 \
    OUT_DIR="${suite_dir}/swebench-gold" \
    DB="${suite_dir}/swebench-lite-gold.db" \
    CLEANUP=1 \
    ./deploy/k8s/run-swebench-lite.sh
else
  run_job humaneval 1 90 "${humaneval_shards}" ""
  run_job humaneval 5 90 "${humaneval_shards}" ""
  run_job humaneval 5 120 "${humaneval_shards}" ""
  run_job humaneval 20 120 "${humaneval_shards}" ""

  run_job mbpp 1 90 "${mbpp_shards}" ""
  run_job mbpp 3 90 "${mbpp_shards}" ""
  run_job mbpp 3 120 "${mbpp_shards}" ""

  MODE=gold SPLIT=test LIMIT=10 PARALLELISM=2 \
    OUT_DIR="${suite_dir}/swebench-gold" \
    DB="${suite_dir}/swebench-lite-gold.db" \
    CLEANUP=1 \
    ./deploy/k8s/run-swebench-lite.sh

  MODE=model SPLIT=test LIMIT=3 PARALLELISM=1 \
    BACKEND=ollama MODEL="${model}" OLLAMA_HOST="${ollama_host}" MCODE_MAX_NEW_TOKENS=512 \
    OUT_DIR="${suite_dir}/swebench-model" \
    DB="${suite_dir}/swebench-lite-model.db" \
    CLEANUP=1 \
    ./deploy/k8s/run-swebench-lite.sh
fi

echo "=== suite complete ==="
echo "Outputs: ${suite_dir}"
