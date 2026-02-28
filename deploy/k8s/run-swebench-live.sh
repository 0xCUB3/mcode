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
  ./deploy/k8s/run-swebench-live.sh <instance_id> [<instance_id> ...]

If you don't pass instance IDs, set LIMIT to pick the first N instances from the dataset.

Env vars (optional):
  MODE=gold|model          Default: model
  SPLIT=verified           Default: verified
  LIMIT=<N>                If no args are given, run the first N instances

  PARALLELISM=<N>          Default: 4 (pods in flight)
  TIMEOUT_S=<seconds>      Default: 1800 (per-pod eval budget)

  OUT_DIR=...              Default: results-swebench-live-<timestamp> (logs + JSON)
  DB=...                   Default: experiments/results/swebench-live-<timestamp>.db
  NAME_PREFIX=...          Default: mcode-sweb-live-<timestamp> (avoid collisions)

  # Patch generation (MODE=model): forwarded to run-swebench-live-one.sh
  BACKEND=openai|ollama    Default: openai
  MODEL=<model_id>         Default: ibm-granite/granite-3.0-8b-instruct
  OPENAI_BASE_URL=...      Default: http://vllm:8000/v1
  OPENAI_API_KEY=...       Default: dummy
  OLLAMA_HOST=...          Default: http://ollama:11434
  MCODE_IMAGE=...          Default: OpenShift internal registry mcode:latest
  MCODE_MAX_NEW_TOKENS=... Default: 4096
  LOOP_BUDGET=<N>          Default: 3 (max patch+test attempts)

  # Cleanup
  CLEANUP=1               Delete pods + configmaps as they finish (recommended)
USAGE
}

mode="${MODE:-model}"
split="${SPLIT:-verified}"
limit="${LIMIT:-}"
parallelism="${PARALLELISM:-4}"
timeout_s="${TIMEOUT_S:-1800}"
loop_budget="${LOOP_BUDGET:-3}"
cleanup="${CLEANUP:-1}"
backend="${BACKEND:-openai}"
model="${MODEL:-ibm-granite/granite-3.0-8b-instruct}"

run_tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
out_dir="${OUT_DIR:-results-swebench-live-${run_tag}}"
db_path="${DB:-experiments/results/swebench-live-${run_tag}.db}"
name_prefix="${NAME_PREFIX:-mcode-sweb-live-${run_tag}}"

if [[ "${mode}" != "gold" && "${mode}" != "model" ]]; then
  echo "ERROR: MODE must be 'gold' or 'model' (got '${mode}')." >&2
  usage
  exit 2
fi

# Bash 3.2 compatible shell quoting (replaces ${var@Q} from bash 4.4+)
shquote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

instance_ids=("$@")

if (( ${#instance_ids[@]} == 0 )); then
  if [[ -z "${limit}" ]]; then
    usage
    exit 2
  fi
  instance_ids=()
  while IFS= read -r line; do
    instance_ids+=("$line")
  done < <(
    uv run python - <<PY
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
try:
    from huggingface_hub.utils import logging as hub_logging
    hub_logging.set_verbosity_error()
except Exception:
    pass
try:
    from datasets.utils.logging import set_verbosity_error as datasets_set_verbosity_error
    datasets_set_verbosity_error()
except Exception:
    pass

from datasets import load_dataset

split = $(shquote "${split}")
limit = int($(shquote "${limit}"))
ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split=split)
for i, row in enumerate(ds):
    if i >= limit:
        break
    print(row["instance_id"])
PY
  )
fi

if (( ${#instance_ids[@]} == 0 )); then
  echo "ERROR: no instance IDs selected." >&2
  exit 1
fi

echo "Namespace:    $(oc project -q)"
echo "Mode:         ${mode}"
echo "Split:        ${split}"
echo "Instances:    ${#instance_ids[@]}"
echo "Parallelism:  ${parallelism}"
echo "Timeout:      ${timeout_s}s"
echo "OUT_DIR:      ${out_dir}"
echo "DB:           ${db_path}"
echo "NAME_PREFIX:  ${name_prefix}"

mkdir -p "${out_dir}"
mkdir -p "$(dirname "${db_path}")"
printf '%s\n' "${instance_ids[@]}" >"${out_dir}/instances.txt"

sanitize() {
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | tr '_' '-' \
    | tr -c 'a-z0-9-' '-' \
    | sed -E 's/-+/-/g; s/^-//; s/-$//'
}

for instance_id in "${instance_ids[@]}"; do
  while (( $(jobs -rp | wc -l | tr -d ' ') >= parallelism )); do
    sleep 1
  done

  hash8="$(printf '%s' "${instance_id}" | shasum -a 256 | cut -c1-8)"
  base="$(sanitize "${instance_id}")"
  base="${base:0:42}"
  result_json="${out_dir}/${base}-${hash8}.result.json"
  run_log="${out_dir}/${base}-${hash8}.run.log"

  # Skip if result already exists AND the pod actually ran (not stuck in image pull).
  # Phase "Pending" means the pod never started (e.g. ImagePullBackOff / rate limit).
  if [[ -s "${result_json}" ]]; then
    result_phase="$(python3 -c "import json,sys; print(json.load(sys.stdin).get('phase',''))" <"${result_json}" 2>/dev/null || true)"
    if [[ "${result_phase}" == "Succeeded" || "${result_phase}" == "Failed" ]]; then
      echo "SKIP ${instance_id} (result exists, phase=${result_phase})"
      continue
    else
      echo "RETRY ${instance_id} (previous phase=${result_phase:-unknown}, removing stale result)"
      rm -f "${result_json}"
    fi
  fi

  (
    set +e
    MODE="${mode}" \
    SPLIT="${split}" \
    TIMEOUT_S="${timeout_s}" \
    LOOP_BUDGET="${loop_budget}" \
    NAME_PREFIX="${name_prefix}" \
    OUT_DIR="${out_dir}" \
    CLEANUP="${cleanup}" \
    ./deploy/k8s/run-swebench-live-one.sh "${instance_id}" >"${run_log}" 2>&1
    echo $? >"${run_log}.exit"
  ) &
done

wait || true

echo "--- ingest to sqlite ---"
uv run python - <<PY
import json
from pathlib import Path

from mcode.bench.results import ResultsDB

out_dir = Path($(shquote "${out_dir}"))
db_path = Path($(shquote "${db_path}"))

mode = $(shquote "${mode}")
split = $(shquote "${split}")
timeout_s = int($(shquote "${timeout_s}"))
parallelism = int($(shquote "${parallelism}"))

backend = $(shquote "${backend}")
model = $(shquote "${model}")

db = ResultsDB(db_path)
run_id = db.start_run(
    "swebench-live",
    {
        "backend_name": backend,
        "model_id": model,
        "loop_budget": int($(shquote "${loop_budget}")),
        "timeout_s": timeout_s,
        "retrieval": False,
        "runner": "openshift-pods",
        "swebench_mode": mode,
        "swebench_split": split,
        "parallelism": parallelism,
        "out_dir": str(out_dir),
    },
)

passed = 0

expected_ids_path = out_dir / "instances.txt"
expected_ids = [
    line.strip()
    for line in expected_ids_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not expected_ids:
    raise SystemExit(f"No instance IDs found in {expected_ids_path}")

results_by_id: dict[str, Path] = {}
for result_path in out_dir.glob("*.result.json"):
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        instance_id = data.get("instance_id")
        if instance_id:
            results_by_id[str(instance_id)] = result_path
    except Exception:
        continue

total = len(expected_ids)

for instance_id in expected_ids:
    result_path = results_by_id.get(instance_id)
    data = {}
    if result_path and result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))

    resolved = bool(data.get("resolved"))
    phase = data.get("phase") or ""
    reason = (data.get("reason") or "").strip()

    eval_log_path = None
    if result_path:
        eval_log_path = result_path.with_suffix("").with_suffix(".eval.log")
    eval_log_text = ""
    if eval_log_path and eval_log_path.exists():
        eval_log_text = eval_log_path.read_text(encoding="utf-8", errors="replace")

    report = data.get("report") or {}
    err = data.get("error")

    timed_out = False
    if isinstance(reason, str) and "deadline" in reason.lower():
        timed_out = True
    if phase.lower() == "failed" and isinstance(err, str) and "timed out" in err.lower():
        timed_out = True

    error_msg = None
    if err:
        error_msg = str(err)
    elif not data:
        error_msg = "Missing result JSON (script failure?)"
    elif timed_out:
        error_msg = "Timed out"
    elif phase and phase != "Succeeded":
        error_msg = f"Pod phase={phase}"
    elif data.get("patch_successfully_applied") is False:
        error_msg = "Patch apply failed"
    elif not resolved:
        error_msg = "Not resolved"

    if resolved:
        passed += 1

    def extract_test_output(log_text, max_chars=8000):
        start = ">>>>> Start Test Output"
        end = ">>>>> End Test Output"
        if start in log_text and end in log_text:
            chunk = log_text.split(start, 1)[1].split(end, 1)[0]
        else:
            chunk = ""
        chunk = chunk.strip()
        if len(chunk) > max_chars:
            chunk = chunk[-max_chars:]
        return chunk

    db.save_task_result(
        run_id,
        {
            "task_id": instance_id,
            "passed": 1 if resolved else 0,
            "attempts_used": int(data.get("attempts_used", 1)) if data else 0,
            "time_ms": int(data.get("time_ms") or 0) if data else 0,
            "exit_code": None,
            "timed_out": 1 if timed_out else 0,
            "stdout": extract_test_output(eval_log_text),
            "stderr": json.dumps(report, sort_keys=True) if report else None,
            "error": error_msg,
            "code_sha256": data.get("patch_sha256") if data else None,
        },
    )

print(f"run_id={run_id} total={total} passed={passed} pass_rate={(passed/total if total else 0.0):.1%}")
print(f"db={db_path}")
PY

echo "Done."
