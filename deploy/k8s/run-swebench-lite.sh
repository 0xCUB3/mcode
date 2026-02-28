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
  ./deploy/k8s/run-swebench-lite.sh <instance_id> [<instance_id> ...]

If you don't pass instance IDs, set LIMIT to pick the first N instances from the dataset.

Env vars (optional):
  MODE=gold|model          Default: model
  SPLIT=test|dev           Default: test
  LIMIT=<N>                If no args are given, run the first N instances

  PARALLELISM=<N>          Default: 4 (pods in flight)
  TIMEOUT_S=<seconds>      Default: 1800 (per-pod eval budget)

  OUT_DIR=...              Default: results-swebench-lite-<timestamp> (logs + JSON)
  DB=...                   Default: experiments/results/swebench-lite-<timestamp>.db
  NAME_PREFIX=...          Default: mcode-sweb-<timestamp> (avoid collisions)

  # Patch generation (MODE=model): forwarded to run-swebench-lite-one.sh
  BACKEND=openai|ollama    Default: openai
  MODEL=<model_id>         Default: ibm-granite/granite-3.0-8b-instruct
  OPENAI_BASE_URL=...      Default: http://vllm:8000/v1
  OPENAI_API_KEY=...       Default: dummy
  OLLAMA_HOST=...          Default: http://ollama:11434
  MCODE_IMAGE=...          Default: OpenShift internal registry mcode:latest
  MCODE_MAX_NEW_TOKENS=... Default: 4096

  # Cleanup
  CLEANUP=1               Delete pods + configmaps as they finish (recommended)
USAGE
}

mode="${MODE:-model}"
split="${SPLIT:-test}"
limit="${LIMIT:-}"
parallelism="${PARALLELISM:-4}"
timeout_s="${TIMEOUT_S:-1800}"
cleanup="${CLEANUP:-1}"
backend="${BACKEND:-openai}"
model="${MODEL:-ibm-granite/granite-3.0-8b-instruct}"

run_tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
out_dir="${OUT_DIR:-results-swebench-lite-${run_tag}}"
db_path="${DB:-experiments/results/swebench-lite-${run_tag}.db}"
name_prefix="${NAME_PREFIX:-mcode-sweb-${run_tag}}"

if [[ "${mode}" != "gold" && "${mode}" != "model" ]]; then
  echo "ERROR: MODE must be 'gold' or 'model' (got ${mode@Q})." >&2
  usage
  exit 2
fi
if [[ "${split}" != "test" && "${split}" != "dev" ]]; then
  echo "ERROR: SPLIT must be 'test' or 'dev' (got ${split@Q})." >&2
  usage
  exit 2
fi

instance_ids=("$@")

if (( ${#instance_ids[@]} == 0 )); then
  if [[ -z "${limit}" ]]; then
    usage
    exit 2
  fi
  mapfile -t instance_ids < <(
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

from swebench.harness.utils import load_swebench_dataset

split = ${split@Q}
limit = int(${limit@Q})
instances = load_swebench_dataset("SWE-bench/SWE-bench_Lite", split)
for inst in instances[:limit]:
    print(inst["instance_id"])
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
  run_log="${out_dir}/${base}-${hash8}.run.log"

  (
    set +e
    MODE="${mode}" \
    SPLIT="${split}" \
    TIMEOUT_S="${timeout_s}" \
    NAME_PREFIX="${name_prefix}" \
    OUT_DIR="${out_dir}" \
    CLEANUP="${cleanup}" \
    ./deploy/k8s/run-swebench-lite-one.sh "${instance_id}" >"${run_log}" 2>&1
    echo $? >"${run_log}.exit"
  ) &
done

wait || true

echo "--- ingest to sqlite ---"
uv run python - <<PY
import json
from pathlib import Path

from mcode.bench.results import ResultsDB

out_dir = Path(${out_dir@Q})
db_path = Path(${db_path@Q})

mode = ${mode@Q}
split = ${split@Q}
timeout_s = int(${timeout_s@Q})
parallelism = int(${parallelism@Q})

backend = ${backend@Q}
model = ${model@Q}

def extract_test_output(log_text: str, max_chars: int = 8000) -> str:
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

db = ResultsDB(db_path)
run_id = db.start_run(
    "swebench-lite",
    {
        "backend_name": backend,
        "model_id": model,
        "loop_budget": 1,
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
    elif report and report.get("patch_successfully_applied") is False:
        error_msg = "Patch apply failed"
    elif not resolved:
        error_msg = "Not resolved"

    if resolved:
        passed += 1

    db.save_task_result(
        run_id,
        {
            "task_id": instance_id,
            "passed": 1 if resolved else 0,
            "attempts_used": 1 if data else 0,
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
