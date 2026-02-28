#!/usr/bin/env bash
set -euo pipefail

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: 'oc' is required (OpenShift CLI)." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is required to generate SWE-bench metadata locally." >&2
  exit 2
fi

usage() {
  cat <<'USAGE' >&2
Usage:
  ./deploy/k8s/run-swebench-lite-one.sh <instance_id>

Env vars (optional):
  MODE=gold|model          Default: model
  SPLIT=test|dev           Default: test
  TIMEOUT_S=<seconds>      Default: 1800 (used for waiting + result metadata)
  POD_DEADLINE_S=<seconds> Default: TIMEOUT_S+600 (k8s activeDeadlineSeconds)
  WAIT_TIMEOUT_S=<seconds> Default: POD_DEADLINE_S+120 (how long the script waits)
  NAME_PREFIX=...          Default: mcode-swebench (set to avoid name collisions)
  OUT_DIR=...              If set, write logs + result JSON locally

  # Patch generation (MODE=model)
  BACKEND=openai|ollama    Default: openai
  MODEL=<model_id>         Default: ibm-granite/granite-3.0-8b-instruct
  OPENAI_BASE_URL=...      Default: http://vllm:8000/v1
  OPENAI_API_KEY=...       Default: dummy
  OLLAMA_HOST=...          Default: http://ollama:11434
  MCODE_IMAGE=...          Default: OpenShift internal registry mcode:latest
  MCODE_MAX_NEW_TOKENS=... Default: 4096

  # Cleanup
  CLEANUP=1               Delete pod + configmap after completion
USAGE
}

instance_id="${1:-}"
if [[ -z "${instance_id}" ]]; then
  usage
  exit 2
fi

mode="${MODE:-model}"
split="${SPLIT:-test}"

if [[ "${mode}" != "gold" && "${mode}" != "model" ]]; then
  echo "ERROR: MODE must be 'gold' or 'model' (got ${mode@Q})." >&2
  exit 2
fi

namespace="$(oc project -q)"

sanitize() {
  # Kubernetes names: lowercase alnum + '-', <=63 chars
  echo "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | tr '_' '-' \
    | tr -c 'a-z0-9-' '-' \
    | sed -E 's/-+/-/g; s/^-//; s/-$//'
}

timeout_s="${TIMEOUT_S:-1800}"
pod_deadline_s="${POD_DEADLINE_S:-$((timeout_s + 600))}"
wait_timeout_s="${WAIT_TIMEOUT_S:-$((pod_deadline_s + 120))}"

out_dir="${OUT_DIR:-}"

tmp_dir="$(mktemp -d -t mcode-swebench.XXXXXX)"
tmp_eval_log=""
tmp_gen_log=""
tmp_result_json=""
cleanup_tmp() {
  rm -rf "${tmp_dir}"
  [[ -n "${tmp_eval_log}" ]] && rm -f "${tmp_eval_log}" || true
  [[ -n "${tmp_gen_log}" ]] && rm -f "${tmp_gen_log}" || true
  [[ -n "${tmp_result_json}" ]] && rm -f "${tmp_result_json}" || true
}
trap cleanup_tmp EXIT

hash8="$(printf '%s' "${instance_id}" | shasum -a 256 | cut -c1-8)"
base="$(sanitize "${instance_id}")"
name_prefix="$(sanitize "${NAME_PREFIX:-mcode-swebench}")"
max_base_len=$((63 - ${#name_prefix} - 1 - ${#hash8} - 1))
if (( max_base_len < 8 )); then
  max_base_len=8
fi
base="${base:0:${max_base_len}}"
pod_name="${name_prefix}-${base}-${hash8}"
cm_name="${pod_name}-inputs"

arch="x86_64"
image_id="$(echo "${instance_id}" | tr '[:upper:]' '[:lower:]' | sed 's/__/_1776_/g')"
eval_image="swebench/sweb.eval.${arch}.${image_id}:latest"

log_prefix=""
eval_log="${tmp_dir}/eval.log"
gen_log="${tmp_dir}/gen.log"
result_json="${tmp_dir}/result.json"
if [[ -n "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
  log_prefix="${out_dir}/${base}-${hash8}"
  eval_log="${log_prefix}.eval.log"
  gen_log="${log_prefix}.gen.log"
  result_json="${log_prefix}.result.json"
else
  tmp_eval_log="$(mktemp -t mcode-swebench-eval.XXXXXX.log)"
  tmp_gen_log="$(mktemp -t mcode-swebench-gen.XXXXXX.log)"
  tmp_result_json="$(mktemp -t mcode-swebench-result.XXXXXX.json)"
  eval_log="${tmp_eval_log}"
  gen_log="${tmp_gen_log}"
  result_json="${tmp_result_json}"
fi

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

from pathlib import Path
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.test_spec.test_spec import make_test_spec

instance_id = ${instance_id@Q}
split = ${split@Q}
mode = ${mode@Q}

instances = load_swebench_dataset("SWE-bench/SWE-bench_Lite", split, [instance_id])
if not instances:
    raise SystemExit(f"Instance not found: {instance_id!r} (split={split})")
inst = instances[0]

spec = make_test_spec(inst, namespace="swebench", arch="x86_64")

out = Path(${tmp_dir@Q})
out.mkdir(parents=True, exist_ok=True)

(out / "eval.sh").write_text(spec.eval_script, encoding="utf-8", errors="replace")

if mode == "gold":
    (out / "patch.diff").write_text(str(inst["patch"]), encoding="utf-8", errors="replace")
else:
    (out / "repo.txt").write_text(str(inst["repo"]), encoding="utf-8")
    (out / "problem.txt").write_text(
        str(inst.get("problem_statement", "")),
        encoding="utf-8",
        errors="replace",
    )
    (out / "hints.txt").write_text(str(inst.get("hints_text", "")), encoding="utf-8", errors="replace")
PY

oc delete pod "${pod_name}" --ignore-not-found=true >/dev/null
oc delete configmap "${cm_name}" --ignore-not-found=true >/dev/null

gold_patch_sha256=""
if [[ "${mode}" == "gold" ]]; then
  gold_patch_sha256="$(shasum -a 256 "${tmp_dir}/patch.diff" | awk '{print $1}')"
fi

if [[ "${mode}" == "gold" ]]; then
  oc create configmap "${cm_name}" \
    --from-file=eval.sh="${tmp_dir}/eval.sh" \
    --from-file=patch.diff="${tmp_dir}/patch.diff" \
    >/dev/null
else
  oc create configmap "${cm_name}" \
    --from-file=eval.sh="${tmp_dir}/eval.sh" \
    --from-file=repo.txt="${tmp_dir}/repo.txt" \
    --from-file=problem.txt="${tmp_dir}/problem.txt" \
    --from-file=hints.txt="${tmp_dir}/hints.txt" \
    >/dev/null
fi

mcode_image_default="image-registry.openshift-image-registry.svc:5000/${namespace}/mcode:latest"
mcode_image="${MCODE_IMAGE:-$mcode_image_default}"

backend="${BACKEND:-openai}"
model="${MODEL:-ibm-granite/granite-3.0-8b-instruct}"
openai_base_url="${OPENAI_BASE_URL:-http://vllm:8000/v1}"
openai_api_key="${OPENAI_API_KEY:-dummy}"
ollama_host="${OLLAMA_HOST:-http://ollama:11434}"
max_new_tokens="${MCODE_MAX_NEW_TOKENS:-4096}"

if [[ "${mode}" == "gold" ]]; then
  cat <<YAML | oc apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
spec:
  restartPolicy: Never
  activeDeadlineSeconds: ${pod_deadline_s}
  volumes:
    - name: inputs
      configMap:
        name: ${cm_name}
  containers:
    - name: eval
      image: ${eval_image}
      env:
        - name: HOME
          value: /tmp
        - name: PYTHONUSERBASE
          value: /tmp/.local
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          mkdir -p /tmp/.config /tmp/.local

          workdir=/tmp/testbed
          rm -rf "\$workdir"
          cp -R /testbed "\$workdir"
          chmod -R u+rwX,go+rX "\$workdir"
          cd "\$workdir"
          git config --global --add safe.directory "\$workdir" || true

          patch_file=/inputs/patch.diff
          if [ ! -s "\$patch_file" ]; then
            echo "patch.diff is missing or empty" >&2
            exit 2
          fi

          if git apply --verbose "\$patch_file"; then
            echo '>>>>> Applied Patch'
          elif git apply --verbose --reject "\$patch_file"; then
            echo '>>>>> Applied Patch'
          elif patch --batch --fuzz=5 -p1 -i "\$patch_file"; then
            echo '>>>>> Applied Patch'
          else
            echo '>>>>> Patch Apply Failed'
            # Leave the pod successful; this is a benchmark outcome, not an infra failure.
            exit 0
          fi

          eval_copy=/tmp/eval.sh
          cp /inputs/eval.sh "\$eval_copy"
          sed -i "s|/testbed|\$workdir|g" "\$eval_copy"
          bash "\$eval_copy"
      volumeMounts:
        - name: inputs
          mountPath: /inputs
          readOnly: true
YAML
else
  cat <<YAML | oc apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
spec:
  restartPolicy: Never
  activeDeadlineSeconds: ${pod_deadline_s}
  volumes:
    - name: work
      emptyDir: {}
    - name: inputs
      configMap:
        name: ${cm_name}
  initContainers:
    - name: copy-testbed
      image: ${eval_image}
      command: ["cp", "-a", "/testbed", "/work/testbed"]
      volumeMounts:
        - name: work
          mountPath: /work
    - name: gen-patch
      image: ${mcode_image}
      env:
        - name: BACKEND
          value: ${backend}
        - name: MODEL
          value: ${model}
        - name: OPENAI_BASE_URL
          value: ${openai_base_url}
        - name: OPENAI_API_KEY
          value: ${openai_api_key}
        - name: OLLAMA_HOST
          value: ${ollama_host}
        - name: MCODE_MAX_NEW_TOKENS
          value: "${max_new_tokens}"
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          python - <<'PY'
          import hashlib
          import os
          from pathlib import Path
          from mcode.llm.session import LLMSession, edits_to_patch

          repo = Path('/inputs/repo.txt').read_text(encoding='utf-8').strip()
          problem = Path('/inputs/problem.txt').read_text(encoding='utf-8', errors='replace')
          hints = Path('/inputs/hints.txt').read_text(encoding='utf-8', errors='replace')

          model_id = os.environ['MODEL']
          backend = os.environ.get('BACKEND', 'openai')

          s = LLMSession(model_id=model_id, backend_name=backend)
          s.check_available()
          with s.open():
              result = s.generate_patch(repo=repo, problem_statement=problem, hints_text=hints)

          patch, _ = edits_to_patch(result.value or "", repo_root="/work/testbed")
          patch = patch or ""
          Path('/work/patch.diff').write_text(patch, encoding='utf-8', errors='replace')
          sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest()
          print(f'generated patch chars={len(patch)}')
          print(f'patch_sha256={sha}')
          PY
      volumeMounts:
        - name: work
          mountPath: /work
        - name: inputs
          mountPath: /inputs
          readOnly: true
  containers:
    - name: eval
      image: ${eval_image}
      env:
        - name: HOME
          value: /tmp
        - name: PYTHONUSERBASE
          value: /tmp/.local
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          mkdir -p /tmp/.config /tmp/.local

          workdir=/tmp/testbed
          rm -rf "\$workdir"
          cp -R /testbed "\$workdir"
          chmod -R u+rwX,go+rX "\$workdir"
          cd "\$workdir"
          git config --global --add safe.directory "\$workdir" || true

          patch_file=/work/patch.diff
          if [ ! -s "\$patch_file" ]; then
            echo "patch.diff is missing or empty" >&2
            exit 2
          fi

          if git apply --verbose "\$patch_file"; then
            echo '>>>>> Applied Patch'
          elif git apply --verbose --reject "\$patch_file"; then
            echo '>>>>> Applied Patch'
          elif patch --batch --fuzz=5 -p1 -i "\$patch_file"; then
            echo '>>>>> Applied Patch'
          else
            echo '>>>>> Patch Apply Failed'
            # Leave the pod successful; this is a benchmark outcome, not an infra failure.
            exit 0
          fi

          eval_copy=/tmp/eval.sh
          cp /inputs/eval.sh "\$eval_copy"
          sed -i "s|/testbed|\$workdir|g" "\$eval_copy"
          bash "\$eval_copy"
      volumeMounts:
        - name: work
          mountPath: /work
        - name: inputs
          mountPath: /inputs
          readOnly: true
YAML
fi

echo "Namespace:    ${namespace}"
echo "Pod:          ${pod_name}"
echo "Instance ID:  ${instance_id}"
echo "Image:        ${eval_image}"
echo "Mode:         ${mode}"
echo "Timeout:      ${timeout_s}s"
echo "Pod deadline: ${pod_deadline_s}s"

phase=""
pod_reason=""
start_wait_s="$(date +%s)"
while true; do
  phase="$(oc get pod "${pod_name}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  pod_reason="$(oc get pod "${pod_name}" -o jsonpath='{.status.reason}' 2>/dev/null || true)"
  if [[ "${phase}" == "Succeeded" || "${phase}" == "Failed" ]]; then
    break
  fi
  now_s="$(date +%s)"
  if (( now_s - start_wait_s >= wait_timeout_s )); then
    break
  fi
  sleep 2
done
elapsed_ms=$(( ( $(date +%s) - start_wait_s ) * 1000 ))

echo "Phase:        ${phase:-unknown}"
if [[ -n "${pod_reason}" ]]; then
  echo "Reason:       ${pod_reason}"
fi

if [[ "${mode}" == "model" ]]; then
  oc logs "${pod_name}" -c gen-patch >"${gen_log}" 2>&1 || true
  echo "--- gen-patch (tail) ---"
  tail -n 200 "${gen_log}" || true
fi

oc logs "${pod_name}" -c eval >"${eval_log}" 2>&1 || true
echo "--- eval (tail) ---"
tail -n 200 "${eval_log}" || true

echo "--- swebench report ---"
MCODE_SWEB_RESULT_JSON="${result_json}" \
MCODE_SWEB_MODE="${mode}" \
MCODE_SWEB_SPLIT="${split}" \
MCODE_SWEB_BACKEND="${backend}" \
MCODE_SWEB_MODEL="${model}" \
MCODE_SWEB_NAMESPACE="${namespace}" \
MCODE_SWEB_POD_NAME="${pod_name}" \
MCODE_SWEB_PHASE="${phase:-unknown}" \
MCODE_SWEB_REASON="${pod_reason}" \
MCODE_SWEB_EVAL_IMAGE="${eval_image}" \
MCODE_SWEB_TIME_MS="${elapsed_ms}" \
MCODE_SWEB_TIMEOUT_S="${timeout_s}" \
MCODE_SWEB_POD_DEADLINE_S="${pod_deadline_s}" \
MCODE_SWEB_GOLD_PATCH_SHA256="${gold_patch_sha256}" \
MCODE_SWEB_GEN_LOG="${gen_log}" \
uv run python - <<PY || true
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

import json
import os
import re
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

instance_id = ${instance_id@Q}
split = ${split@Q}
log_fp = ${eval_log@Q}

mode = os.environ.get("MCODE_SWEB_MODE", "")
phase = os.environ.get("MCODE_SWEB_PHASE", "")
reason = os.environ.get("MCODE_SWEB_REASON") or None
gen_log = os.environ.get("MCODE_SWEB_GEN_LOG") or ""

patch_sha256 = os.environ.get("MCODE_SWEB_GOLD_PATCH_SHA256") or None
if mode == "model" and gen_log and Path(gen_log).exists():
    m = re.search(r"^patch_sha256=([0-9a-f]{64})\\s*$", Path(gen_log).read_text(errors="replace"), re.M)
    if m:
        patch_sha256 = m.group(1)

row = {}
err = None
try:
    inst = load_swebench_dataset("SWE-bench/SWE-bench_Lite", split, [instance_id])[0]
    spec = make_test_spec(inst, namespace="swebench", arch="x86_64")
    pred = {KEY_INSTANCE_ID: instance_id, KEY_MODEL: "mcode", KEY_PREDICTION: "non-empty"}
    report = get_eval_report(spec, pred, log_fp, include_tests_status=False)
    row = report.get(instance_id, {})
except Exception as e:
    err = str(e)

resolved = row.get("resolved")
patch_successfully_applied = row.get("patch_successfully_applied")
patch_exists = row.get("patch_exists")

result = {
    "instance_id": instance_id,
    "split": split,
    "mode": mode,
    "backend": os.environ.get("MCODE_SWEB_BACKEND"),
    "model": os.environ.get("MCODE_SWEB_MODEL"),
    "namespace": os.environ.get("MCODE_SWEB_NAMESPACE"),
    "pod_name": os.environ.get("MCODE_SWEB_POD_NAME"),
    "phase": phase,
    "reason": reason,
    "eval_image": os.environ.get("MCODE_SWEB_EVAL_IMAGE"),
    "time_ms": int(os.environ.get("MCODE_SWEB_TIME_MS", "0") or "0"),
    "timeout_s": int(os.environ.get("MCODE_SWEB_TIMEOUT_S", "0") or "0"),
    "pod_deadline_s": int(os.environ.get("MCODE_SWEB_POD_DEADLINE_S", "0") or "0"),
    "patch_sha256": patch_sha256,
    "resolved": resolved,
    "patch_successfully_applied": patch_successfully_applied,
    "patch_exists": patch_exists,
    "report": row or None,
    "error": err,
}

out_path = os.environ.get("MCODE_SWEB_RESULT_JSON")
if out_path:
    Path(out_path).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

print(f"resolved={resolved} patch_successfully_applied={patch_successfully_applied} patch_exists={patch_exists}")
PY

if [[ "${CLEANUP:-0}" == "1" ]]; then
  oc delete pod "${pod_name}" --ignore-not-found=true >/dev/null
  oc delete configmap "${cm_name}" --ignore-not-found=true >/dev/null
fi

if [[ "${phase}" != "Succeeded" ]]; then
  echo "ERROR: pod did not succeed (phase=${phase:-unknown})." >&2
  if [[ -n "${out_dir}" ]]; then
    echo "Note: wrote logs + result JSON under ${out_dir@Q} (prefix=${log_prefix@Q})." >&2
  fi
  exit 1
fi
