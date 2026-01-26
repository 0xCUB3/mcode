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

  # Patch generation (MODE=model)
  BACKEND=openai|ollama    Default: openai
  MODEL=<model_id>         Default: ibm-granite/granite-3.0-8b-instruct
  OPENAI_BASE_URL=...      Default: http://vllm:8000/v1
  OPENAI_API_KEY=...       Default: dummy
  OLLAMA_HOST=...          Default: http://ollama:11434
  MCODE_IMAGE=...          Default: OpenShift internal registry mcode:latest
  MCODE_MAX_NEW_TOKENS=... Default: 512

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

hash8="$(printf '%s' "${instance_id}" | shasum -a 256 | cut -c1-8)"
base="$(sanitize "${instance_id}")"
base="${base:0:42}"
pod_name="mcode-swebench-${base}-${hash8}"
cm_name="${pod_name}-inputs"

arch="x86_64"
image_id="$(echo "${instance_id}" | tr '[:upper:]' '[:lower:]' | sed 's/__/_1776_/g')"
eval_image="swebench/sweb.eval.${arch}.${image_id}:latest"

tmp_dir="$(mktemp -d -t mcode-swebench.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT

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
(out / "repo.txt").write_text(str(inst["repo"]), encoding="utf-8")
(out / "problem.txt").write_text(str(inst.get("problem_statement", "")), encoding="utf-8", errors="replace")
(out / "hints.txt").write_text(str(inst.get("hints_text", "")), encoding="utf-8", errors="replace")
(out / "instance_id.txt").write_text(str(inst["instance_id"]), encoding="utf-8")

if mode == "gold":
    (out / "patch.diff").write_text(str(inst["patch"]), encoding="utf-8", errors="replace")
PY

oc delete pod "${pod_name}" --ignore-not-found=true >/dev/null
oc delete configmap "${cm_name}" --ignore-not-found=true >/dev/null

if [[ "${mode}" == "gold" ]]; then
  oc create configmap "${cm_name}" \
    --from-file=eval.sh="${tmp_dir}/eval.sh" \
    --from-file=patch.diff="${tmp_dir}/patch.diff" \
    --from-file=repo.txt="${tmp_dir}/repo.txt" \
    --from-file=problem.txt="${tmp_dir}/problem.txt" \
    --from-file=hints.txt="${tmp_dir}/hints.txt" \
    --from-file=instance_id.txt="${tmp_dir}/instance_id.txt" >/dev/null
else
  oc create configmap "${cm_name}" \
    --from-file=eval.sh="${tmp_dir}/eval.sh" \
    --from-file=repo.txt="${tmp_dir}/repo.txt" \
    --from-file=problem.txt="${tmp_dir}/problem.txt" \
    --from-file=hints.txt="${tmp_dir}/hints.txt" \
    --from-file=instance_id.txt="${tmp_dir}/instance_id.txt" >/dev/null
fi

mcode_image_default="image-registry.openshift-image-registry.svc:5000/${namespace}/mcode:latest"
mcode_image="${MCODE_IMAGE:-$mcode_image_default}"

backend="${BACKEND:-openai}"
model="${MODEL:-ibm-granite/granite-3.0-8b-instruct}"
openai_base_url="${OPENAI_BASE_URL:-http://vllm:8000/v1}"
openai_api_key="${OPENAI_API_KEY:-dummy}"
ollama_host="${OLLAMA_HOST:-http://ollama:11434}"
max_new_tokens="${MCODE_MAX_NEW_TOKENS:-512}"

if [[ "${mode}" == "gold" ]]; then
  cat <<YAML | oc apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
spec:
  restartPolicy: Never
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
  volumes:
    - name: work
      emptyDir: {}
    - name: inputs
      configMap:
        name: ${cm_name}
  initContainers:
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
          import os
          from pathlib import Path
          from mcode.llm.session import LLMSession

          repo = Path('/inputs/repo.txt').read_text(encoding='utf-8').strip()
          problem = Path('/inputs/problem.txt').read_text(encoding='utf-8', errors='replace')
          hints = Path('/inputs/hints.txt').read_text(encoding='utf-8', errors='replace')

          model_id = os.environ['MODEL']
          backend = os.environ.get('BACKEND', 'openai')

          s = LLMSession(model_id=model_id, backend_name=backend)
          s.check_available()
          with s.open():
              patch = s.generate_patch(repo=repo, problem_statement=problem, hints_text=hints)

          Path('/work/patch.diff').write_text(patch, encoding='utf-8', errors='replace')
          print(f'generated patch chars={len(patch)}')
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

phase=""
for _ in $(seq 1 360); do
  phase="$(oc get pod "${pod_name}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [[ "${phase}" == "Succeeded" || "${phase}" == "Failed" ]]; then
    break
  fi
  sleep 2
done

echo "Phase:        ${phase:-unknown}"

if [[ "${mode}" == "model" ]]; then
  echo "--- gen-patch (tail) ---"
  oc logs "${pod_name}" -c gen-patch --tail=200 || true
fi

echo "--- eval (tail) ---"
oc logs "${pod_name}" -c eval --tail=200 || true

echo "--- swebench report ---"
tmp_log="$(mktemp -t mcode-swebench-log.XXXXXX)"
oc logs "${pod_name}" -c eval >"${tmp_log}" || true
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

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

instance_id = ${instance_id@Q}
split = ${split@Q}
log_fp = ${tmp_log@Q}

inst = load_swebench_dataset("SWE-bench/SWE-bench_Lite", split, [instance_id])[0]
spec = make_test_spec(inst, namespace="swebench", arch="x86_64")

pred = {KEY_INSTANCE_ID: instance_id, KEY_MODEL: "mcode", KEY_PREDICTION: "non-empty"}
report = get_eval_report(spec, pred, log_fp, include_tests_status=False)
row = report.get(instance_id, {})
print(
    f"resolved={row.get('resolved')} "
    f"patch_successfully_applied={row.get('patch_successfully_applied')} "
    f"patch_exists={row.get('patch_exists')}"
)
PY
rm -f "${tmp_log}" || true

if [[ "${CLEANUP:-0}" == "1" ]]; then
  oc delete pod "${pod_name}" --ignore-not-found=true >/dev/null
  oc delete configmap "${cm_name}" --ignore-not-found=true >/dev/null
fi

if [[ "${phase}" != "Succeeded" ]]; then
  echo "ERROR: pod did not succeed (phase=${phase:-unknown})." >&2
  exit 1
fi
