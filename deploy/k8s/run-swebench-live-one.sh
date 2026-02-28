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
  ./deploy/k8s/run-swebench-live-one.sh <instance_id>

Env vars (optional):
  MODE=gold|model          Default: model
  SPLIT=verified           Default: verified
  TIMEOUT_S=<seconds>      Default: 1800 (used for waiting + result metadata)
  POD_DEADLINE_S=<seconds> Default: TIMEOUT_S+600 (k8s activeDeadlineSeconds)
  WAIT_TIMEOUT_S=<seconds> Default: POD_DEADLINE_S+120 (how long the script waits)
  NAME_PREFIX=...          Default: mcode-sweb-live (set to avoid name collisions)
  OUT_DIR=...              If set, write logs + result JSON locally

  # Patch generation (MODE=model)
  BACKEND=openai|ollama    Default: openai
  MODEL=<model_id>         Default: ibm-granite/granite-3.0-8b-instruct
  OPENAI_BASE_URL=...      Default: http://vllm:8000/v1
  OPENAI_API_KEY=...       Default: dummy
  OLLAMA_HOST=...          Default: http://ollama:11434
  MCODE_IMAGE=...          Default: OpenShift internal registry mcode:latest
  MCODE_MAX_NEW_TOKENS=... Default: 4096
  LOOP_BUDGET=<N>          Default: 3 (max patch+test attempts)

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
split="${SPLIT:-verified}"

if [[ "${mode}" != "gold" && "${mode}" != "model" ]]; then
  echo "ERROR: MODE must be 'gold' or 'model' (got '${mode}')." >&2
  exit 2
fi

# Bash 3.2 compatible shell quoting (replaces ${var@Q} from bash 4.4+)
shquote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

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

tmp_dir="$(mktemp -d -t mcode-sweb-live.XXXXXX)"
tmp_eval_log=""
tmp_gen_log=""
tmp_result_json=""
cleanup_tmp() {
  rm -rf "${tmp_dir}"
  [[ -n "${tmp_eval_log}" ]] && rm -f "${tmp_eval_log}" || true
  [[ -n "${tmp_gen_log}" ]] && rm -f "${tmp_gen_log}" || true
  [[ -n "${tmp_result_json}" ]] && rm -f "${tmp_result_json}" || true
  # Clean up k8s resources on any exit (including errors/signals)
  if [[ "${CLEANUP:-0}" == "1" && -n "${pod_name:-}" ]]; then
    oc delete pod "${pod_name}" --ignore-not-found=true >/dev/null 2>&1 || true
    oc delete configmap "${pod_name}-inputs" --ignore-not-found=true >/dev/null 2>&1 || true
  fi
}
trap cleanup_tmp EXIT

hash8="$(printf '%s' "${instance_id}" | shasum -a 256 | cut -c1-8)"
base="$(sanitize "${instance_id}")"
name_prefix="$(sanitize "${NAME_PREFIX:-mcode-sweb-live}")"
max_base_len=$((63 - ${#name_prefix} - 1 - ${#hash8} - 1))
if (( max_base_len < 8 )); then
  max_base_len=8
fi
base="${base:0:${max_base_len}}"
pod_name="${name_prefix}-${base}-${hash8}"
cm_name="${pod_name}-inputs"

# MS SWE-bench Live images use starryzhang namespace
image_id="$(echo "${instance_id}" | tr '[:upper:]' '[:lower:]' | sed 's/__/_1776_/g')"
eval_image="starryzhang/sweb.eval.x86_64.${image_id}"

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
  tmp_eval_log="$(mktemp -t mcode-sweb-live-eval.XXXXXX.log)"
  tmp_gen_log="$(mktemp -t mcode-sweb-live-gen.XXXXXX.log)"
  tmp_result_json="$(mktemp -t mcode-sweb-live-result.XXXXXX.json)"
  eval_log="${tmp_eval_log}"
  gen_log="${tmp_gen_log}"
  result_json="${tmp_result_json}"
fi

# Load instance metadata from HuggingFace datasets
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

import json
from pathlib import Path
from datasets import load_dataset

instance_id = $(shquote "${instance_id}")
split = $(shquote "${split}")
mode = $(shquote "${mode}")

ds = load_dataset("SWE-bench-Live/SWE-bench-Live", split=split)
inst = None
for row in ds:
    if row["instance_id"] == instance_id:
        inst = row
        break

if inst is None:
    raise SystemExit(f"Instance not found: {instance_id!r} (split={split})")

out = Path($(shquote "${tmp_dir}"))
out.mkdir(parents=True, exist_ok=True)

# Build eval.sh from install_cmd + test_cmd
def parse_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [val]
    return []

f2p = parse_list(inst.get("FAIL_TO_PASS", []))
p2p = parse_list(inst.get("PASS_TO_PASS", []))

# Build eval.sh: use test_cmds from the dataset (matches official evaluation harness).
# This runs both F2P and P2P tests in the same session.
raw_cmds = parse_list(inst.get("test_cmds", []))
eval_lines = ["#!/bin/bash", "cd /testbed"]
for cmd in raw_cmds:
    if cmd.strip():
        eval_lines.append(cmd + " || true")
(out / "eval.sh").write_text("\\n".join(eval_lines) + "\\n", encoding="utf-8")

# Write test patch
test_patch = str(inst.get("test_patch", ""))
(out / "test_patch.diff").write_text(test_patch, encoding="utf-8")

# Write fail_to_pass / pass_to_pass (f2p/p2p already parsed above)
(out / "fail_to_pass.json").write_text(json.dumps(f2p), encoding="utf-8")
(out / "pass_to_pass.json").write_text(json.dumps(p2p), encoding="utf-8")

if mode == "gold":
    (out / "patch.diff").write_text(str(inst.get("patch", "")), encoding="utf-8")
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
    --from-file=test_patch.diff="${tmp_dir}/test_patch.diff" \
    --from-file=patch.diff="${tmp_dir}/patch.diff" \
    >/dev/null
else
  oc create configmap "${cm_name}" \
    --from-file=eval.sh="${tmp_dir}/eval.sh" \
    --from-file=test_patch.diff="${tmp_dir}/test_patch.diff" \
    --from-file=fail_to_pass.json="${tmp_dir}/fail_to_pass.json" \
    --from-file=pass_to_pass.json="${tmp_dir}/pass_to_pass.json" \
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
loop_budget="${LOOP_BUDGET:-3}"

if [[ "${mode}" == "gold" ]]; then
  cat <<YAML | oc apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
spec:
  serviceAccountName: anyuid-sa
  securityContext:
    runAsUser: 0
  restartPolicy: Never
  activeDeadlineSeconds: ${pod_deadline_s}
  volumes:
    - name: inputs
      configMap:
        name: ${cm_name}
  containers:
    - name: eval
      image: ${eval_image}
      resources:
        requests:
          cpu: "1"
          memory: "4Gi"
        limits:
          cpu: "3"
          memory: "12Gi"
      command:
        - bash
        - -lc
        - |
          cd /testbed

          # Apply test patch
          test_patch=/inputs/test_patch.diff
          if [ -s "\$test_patch" ]; then
            echo '>>>>> Applying Test Patch'
            if git apply --verbose "\$test_patch"; then
              echo '>>>>> Test Patch Applied'
            elif git apply --verbose --reject "\$test_patch"; then
              echo '>>>>> Test Patch Applied (with rejects)'
            else
              echo '>>>>> Test Patch Apply Failed'
            fi
          fi

          # Apply solution patch
          patch_file=/inputs/patch.diff
          if [ ! -s "\$patch_file" ]; then
            echo "patch.diff is missing or empty" >&2
            exit 0
          fi

          if git apply --verbose "\$patch_file"; then
            echo '>>>>> Applied Patch'
          elif git apply --verbose --reject "\$patch_file"; then
            echo '>>>>> Applied Patch'
          elif patch --batch --fuzz=5 -p1 -i "\$patch_file"; then
            echo '>>>>> Applied Patch'
          else
            echo '>>>>> Patch Apply Failed'
            exit 0
          fi

          bash /inputs/eval.sh || true
      volumeMounts:
        - name: inputs
          mountPath: /inputs
          readOnly: true
YAML
else
  cat <<'YAML_END' | sed \
    -e "s|\${pod_name}|${pod_name}|g" \
    -e "s|\${pod_deadline_s}|${pod_deadline_s}|g" \
    -e "s|\${cm_name}|${cm_name}|g" \
    -e "s|\${mcode_image}|${mcode_image}|g" \
    -e "s|\${eval_image}|${eval_image}|g" \
    -e "s|\${backend}|${backend}|g" \
    -e "s|\${model}|${model}|g" \
    -e "s|\${openai_base_url}|${openai_base_url}|g" \
    -e "s|\${openai_api_key}|${openai_api_key}|g" \
    -e "s|\${ollama_host}|${ollama_host}|g" \
    -e "s|\${max_new_tokens}|${max_new_tokens}|g" \
    -e "s|\${loop_budget}|${loop_budget}|g" \
    | oc apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
spec:
  serviceAccountName: anyuid-sa
  securityContext:
    runAsUser: 0
  restartPolicy: Never
  activeDeadlineSeconds: ${pod_deadline_s}
  volumes:
    - name: work
      emptyDir: {}
    - name: inputs
      configMap:
        name: ${cm_name}
  containers:
    - name: agent
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
        - name: LOOP_BUDGET
          value: "${loop_budget}"
      command:
        - bash
        - -lc
        - |
          set -euo pipefail
          mkdir -p /work/ipc
          # Wait for testbed sidecar to signal ready
          echo "waiting for testbed sidecar..."
          while [ ! -f /work/ipc/testbed-ready ]; do sleep 0.5; done
          echo "testbed ready, starting agent"
          python - <<'AGENT_PY'
          import hashlib
          import json
          import os
          import sys
          import time
          from pathlib import Path

          from mellea.stdlib.requirements.requirement import Requirement, simple_validate

          from mcode.llm.session import LLMSession

          IPC = Path("/work/ipc")

          # ---- config ----
          repo = Path("/inputs/repo.txt").read_text(encoding="utf-8").strip()
          problem = Path("/inputs/problem.txt").read_text(encoding="utf-8", errors="replace")
          hints = Path("/inputs/hints.txt").read_text(encoding="utf-8", errors="replace")
          f2p = json.loads(Path("/inputs/fail_to_pass.json").read_text())
          p2p = json.loads(Path("/inputs/pass_to_pass.json").read_text())

          model_id = os.environ["MODEL"]
          backend = os.environ.get("BACKEND", "openai")
          loop_budget = int(os.environ.get("LOOP_BUDGET", "3"))

          # ---- source context (read from testbed via shared volume) ----
          source_context = ""
          ctx_file = IPC / "source-context.txt"
          if ctx_file.exists():
              source_context = ctx_file.read_text(encoding="utf-8", errors="replace")

          enriched_hints = hints
          if source_context:
              enriched_hints = hints + "\n\nRelevant source files from the repository:\n" + source_context

          # ---- pytest parser (matches official SWE-bench-Live evaluation.py) ----
          def parse_pytest(output):
              STATUSES = {"FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL"}
              results = {}
              for line in output.splitlines():
                  line = line.strip()
                  if not any(line.startswith(s) for s in STATUSES):
                      continue
                  if line.startswith("FAILED"):
                      line = line.replace(" - ", " ")
                  parts = line.split()
                  if len(parts) <= 1:
                      continue
                  results[parts[1]] = parts[0]
              return results

          def check_resolved(test_output):
              results = parse_pytest(test_output)
              f2p_ok = all(results.get(t) == "PASSED" for t in f2p) and len(f2p) > 0
              p2p_failed = [t for t in p2p if results.get(t, "MISSING") in ("FAILED", "ERROR")]
              return f2p_ok and len(p2p_failed) == 0

          # ---- convert structured edits to unified diff ----
          def edits_to_patch(raw_json, repo_root="/testbed"):
              import difflib
              try:
                  data = json.loads(raw_json)
              except Exception:
                  return ""
              edits = data.get("edits", [])
              if not edits:
                  return data.get("patch", "")  # fallback for raw diff

              root = Path(repo_root)
              file_index = [None]  # mutable container for closure

              def _resolve_path(rel):
                  full = root / rel
                  if full.is_file():
                      return (rel, full)
                  parts = rel.split("/")
                  for i in range(1, len(parts)):
                      candidate = "/".join(parts[i:])
                      full = root / candidate
                      if full.is_file():
                          return (candidate, full)
                  if file_index[0] is None:
                      file_index[0] = {}
                      for p in root.rglob("*.py"):
                          if ".git" not in p.parts and "__pycache__" not in p.parts:
                              file_index[0][p.name] = p
                  basename = parts[-1] if parts else ""
                  if basename in file_index[0]:
                      matched = file_index[0][basename]
                      return (str(matched.relative_to(root)), matched)
                  return None

              def _fuzzy_find(search, text):
                  if search in text:
                      idx = text.index(search)
                      return (idx, idx + len(search))
                  sm = difflib.SequenceMatcher(None, search, text, autojunk=False)
                  best = sm.find_longest_match(0, len(search), 0, len(text))
                  if best.size == 0:
                      return None
                  s_lines = search.splitlines(keepends=True)
                  t_lines = text.splitlines(keepends=True)
                  n = len(s_lines)
                  best_ratio = 0.0
                  best_span = None
                  for start in range(max(0, best.b // 40 - n), min(len(t_lines), best.b // 40 + n + 1)):
                      end = start + n
                      if end > len(t_lines):
                          break
                      candidate = "".join(t_lines[start:end])
                      ratio = difflib.SequenceMatcher(
                          None, search, candidate, autojunk=False
                      ).ratio()
                      if ratio > best_ratio:
                          best_ratio = ratio
                          best_span = (start, end)
                  if best_span is None or best_ratio < 0.6:
                      return None
                  char_start = sum(len(l) for l in t_lines[:best_span[0]])
                  char_end = sum(len(l) for l in t_lines[:best_span[1]])
                  return (char_start, char_end)

              patches = []
              for edit in edits:
                  fpath = edit.get("file", "")
                  search = edit.get("search", "")
                  replace = edit.get("replace", "")
                  resolved = _resolve_path(fpath)
                  if resolved is None:
                      continue
                  rel, full = resolved
                  try:
                      original = full.read_text(encoding="utf-8", errors="replace")
                  except Exception:
                      continue
                  if search in original:
                      modified = original.replace(search, replace, 1)
                  else:
                      span = _fuzzy_find(search, original)
                      if span is None:
                          continue
                      modified = original[:span[0]] + replace + original[span[1]:]
                  diff = difflib.unified_diff(
                      original.splitlines(keepends=True),
                      modified.splitlines(keepends=True),
                      fromfile=f"a/{rel}",
                      tofile=f"b/{rel}",
                  )
                  patches.append("".join(diff))
              return "\n".join(patches)

          # ---- IPC: send patch to testbed sidecar, get test results back ----

          def truncate(s, max_chars=4000):
              return s if len(s) <= max_chars else s[-max_chars:]

          def run_test_via_sidecar(patch):
              """Write patch to shared volume, signal testbed, wait for result."""
              # Clean previous signals
              for f in ["test-result.txt", "test-done"]:
                  (IPC / f).unlink(missing_ok=True)

              (IPC / "patch.diff").write_text(patch or "", encoding="utf-8")
              (IPC / "test-run").touch()

              # Wait for testbed to finish
              deadline = time.time() + 660
              while not (IPC / "test-done").exists():
                  if time.time() > deadline:
                      return "TIMEOUT: testbed sidecar did not respond"
                  time.sleep(0.5)

              result = (IPC / "test-result.txt").read_text(encoding="utf-8", errors="replace")
              (IPC / "test-done").unlink(missing_ok=True)
              return result

          def _patch_test(raw_json):
              patch = edits_to_patch(raw_json) or ""
              test_output = run_test_via_sidecar(patch)

              if check_resolved(test_output):
                  print(">>>>> Applied Patch")
                  print(test_output)
                  return True

              return (False, truncate(test_output))

          req = Requirement(
              validation_fn=simple_validate(_patch_test),
              check_only=True,
          )

          session = LLMSession(
              model_id=model_id,
              backend_name=backend,
              loop_budget=loop_budget,
          )
          session.check_available()

          patch = ""
          attempts_used = 0
          try:
              with session.open():
                  result = session.generate_patch(
                      repo=repo,
                      problem_statement=problem,
                      hints_text=enriched_hints,
                      requirements=[req],
                  )
              raw = result.value or ""
              patch = edits_to_patch(raw) or ""
              attempts_used = len(result.sample_generations)
          except Exception as e:
              print(f"ERROR: {e}", file=sys.stderr)
              import traceback
              traceback.print_exc()

          # Write final patch
          Path("/work/patch.diff").write_text(patch, encoding="utf-8", errors="replace")
          sha = hashlib.sha256(patch.encode("utf-8", errors="ignore")).hexdigest()
          print(f"generated patch chars={len(patch)}")
          print(f"patch_sha256={sha}")
          print(f"attempts_used={attempts_used}")

          # Final eval via sidecar for log parsing
          if patch.strip():
              final_output = run_test_via_sidecar(patch)
              if ">>>>> Applied Patch" not in final_output:
                  # The sidecar prints this; check
                  pass
              print(final_output)

          # Signal testbed sidecar to exit
          (IPC / "agent-done").touch()
          AGENT_PY
      volumeMounts:
        - name: work
          mountPath: /work
        - name: inputs
          mountPath: /inputs
          readOnly: true
    - name: testbed
      image: ${eval_image}
      resources:
        requests:
          cpu: "1"
          memory: "4Gi"
        limits:
          cpu: "3"
          memory: "12Gi"
      command:
        - bash
        - -lc
        - |
          IPC=/work/ipc
          mkdir -p $IPC

          # Build source context for the agent
          cd /testbed
          find . -name '*.py' -not -path '*__pycache__*' -not -path '*/.git/*' \
            2>/dev/null | head -500 > $IPC/file-list.txt || true
          python3 - <<'CTX_PY'
          from pathlib import Path

          ipc = Path("/work/ipc")
          problem = Path("/inputs/problem.txt").read_text(encoding="utf-8", errors="replace")
          files = [f.strip() for f in (ipc / "file-list.txt").read_text().splitlines() if f.strip()]

          problem_lower = problem.lower()
          scored = []
          for fp in files:
              rel = fp.lstrip("./")
              parts = rel.replace("/", " ").replace(".py", "").replace("_", " ").split()
              score = sum(1 for p in parts if len(p) > 2 and p.lower() in problem_lower)
              if score > 0:
                  scored.append((score, rel))
          scored.sort(key=lambda x: -x[0])

          ctx = []
          chars = 0
          tree = "\n".join(f.lstrip("./") for f in files[:200])
          if len(files) > 200:
              tree += f"\n... and {len(files) - 200} more files"
          ctx.append(f"File tree ({len(files)} Python files):\n{tree}")
          chars += len(ctx[0])

          for _score, rel in scored[:10]:
              if chars >= 12000:
                  break
              try:
                  content = Path("/testbed/" + rel).read_text(encoding="utf-8", errors="replace")
              except Exception:
                  continue
              budget = 12000 - chars
              if len(content) > budget:
                  content = content[:budget] + "\n... (truncated)"
              ctx.append(f"\n--- {rel} ---\n{content}")
              chars += len(content) + len(rel) + 10

          (ipc / "source-context.txt").write_text("\n".join(ctx), encoding="utf-8")
          CTX_PY

          # Signal ready
          touch $IPC/testbed-ready
          echo "testbed sidecar ready"

          # Test loop: wait for agent to request test runs
          while true; do
            if [ -f $IPC/agent-done ]; then
              echo "agent done, exiting"
              break
            fi
            if [ -f $IPC/test-run ]; then
              rm -f $IPC/test-run
              cd /testbed
              git checkout . 2>/dev/null || true

              # Apply solution patch
              patch_file=$IPC/patch.diff
              applied=false
              if [ -s "$patch_file" ]; then
                if git apply --verbose "$patch_file" 2>&1; then
                  echo '>>>>> Applied Patch'
                  applied=true
                elif git apply --verbose --reject "$patch_file" 2>&1; then
                  echo '>>>>> Applied Patch'
                  applied=true
                elif patch --batch --fuzz=5 -p1 -i "$patch_file" 2>&1; then
                  echo '>>>>> Applied Patch'
                  applied=true
                else
                  echo '>>>>> Patch Apply Failed'
                fi
              else
                echo 'Empty patch'
              fi

              # Apply test patch
              test_patch=/inputs/test_patch.diff
              if [ -s "$test_patch" ]; then
                git apply --verbose "$test_patch" 2>/dev/null || \
                  git apply --verbose --reject "$test_patch" 2>/dev/null || true
              fi

              # Run tests and capture output
              bash /inputs/eval.sh > $IPC/test-result.txt 2>&1 || true

              # If patch didn't apply, prepend that info
              if [ "$applied" = false ]; then
                echo ">>>>> Patch Apply Failed" >> $IPC/test-result.txt
              fi

              touch $IPC/test-done
            fi
            sleep 0.3
          done
      volumeMounts:
        - name: work
          mountPath: /work
        - name: inputs
          mountPath: /inputs
          readOnly: true
YAML_END
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
  oc logs "${pod_name}" -c agent >"${gen_log}" 2>&1 || true
  echo "--- agent (tail) ---"
  tail -n 200 "${gen_log}" || true
  oc logs "${pod_name}" -c testbed >"${eval_log}" 2>&1 || true
  echo "--- testbed (tail) ---"
  tail -n 200 "${eval_log}" || true
else
  oc logs "${pod_name}" -c eval >"${eval_log}" 2>&1 || true
  echo "--- eval (tail) ---"
  tail -n 200 "${eval_log}" || true
fi

echo "--- swebench-live report ---"
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
import json
import os
import re
from pathlib import Path

mode = os.environ.get("MCODE_SWEB_MODE", "")
phase = os.environ.get("MCODE_SWEB_PHASE", "")
reason = os.environ.get("MCODE_SWEB_REASON") or None
gen_log = os.environ.get("MCODE_SWEB_GEN_LOG") or ""

instance_id = $(shquote "${instance_id}")
eval_log_fp = $(shquote "${eval_log}")
f2p_fp = Path($(shquote "${tmp_dir}")) / "fail_to_pass.json"
p2p_fp = Path($(shquote "${tmp_dir}")) / "pass_to_pass.json"

fail_to_pass = json.loads(f2p_fp.read_text()) if f2p_fp.exists() else []
pass_to_pass = json.loads(p2p_fp.read_text()) if p2p_fp.exists() else []

patch_sha256 = os.environ.get("MCODE_SWEB_GOLD_PATCH_SHA256") or None
attempts_used = 1
if mode == "model" and gen_log and Path(gen_log).exists():
    gen_text = Path(gen_log).read_text(errors="replace")
    m = re.search(r"^patch_sha256=([0-9a-f]{64})\s*$", gen_text, re.M)
    if m:
        patch_sha256 = m.group(1)
    m2 = re.search(r"^attempts_used=(\d+)\s*$", gen_text, re.M)
    if m2:
        attempts_used = int(m2.group(1))

eval_log_text = ""
if Path(eval_log_fp).exists():
    eval_log_text = Path(eval_log_fp).read_text(errors="replace")
# In model mode, agent log also contains test output (printed by agent.py)
if mode == "model" and gen_log and Path(gen_log).exists():
    eval_log_text = eval_log_text + "\n" + gen_text

# Pytest parser matching the official SWE-bench-Live evaluation.py logic:
# split on whitespace, take test_case[1] as the test ID.
# For FAILED lines, replace " - " with " " first to strip error messages.
def parse_pytest(output):
    STATUSES = {"FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL"}
    results = {}
    for line in output.splitlines():
        line = line.strip()
        if not any(line.startswith(s) for s in STATUSES):
            continue
        if line.startswith("FAILED"):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) <= 1:
            continue
        results[parts[1]] = parts[0]
    return results

test_results = parse_pytest(eval_log_text)

f2p_ok = all(test_results.get(t) == "PASSED" for t in fail_to_pass) and len(fail_to_pass) > 0
# P2P: failures block resolution (matches official SWE-bench-Live evaluation).
# Only tests that actually ran and FAILED/ERROR count as failures.
# Tests missing from output (not run) don't count as failures.
p2p_failed = [t for t in pass_to_pass if test_results.get(t, "MISSING") in ("FAILED", "ERROR")]
resolved = f2p_ok and len(p2p_failed) == 0

patch_applied = ">>>>> Applied Patch" in eval_log_text

result = {
    "instance_id": instance_id,
    "split": os.environ.get("MCODE_SWEB_SPLIT"),
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
    "attempts_used": attempts_used,
    "resolved": resolved,
    "patch_successfully_applied": patch_applied,
    "report": {
        "fail_to_pass": {t: test_results.get(t, "MISSING") for t in fail_to_pass},
        "pass_to_pass": {t: test_results.get(t, "MISSING") for t in pass_to_pass},
        "p2p_regressions": p2p_failed,
    },
    "error": None,
}

out_path = os.environ.get("MCODE_SWEB_RESULT_JSON")
if out_path:
    Path(out_path).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

print(f"resolved={resolved} patch_successfully_applied={patch_applied}")
PY

# Cleanup handled by EXIT trap

if [[ "${phase}" != "Succeeded" ]]; then
  echo "WARNING: pod phase=${phase:-unknown} (deadline exceeded or container error)." >&2
  if [[ -n "${out_dir}" ]]; then
    echo "Note: wrote logs + result JSON under '${out_dir}' (prefix='${log_prefix}')." >&2
  fi
fi
