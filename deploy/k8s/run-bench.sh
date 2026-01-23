#!/usr/bin/env bash
set -euo pipefail

job_name="${JOB_NAME:-mcode-bench}"
manifest="deploy/k8s/mcode-bench-indexed-job.yaml"

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: 'oc' is required (OpenShift CLI)." >&2
  exit 2
fi

namespace="$(oc project -q)"

tmp_env="$(mktemp -t mcode-bench.XXXXXX.env)"
trap 'rm -f "${tmp_env}"' EXIT
bench_env="deploy/k8s/bench.env"

override_keys=()
override_lines=()
for key in BENCHMARK MODEL BACKEND OLLAMA_HOST OPENAI_BASE_URL OPENAI_API_KEY MCODE_MAX_NEW_TOKENS SAMPLES DEBUG_ITERS TIMEOUT_S SHARD_COUNT LIMIT; do
  override="OVERRIDE_${key}"
  if [[ -n "${!override:-}" ]]; then
    override_keys+=("${key}")
    override_lines+=("$(printf '%s=%s' "${key}" "${!override}")")
  fi
done

if (( ${#override_keys[@]} == 0 )); then
  cp "${bench_env}" "${tmp_env}"
else
  # `oc create configmap --from-env-file` rejects duplicate keys, so we "merge" by removing the
  # overridden keys from the base file and then appending the override values exactly once.
  keys_csv="$(IFS=,; echo "${override_keys[*]}")"
  awk -v keys_csv="${keys_csv}" '
    BEGIN {
      n = split(keys_csv, a, ",");
      for (i = 1; i <= n; i++) drop[a[i]] = 1;
    }
    {
      if (match($0, /^[A-Za-z_][A-Za-z0-9_]*=/)) {
        k = substr($0, RSTART, RLENGTH - 1);
        if (drop[k] == 1) next;
      }
      print $0;
    }
  ' "${bench_env}" >"${tmp_env}"

  for line in "${override_lines[@]}"; do
    printf '%s\n' "${line}" >>"${tmp_env}"
  done
fi

set -a
# shellcheck source=/dev/null
source "${tmp_env}"
set +a

if [[ -z "${SHARD_COUNT:-}" ]]; then
  echo "ERROR: SHARD_COUNT must be set in deploy/k8s/bench.env" >&2
  exit 2
fi

parallelism="${PARALLELISM:-$SHARD_COUNT}"

image_default="image-registry.openshift-image-registry.svc:5000/${namespace}/mcode:latest"
image="${MCODE_IMAGE:-$image_default}"

# This namespace has a ResourceQuota; keep requested resources small enough so the scheduler can
# actually create multiple pods. Our default per-pod requests are:
pod_req_cpu_m=500
pod_req_mem_gi=2

quota_line="$(oc get resourcequota -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.hard.cpu}{"\t"}{.status.used.cpu}{"\t"}{.status.hard.memory}{"\t"}{.status.used.memory}{"\n"}{end}' | head -n 1)"
if [[ -n "${quota_line}" ]]; then
  quota_name="$(echo "${quota_line}" | cut -f1)"
  hard_cpu="$(echo "${quota_line}" | cut -f2)"
  used_cpu="$(echo "${quota_line}" | cut -f3)"
  hard_mem="$(echo "${quota_line}" | cut -f4)"
  used_mem="$(echo "${quota_line}" | cut -f5)"

  # Convert quota CPU to millicores.
  cpu_to_m() {
    local v="$1"
    if [[ "${v}" == *m ]]; then
      echo "${v%m}"
    else
      python - <<PY
v=float("${v}")
print(int(v*1000))
PY
    fi
  }
  mem_to_gi() {
    local v="$1"
    python - <<PY
import re
s="${v}".strip()
m=re.match(r"^([0-9]+)([KMGTP]i)?$", s)
if not m:
    raise SystemExit(1)
n=int(m.group(1)); unit=m.group(2) or ""
scale={"Ki":1/1024/1024,"Mi":1/1024,"Gi":1,"Ti":1024,"Pi":1024*1024}.get(unit, 1/1024/1024/1024)
print(n*scale)
PY
  }

  hard_cpu_m="$(cpu_to_m "${hard_cpu}")"
  used_cpu_m="$(cpu_to_m "${used_cpu}")"
  hard_mem_gi="$(mem_to_gi "${hard_mem}")"
  used_mem_gi="$(mem_to_gi "${used_mem}")"

  avail_cpu_m=$(( hard_cpu_m - used_cpu_m ))
  avail_mem_gi="$(python - <<PY
print(max(0.0, float("${hard_mem_gi}") - float("${used_mem_gi}")))
PY
)"

  max_by_cpu=$(( avail_cpu_m / pod_req_cpu_m ))
  max_by_mem="$(python - <<PY
import math
avail=float("${avail_mem_gi}")
print(int(math.floor(avail/float("${pod_req_mem_gi}"))))
PY
)"

  max_parallelism="${max_by_cpu}"
  if (( max_by_mem < max_parallelism )); then
    max_parallelism="${max_by_mem}"
  fi
  if (( max_parallelism < 1 )); then
    max_parallelism=1
  fi
  if (( parallelism > max_parallelism )); then
    echo "NOTE: clamping PARALLELISM=${parallelism} -> ${max_parallelism} due to ResourceQuota (${quota_name})."
    parallelism="${max_parallelism}"
  fi
fi

echo "Namespace:      ${namespace}"
echo "Job:            ${job_name}"
echo "Image:          ${image}"
echo "Completions:    ${SHARD_COUNT}"
echo "Parallelism:    ${parallelism}"

oc create configmap mcode-bench-config \
  --from-env-file="${tmp_env}" \
  -o yaml --dry-run=client \
  | oc apply -f -

if oc get job "${job_name}" >/dev/null 2>&1; then
  active="$(oc get job "${job_name}" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  active="${active:-0}"
  if [[ "${active}" != "0" && "${FORCE_RECREATE:-0}" != "1" ]]; then
    echo "ERROR: job ${job_name} is still running (active=${active})." >&2
    echo "Set FORCE_RECREATE=1 to delete + recreate it." >&2
    exit 1
  fi
  echo "Deleting existing job ${job_name}..."
  oc delete job "${job_name}"
fi

sed -E \
  -e "s#^([[:space:]]*completions:).*#\\1 ${SHARD_COUNT}#" \
  -e "s#^([[:space:]]*parallelism:).*#\\1 ${parallelism}#" \
  -e "s#^([[:space:]]*image:).*#\\1 ${image}#" \
  "${manifest}" \
  | oc create -f -

if [[ "${WAIT:-1}" == "1" ]]; then
  wait_timeout="${WAIT_TIMEOUT:-2h}"
  oc wait --for=condition=complete "job/${job_name}" --timeout="${wait_timeout}"
fi

if [[ "${FETCH_RESULTS:-0}" == "1" ]]; then
  ./deploy/k8s/fetch-results.sh "${job_name}"
fi
