#!/usr/bin/env bash
set -euo pipefail

job_name="${1:-mcode-bench}"
out_dir="${OUT_DIR:-results-${job_name}-$(date +%Y%m%d-%H%M%S)}"
wait_timeout_s="${WAIT_TIMEOUT_S:-}"
wait_timeout="${WAIT_TIMEOUT:-}"
cleanup="${CLEANUP:-0}"

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: 'oc' is required (OpenShift CLI)." >&2
  exit 2
fi

mkdir -p "${out_dir}"

parse_duration_to_seconds() {
  # Accepts a subset of kubectl/oc durations like: 90s, 30m, 12h, 2h30m.
  # Returns seconds on stdout; non-zero on parse failure.
  python - "$1" <<'PY'
import re
import sys

s = sys.argv[1].strip()
if not s:
    raise SystemExit(1)

total = 0
pos = 0
for m in re.finditer(r"(\d+)([smh])", s):
    if m.start() != pos:
        raise SystemExit(1)
    n = int(m.group(1))
    unit = m.group(2)
    total += n * {"s": 1, "m": 60, "h": 3600}[unit]
    pos = m.end()

if pos != len(s) or total <= 0:
    raise SystemExit(1)

print(total)
PY
}

if [[ -z "${wait_timeout_s}" ]]; then
  if [[ -n "${wait_timeout}" ]]; then
    if ! wait_timeout_s="$(parse_duration_to_seconds "${wait_timeout}")"; then
      echo "ERROR: could not parse WAIT_TIMEOUT=${wait_timeout@Q} into seconds (set WAIT_TIMEOUT_S explicitly)." >&2
      exit 2
    fi
  else
    wait_timeout_s=1800
  fi
fi

pods="$(oc get pods -l "job-name=${job_name}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
completions="$(oc get job "${job_name}" -o jsonpath='{.spec.completions}' 2>/dev/null || true)"
completions="${completions:-0}"
if [[ -z "${completions}" || "${completions}" == "0" ]]; then
  echo "ERROR: could not determine .spec.completions for job ${job_name@Q}" >&2
  exit 1
fi

echo "Job ${job_name} completions=${completions}"

declare -A copied_idx=()
start_all_s="$(date +%s)"
timed_out=0

while (( ${#copied_idx[@]} < completions )); do
  now_s="$(date +%s)"
  if (( now_s - start_all_s >= wait_timeout_s )); then
    echo "ERROR: timed out waiting for all shards (copied=${#copied_idx[@]}/${completions})." >&2
    timed_out=1
    break
  fi

  pods="$(oc get pods -l "job-name=${job_name}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
  if [[ -z "${pods}" ]]; then
    sleep 2
    continue
  fi

  while IFS= read -r pod; do
    [[ -n "${pod}" ]] || continue
    idx="$(oc get pod "${pod}" -o go-template='{{index .metadata.annotations "batch.kubernetes.io/job-completion-index"}}' 2>/dev/null || true)"
    [[ -n "${idx}" ]] || continue
    if [[ -n "${copied_idx[${idx}]:-}" ]]; then
      continue
    fi

    phase="$(oc get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    if [[ "${phase}" == "Succeeded" ]]; then
      # The pod already finished (unexpected when the hold container is working). We can't copy anymore,
      # so treat it as done for this completion index.
      echo "WARNING: ${pod} already completed (phase=Succeeded); skipping copy." >&2
      copied_idx["${idx}"]=1
      continue
    fi
    if [[ "${phase}" == "Failed" ]]; then
      # For Indexed Jobs, a failed pod may be retried with the same completion index. Don't count it as copied.
      echo "WARNING: ${pod} failed; waiting for retry (idx=${idx})." >&2
      continue
    fi

    if ! oc exec "${pod}" -c hold -- test -f /results/_READY >/dev/null 2>&1; then
      # If the worker container has already terminated (e.g. OOMKilled) but didn't reach _READY,
      # still capture logs and release the hold container so the Job can retry this index.
      mcode_term_reason="$(oc get pod "${pod}" -o go-template='{{range .status.containerStatuses}}{{if eq .name "mcode"}}{{if .state.terminated}}{{.state.terminated.reason}}{{end}}{{end}}{{end}}' 2>/dev/null || true)"
      mcode_exit_code="$(oc get pod "${pod}" -o go-template='{{range .status.containerStatuses}}{{if eq .name "mcode"}}{{if .state.terminated}}{{.state.terminated.exitCode}}{{end}}{{end}}{{end}}' 2>/dev/null || true)"
      if [[ -n "${mcode_term_reason}" ]]; then
        echo "WARNING: ${pod} mcode terminated before _READY (idx=${idx} reason=${mcode_term_reason} exit=${mcode_exit_code:-?}); releasing hold to allow retry." >&2
        mkdir -p "${out_dir}/${pod}"
        oc logs "${pod}" -c mcode >"${out_dir}/${pod}/mcode.log" || true
        oc logs "${pod}" -c hold >"${out_dir}/${pod}/hold.log" || true
        # Best-effort copy of whatever exists.
        oc cp -c hold "${pod}:/results" "${out_dir}/${pod}" >/dev/null 2>&1 || true
        oc exec "${pod}" -c hold -- sh -lc "touch /results/_COPIED" >/dev/null 2>&1 || true
      fi
      continue
    fi

    echo "Copying /results from ${pod} -> ${out_dir}/${pod}/"
    mkdir -p "${out_dir}/${pod}"
    copied=0
    for attempt in 1 2 3; do
      if oc cp -c hold "${pod}:/results" "${out_dir}/${pod}"; then
        copied=1
        break
      fi
      sleep 2
    done
    if [[ "${copied}" != "1" ]]; then
      echo "WARNING: failed to copy results from ${pod} after 3 attempts; will retry later." >&2
      continue
    fi
    oc logs "${pod}" -c mcode >"${out_dir}/${pod}/mcode.log" || true
    oc logs "${pod}" -c hold >"${out_dir}/${pod}/hold.log" || true

    # Signal the hold container so this pod can complete and the Job can progress.
    oc exec "${pod}" -c hold -- sh -lc "touch /results/_COPIED" >/dev/null 2>&1 || true
    copied_idx["${idx}"]=1
    echo "Copied ${#copied_idx[@]}/${completions}"
  done <<<"${pods}"

  sleep 1
done

if [[ "${cleanup}" == "1" ]]; then
  echo "Cleaning up job ${job_name}..."
  oc delete job "${job_name}" --ignore-not-found=true >/dev/null || true
fi

echo "Done. Results copied to ${out_dir}/"

if [[ "${timed_out}" == "1" ]]; then
  exit 1
fi
