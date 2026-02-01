#!/usr/bin/env bash
set -euo pipefail

job_name="${1:-mcode-bench}"
out_dir="${OUT_DIR:-results-${job_name}-$(date +%Y%m%d-%H%M%S)}"
wait_timeout_s="${WAIT_TIMEOUT_S:-1800}"
cleanup="${CLEANUP:-0}"

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: 'oc' is required (OpenShift CLI)." >&2
  exit 2
fi

mkdir -p "${out_dir}"

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

while (( ${#copied_idx[@]} < completions )); do
  now_s="$(date +%s)"
  if (( now_s - start_all_s >= wait_timeout_s )); then
    echo "ERROR: timed out waiting for all shards (copied=${#copied_idx[@]}/${completions})." >&2
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
    if [[ "${phase}" == "Succeeded" || "${phase}" == "Failed" ]]; then
      # Shouldn't happen (hold container keeps pods running), but don't spin forever.
      echo "WARNING: ${pod} already completed (phase=${phase}); skipping copy." >&2
      copied_idx["${idx}"]=1
      continue
    fi

    if ! oc exec "${pod}" -c hold -- test -f /results/_READY >/dev/null 2>&1; then
      continue
    fi

    echo "Copying /results from ${pod} -> ${out_dir}/${pod}/"
    mkdir -p "${out_dir}/${pod}"
    oc cp -c hold "${pod}:/results" "${out_dir}/${pod}" || true
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
