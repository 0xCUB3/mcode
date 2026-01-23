#!/usr/bin/env bash
set -euo pipefail

job_name="${1:-mcode-bench}"
out_dir="${OUT_DIR:-results-${job_name}-$(date +%Y%m%d-%H%M%S)}"

if ! command -v oc >/dev/null 2>&1; then
  echo "ERROR: 'oc' is required (OpenShift CLI)." >&2
  exit 2
fi

mkdir -p "${out_dir}"

pods="$(oc get pods -l "job-name=${job_name}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')"
if [[ -z "${pods}" ]]; then
  echo "ERROR: no pods found for job ${job_name}" >&2
  exit 1
fi

echo "${pods}" | while IFS= read -r pod; do
  [[ -n "${pod}" ]] || continue
  phase="$(oc get pod "${pod}" -o jsonpath='{.status.phase}')"
  if [[ "${phase}" != "Succeeded" && "${phase}" != "Failed" ]]; then
    echo "Skipping ${pod} (phase=${phase})"
    continue
  fi
  echo "Copying /results from ${pod} -> ${out_dir}/${pod}/"
  mkdir -p "${out_dir}/${pod}"
  oc cp "${pod}:/results" "${out_dir}/${pod}"
  oc logs "${pod}" >"${out_dir}/${pod}/pod.log" || true
done

echo "Done. Results copied to ${out_dir}/"

