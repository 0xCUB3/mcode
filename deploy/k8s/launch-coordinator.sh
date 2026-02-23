#!/usr/bin/env bash
set -euo pipefail

# Launch the sweep coordinator as an OC job so it survives terminal disconnects.
# All arguments after "--" are forwarded to oc_bench_sweep.py.
#
# Usage:
#   ./deploy/k8s/launch-coordinator.sh -- \
#     --benchmarks bigcodebench-complete,bigcodebench-instruct \
#     --model granite4:latest --loop-budget 1,3,5 --timeout 60,120 \
#     --shard-count 20 --parallelism 3 --no-build --resume \
#     --run-id benchmark-expansion --out-dir /results/benchmark-expansion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-$(oc project -q)}"
JOB_NAME="${COORDINATOR_JOB_NAME:-mcode-coordinator}"
IMAGE_NAME="${COORDINATOR_IMAGE:-mcode-coordinator}"
BUILD_CONTEXT="$(mktemp -d -t mcode-coordinator-build.XXXXXX)"
trap 'rm -rf "${BUILD_CONTEXT}"' EXIT
cp "${SCRIPT_DIR}/Dockerfile.coordinator" "${BUILD_CONTEXT}/Dockerfile"
cp "${SCRIPT_DIR}/oc_bench_sweep.py" "${BUILD_CONTEXT}/oc_bench_sweep.py"

# Collect sweep args: everything after "--"
SWEEP_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      SWEEP_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option before '--': $1" >&2
      echo "Usage: $0 [--] <sweep-args...>" >&2
      exit 1
      ;;
  esac
done

echo "==> Namespace: ${NAMESPACE}"
echo "==> Applying RBAC and PVC (idempotent) ..."
oc apply -n "${NAMESPACE}" -f "${SCRIPT_DIR}/coordinator-rbac.yaml"
oc apply -n "${NAMESPACE}" -f "${SCRIPT_DIR}/coordinator-pvc.yaml"

# Build coordinator image via OpenShift BuildConfig
if oc get buildconfig "${IMAGE_NAME}" -n "${NAMESPACE}" &>/dev/null; then
  echo "==> Starting build (BuildConfig exists) ..."
  oc start-build "${IMAGE_NAME}" \
    --from-dir="${BUILD_CONTEXT}" \
    --follow \
    -n "${NAMESPACE}"
else
  echo "==> Creating BuildConfig and building ..."
  oc new-build \
    --name="${IMAGE_NAME}" \
    --binary \
    --strategy=docker \
    --dockerfile="$(cat "${SCRIPT_DIR}/Dockerfile.coordinator")" \
    -n "${NAMESPACE}" || true
  oc start-build "${IMAGE_NAME}" \
    --from-dir="${BUILD_CONTEXT}" \
    --follow \
    -n "${NAMESPACE}"
fi

COORDINATOR_IMAGE="image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:latest"

# Delete old coordinator job if it exists
oc delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found

# Build the args array as a JSON list for the job YAML
ARGS_JSON="["
first=true
for arg in "${SWEEP_ARGS[@]}"; do
  if [ "$first" = true ]; then
    first=false
  else
    ARGS_JSON+=","
  fi
  # Escape quotes in arg value
  escaped=$(printf '%s' "$arg" | sed 's/"/\\"/g')
  ARGS_JSON+="\"${escaped}\""
done
ARGS_JSON+="]"

# Create coordinator job inline
cat <<EOF | oc apply -n "${NAMESPACE}" -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 0
  template:
    spec:
      serviceAccountName: mcode-coordinator
      restartPolicy: Never
      containers:
        - name: coordinator
          image: ${COORDINATOR_IMAGE}
          imagePullPolicy: Always
          args: ${ARGS_JSON}
          volumeMounts:
            - name: results
              mountPath: /results
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            runAsNonRoot: true
            seccompProfile:
              type: RuntimeDefault
      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: mcode-sweep-results
EOF

echo ""
echo "==> Coordinator job created: ${JOB_NAME}"
echo ""
echo "Monitor:"
echo "  oc logs -f job/${JOB_NAME} -n ${NAMESPACE}"
echo ""
echo "Get results when done:"
echo "  oc rsync \$(oc get pod -l job-name=${JOB_NAME} -n ${NAMESPACE} -o name):/results/ ./results/"
