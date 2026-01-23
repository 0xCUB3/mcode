## Kubernetes / IBM Cloud

This repo is intentionally light on cluster-specific tooling. The goal is: build one `mcode` container
image, then run sharded benchmark Jobs.

### 0) Inference backend (Mellea)

`mcode` talks to models via Mellea. You need a model backend endpoint reachable from the benchmark
Jobs.

This cluster exposes both:
- `service/vllm` (OpenAI-compatible; best throughput with parallel shards)
- `service/ollama` (simple, but easier to bottleneck under load)

Pick one in `deploy/k8s/bench.env`.

### 1) Build + push the `mcode` image

Use any registry your cluster can pull from (IBM Container Registry, OpenShift internal registry, etc).

Example (IBM Container Registry):

```bash
# login + choose your registry namespace first
ibmcloud cr login
ibmcloud cr namespace-add <icr-namespace>

docker build -t icr.io/<icr-namespace>/mcode:latest .
docker push icr.io/<icr-namespace>/mcode:latest
```

### 2) Configure knobs (no YAML edits)

Edit `deploy/k8s/bench.env` and apply it as a ConfigMap:

```bash
oc create configmap mcode-bench-config --from-env-file=deploy/k8s/bench.env -o yaml --dry-run=client | oc apply -f -
```

Notes:
- `MCODE_MAX_NEW_TOKENS` caps output length to prevent rare runaway generations (which can OOM pods).
- In OpenShift, `PARALLELISM` is also limited by your namespace ResourceQuota. `run-bench.sh` will
  clamp it down automatically if needed.

### 3) Run a sharded benchmark Job

The simplest way to submit the Job is:

```bash
./deploy/k8s/run-bench.sh
```

Defaults:
- Writes shard DBs to an ephemeral `/results` (`emptyDir`), so shards can run concurrently without
  any storage provisioning.
- Uses OpenShift internal registry image:
  `image-registry.openshift-image-registry.svc:5000/<current-namespace>/mcode:latest`

Common flags:

```bash
# Copy results back locally after the Job completes (one folder per pod under ./results-.../)
FETCH_RESULTS=1 ./deploy/k8s/run-bench.sh

# Run fewer shards at once (useful if your model server is the bottleneck)
PARALLELISM=2 ./deploy/k8s/run-bench.sh

# Quick one-off overrides without editing `bench.env` (prefixed with `OVERRIDE_`)
OVERRIDE_LIMIT=10 OVERRIDE_SHARD_COUNT=2 PARALLELISM=2 ./deploy/k8s/run-bench.sh

# If a previous run is still active, `run-bench.sh` refuses to delete it.
# To delete + recreate anyway:
FORCE_RECREATE=1 ./deploy/k8s/run-bench.sh
```

If you see `OOMKilled` pods, either lower `PARALLELISM` or increase the container memory request/limit
in `deploy/k8s/mcode-bench-indexed-job.yaml`.

### 4) Query results locally

If you used `FETCH_RESULTS=1`, query any shard DB:

```bash
mcode results --db ./results-*/<pod-name>/results/humaneval-shard-0.db --benchmark humaneval
```

If you already ran the Job and just want to copy results without resubmitting:

```bash
./deploy/k8s/fetch-results.sh mcode-bench
```
