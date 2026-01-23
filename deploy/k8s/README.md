## Kubernetes / IBM Cloud

This repo is intentionally light on cluster-specific tooling. The goal is: build one `mcode` container
image, then run sharded benchmark Jobs.

### 0) Inference backend (Mellea)

`mcode` talks to models via Mellea. You need a model backend endpoint reachable from the benchmark
Jobs.

For the default `ollama` backend, set `OLLAMA_HOST` (for example `http://ollama:11434`) in
`deploy/k8s/mcode-bench-indexed-job.yaml`.

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

### 2) Create a results PVC (recommended)

The Job template is configured to write one SQLite DB per shard into `/results/` via a PVC.
Create the PVC first and wait until it is `Bound` before starting the Job.

```bash
kubectl apply -f deploy/k8s/results-pvc.yaml
kubectl get pvc mcode-results -w
```

If your cluster supports RWX storage and you want true parallel sharded jobs, apply the RWX
variant instead and set a RWX storage class:

```bash
kubectl apply -f deploy/k8s/results-pvc-rwx.yaml
```

### 3) Run a sharded benchmark Job

#### Configure knobs (no YAML edits)

Edit `deploy/k8s/bench.env` and apply it as a ConfigMap:

```bash
kubectl create configmap mcode-bench-config --from-env-file=deploy/k8s/bench.env -o yaml --dry-run=client | kubectl apply -f -
```

#### Run

Edit `deploy/k8s/mcode-bench-indexed-job.yaml` once:

- set `image: ...`
- keep `spec.completions == SHARD_COUNT` (in `deploy/k8s/bench.env`)

If you are using a ReadWriteOnce (RWO) results PVC (common on block storage), set
`parallelism: 1` so shards run sequentially.

Then apply:

```bash
kubectl delete job mcode-bench --ignore-not-found
kubectl create -f deploy/k8s/mcode-bench-indexed-job.yaml
kubectl logs -f job/mcode-bench
```

### 4) Collect results

The Job writes one SQLite DB per shard into `/results/` (e.g. `humaneval-shard-0.db`).
If you used a PVC, those DBs persist after the Job completes.

You can copy them back locally and query them with `mcode results --db ...`:

```bash
kubectl cp <pod-name>:/results ./results
mcode results --db ./results/humaneval-shard-0.db --benchmark humaneval
```
