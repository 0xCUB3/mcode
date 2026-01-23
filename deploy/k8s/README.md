## Kubernetes / IBM Cloud

This repo is intentionally light on cluster-specific tooling. The goal is: build one `mcode` container
image, then run sharded benchmark Jobs.

### 0) Inference backend (Mellea)

`mcode` talks to models via Mellea. The default backend is `ollama`, so you need an Ollama endpoint
reachable from the benchmark Jobs.

Option A: run Ollama in-cluster:

```bash
kubectl apply -f deploy/k8s/ollama.yaml
```

Then point `OLLAMA_HOST` to `http://ollama:11434` in `deploy/k8s/mcode-bench-indexed-job.yaml`.

Option B: use an external endpoint:

- Set `OLLAMA_HOST` to your external URL (must be reachable from the cluster).

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

For parallel sharded Jobs, you generally want a RWX volume (file/NFS). If you only have RWO storage,
set `parallelism: 1` in the Job and shards will run sequentially.

```bash
kubectl apply -f deploy/k8s/results-pvc.yaml
```

### 3) Run a sharded benchmark Job

Edit `deploy/k8s/mcode-bench-indexed-job.yaml`:

- set `image: ...`
- set `BENCHMARK` (`humaneval` or `mbpp`)
- set `MODEL` and `OLLAMA_HOST` (or change `BACKEND`)
- keep `completions == parallelism == SHARD_COUNT`

Then apply:

```bash
kubectl apply -f deploy/k8s/mcode-bench-indexed-job.yaml
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
