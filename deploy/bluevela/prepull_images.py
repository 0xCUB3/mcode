"""Pre-pull all SWE-bench Lite Docker images sequentially."""
from __future__ import annotations

import sys


def main(namespace: str, limit: int) -> None:
    import docker
    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec

    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    tasks = list(ds)[:limit]
    print(f"Pre-pulling images for {len(tasks)} tasks (namespace={namespace})...")

    client = docker.from_env()
    pulled = 0
    failed = 0
    cached = 0

    for i, task in enumerate(tasks):
        spec = make_test_spec(task, namespace=namespace)
        name = spec.instance_image_key

        try:
            client.images.get(name)
            cached += 1
            continue
        except docker.errors.ImageNotFound:
            pass

        try:
            print(f"  [{i+1}/{len(tasks)}] pulling {name}...", flush=True)
            client.images.pull(name)
            pulled += 1
        except Exception as e:
            print(f"  [{i+1}/{len(tasks)}] FAILED {name}: {e}", flush=True)
            failed += 1

    print(f"Pre-pull done: {pulled} pulled, {failed} failed, {cached} cached")


if __name__ == "__main__":
    namespace = sys.argv[1] if len(sys.argv) > 1 else "swebench"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(namespace, limit)
