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

    # Build set of all local image tags for fast cache lookup.
    # images.get() fails on podman with name mismatches, so list all.
    local_tags: set[str] = set()
    for img in client.images.list():
        for tag in img.tags or []:
            local_tags.add(tag)
            # Also add without docker.io/ prefix and with it
            if tag.startswith("docker.io/"):
                local_tags.add(tag[len("docker.io/") :])
            else:
                local_tags.add(f"docker.io/{tag}")

    for i, task in enumerate(tasks):
        spec = make_test_spec(task, namespace=namespace)
        name = spec.instance_image_key

        # Check cache using both short and FQ names
        if name in local_tags:
            cached += 1
            continue

        fq_name = name if "/" in name and "." in name.split("/")[0] else f"docker.io/{name}"

        try:
            print(f"  [{i + 1}/{len(tasks)}] pulling {fq_name}...", flush=True)
            # Use low-level API to avoid post-pull image lookup that
            # fails on podman's Docker-compat API due to name mismatch.
            for line in client.api.pull(fq_name, stream=True, decode=True):
                if "error" in line:
                    raise RuntimeError(line["error"])
            pulled += 1
        except Exception as e:
            print(f"  [{i + 1}/{len(tasks)}] FAILED {fq_name}: {e}", flush=True)
            failed += 1

    print(f"Pre-pull done: {pulled} pulled, {failed} failed, {cached} cached")


if __name__ == "__main__":
    namespace = sys.argv[1] if len(sys.argv) > 1 else "swebench"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    main(namespace, limit)
