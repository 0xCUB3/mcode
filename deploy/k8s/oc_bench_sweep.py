#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class SweepConfig:
    namespace: str
    image: str
    job_name: str
    configmap_name: str
    benchmark: str
    model: str
    backend: str
    ollama_host: str
    samples: int
    debug_iters: int
    timeout_s: int
    shard_count: int
    parallelism: int
    limit: int | None
    extra_env: dict[str, str]

    @staticmethod
    def make_job_name(
        *,
        benchmark: str,
        samples: int,
        debug_iters: int,
        timeout_s: int,
        limit: int | None,
        ts: str,
    ) -> str:
        limit_part = f"-l{limit}" if limit is not None else ""
        # Example: mcode-mbpp-s3-d1-t120-l200-20260208-071530
        return f"mcode-{benchmark}-s{samples}-d{debug_iters}-t{timeout_s}{limit_part}-{ts}"


def _run(
    cmd: list[str],
    *,
    input_text: str | None = None,
    capture: bool = True,
    check: bool = True,
    timeout_s: int | None = None,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict = {
        "text": True,
        "check": False,
        "input": input_text,
    }
    if capture:
        kwargs["capture_output"] = True
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    proc = subprocess.run(cmd, **kwargs)
    if check and proc.returncode != 0:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        msg = f"Command failed ({proc.returncode}): {' '.join(cmd)}"
        if stdout:
            msg += f"\nstdout:\n{stdout}"
        if stderr:
            msg += f"\nstderr:\n{stderr}"
        raise RuntimeError(msg)
    return proc


def _oc(args: list[str], *, namespace: str | None = None, input_text: str | None = None) -> str:
    cmd = ["oc", *args]
    if namespace and "-n" not in args and "--namespace" not in args:
        cmd = ["oc", "-n", namespace, *args]
    proc = _run(cmd, input_text=input_text, capture=True, check=True)
    return (proc.stdout or "").strip()


def _current_namespace() -> str:
    return _oc(["project", "-q"])


def _yaml_string(v: str) -> str:
    # JSON string escaping is YAML-compatible.
    return json.dumps(str(v))


def _apply(namespace: str, yaml_text: str) -> None:
    _oc(["apply", "-f", "-"], namespace=namespace, input_text=yaml_text)


def _build_image(from_dir: Path) -> None:
    print("==> Building OpenShift image (BuildConfig: mcode) ...", file=sys.stderr)
    _run(
        ["oc", "start-build", "mcode", f"--from-dir={from_dir}", "--follow"],
        capture=False,
        check=True,
        timeout_s=60 * 30,
    )


def _render_configmap(cfg: SweepConfig) -> str:
    data = {
        "BENCHMARK": cfg.benchmark,
        "MODEL": cfg.model,
        "BACKEND": cfg.backend,
        "OLLAMA_HOST": cfg.ollama_host,
        "SAMPLES": str(cfg.samples),
        "DEBUG_ITERS": str(cfg.debug_iters),
        "TIMEOUT_S": str(cfg.timeout_s),
        "SHARD_COUNT": str(cfg.shard_count),
        **cfg.extra_env,
    }
    if cfg.limit is not None:
        data["LIMIT"] = str(cfg.limit)

    lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        f"  name: {cfg.configmap_name}",
        "data:",
    ]
    for k in sorted(data.keys()):
        lines.append(f"  {k}: {_yaml_string(data[k])}")
    return "\n".join(lines) + "\n"


def _render_job(cfg: SweepConfig) -> str:
    # Keep results in an EmptyDir and copy them out via `oc cp` while a "hold" container is running.
    # This avoids RWX storage requirements on OpenShift.
    db_path = f"/results/{cfg.benchmark}-shard-${{JOB_COMPLETION_INDEX}}.db"
    bash = r"""set -euo pipefail
limit_args=""
if [ -n "${LIMIT:-}" ]; then
  limit_args="--limit ${LIMIT}"
fi

status=0
mcode bench "${BENCHMARK}" \
  --model "${MODEL}" \
  --backend "${BACKEND}" \
  --samples "${SAMPLES}" \
  --debug-iters "${DEBUG_ITERS}" \
  --timeout "${TIMEOUT_S}" \
  --sandbox process \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${JOB_COMPLETION_INDEX}" \
  --db "__DB_PATH__" \
  ${limit_args} || status=$?

echo "${status}" > /results/_EXIT_CODE
touch /results/_READY
exit "${status}"
"""
    bash = bash.replace("__DB_PATH__", db_path)
    hold = r"""set -euo pipefail
while [ ! -f /results/_COPIED ]; do
  sleep 1
done
"""

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {cfg.job_name}
spec:
  completionMode: Indexed
  completions: {cfg.shard_count}
  parallelism: {cfg.parallelism}
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: mcode
          image: {cfg.image}
          imagePullPolicy: Always
          envFrom:
            - configMapRef:
                name: {cfg.configmap_name}
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: MCODE_CACHE_DIR
              value: /cache
          command: ["bash", "-lc"]
          args:
            - |
{_indent(bash, 14)}
          volumeMounts:
            - name: results
              mountPath: /results
            - name: cache
              mountPath: /cache
          resources:
            requests:
              cpu: "500m"
              memory: 2Gi
            limits:
              cpu: "2"
              memory: 8Gi
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            runAsNonRoot: true
            seccompProfile:
              type: RuntimeDefault
        - name: hold
          image: {cfg.image}
          imagePullPolicy: Always
          command: ["bash", "-lc"]
          args:
            - |
{_indent(hold, 14)}
          volumeMounts:
            - name: results
              mountPath: /results
            - name: cache
              mountPath: /cache
          resources:
            requests:
              cpu: "50m"
              memory: 64Mi
            limits:
              cpu: "200m"
              memory: 256Mi
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            runAsNonRoot: true
            seccompProfile:
              type: RuntimeDefault
      volumes:
        - name: results
          emptyDir: {{}}
        - name: cache
          emptyDir: {{}}
"""


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else prefix for line in text.splitlines())


def _pods_for_job(namespace: str, job_name: str) -> list[dict]:
    raw = _oc(["get", "pods", "-l", f"job-name={job_name}", "-o", "json"], namespace=namespace)
    parsed = json.loads(raw)
    return list(parsed.get("items", []))


def _job_failed(namespace: str, job_name: str) -> tuple[bool, str]:
    raw = _oc(["get", "job", job_name, "-o", "json"], namespace=namespace)
    job = json.loads(raw)
    status = job.get("status") or {}
    conditions = status.get("conditions") or []
    for cond in conditions:
        if cond.get("type") == "Failed" and str(cond.get("status", "")).lower() == "true":
            reason = cond.get("reason") or "Failed"
            message = cond.get("message") or ""
            return True, f"{reason}: {message}".strip()
    return False, ""


def _container_state(pod: dict, container_name: str) -> dict | None:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    for st in statuses:
        if st.get("name") == container_name:
            return st.get("state") or {}
    return None


def _container_terminated(pod: dict, container_name: str) -> dict | None:
    state = _container_state(pod, container_name)
    if not state:
        return None
    terminated = state.get("terminated")
    if not isinstance(terminated, dict):
        return None
    return terminated


def _container_running(pod: dict, container_name: str) -> bool:
    state = _container_state(pod, container_name)
    if not state:
        return False
    return "running" in state


def _pod_index(pod: dict) -> int | None:
    ann = (pod.get("metadata") or {}).get("annotations") or {}
    idx = ann.get("batch.kubernetes.io/job-completion-index")
    if idx is None:
        return None
    try:
        return int(idx)
    except Exception:
        return None


def _exec_hold(namespace: str, pod_name: str, cmd: str) -> subprocess.CompletedProcess[str]:
    return _run(
        ["oc", "-n", namespace, "exec", "-c", "hold", pod_name, "--", "bash", "-lc", cmd],
        capture=True,
        check=False,
    )


def _copy_from_pod(namespace: str, pod_name: str, src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["oc", "-n", namespace, "cp", "-c", "hold", f"{pod_name}:{src}", str(dst)],
        capture=True,
        check=True,
        timeout_s=60 * 5,
    )


def _logs(namespace: str, pod_name: str, container: str) -> str:
    proc = _run(
        ["oc", "-n", namespace, "logs", pod_name, "-c", container],
        capture=True,
        check=False,
        timeout_s=60,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _fetch_results(cfg: SweepConfig, *, out_dir: Path, save_all_logs: bool) -> Path:
    job_dir = out_dir / cfg.job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    todo = set(range(cfg.shard_count))
    last_status = time.time()
    print(f"==> Fetching results for {cfg.job_name} ...", file=sys.stderr)

    while todo:
        failed, msg = _job_failed(cfg.namespace, cfg.job_name)
        if failed:
            raise RuntimeError(f"Job {cfg.job_name} failed: {msg}")

        pods = _pods_for_job(cfg.namespace, cfg.job_name)
        idx_to_pod: dict[int, dict] = {}
        for pod in pods:
            idx = _pod_index(pod)
            if idx is None or idx not in todo:
                continue
            idx_to_pod[idx] = pod

        for idx, pod in sorted(idx_to_pod.items()):
            pod_name = str(((pod.get("metadata") or {}).get("name")) or "")
            if not pod_name:
                continue
            # Ready gate: mcode container writes _READY and _EXIT_CODE.
            ready_cmd = (
                "if [ -f /results/_READY ]; then "
                "cat /results/_EXIT_CODE 2>/dev/null || echo -1; "
                "else exit 1; fi"
            )
            check = _exec_hold(
                cfg.namespace,
                pod_name,
                ready_cmd,
            )
            if check.returncode != 0:
                # If the mcode container died before it could write _READY (e.g., OOMKilled),
                # the hold container will keep the pod Running forever and we'll hang.
                terminated = _container_terminated(pod, "mcode")
                if terminated is not None:
                    reason = str(terminated.get("reason") or "").strip() or "Terminated"
                    exit_code = terminated.get("exitCode")
                    db_name = f"{cfg.benchmark}-shard-{idx}.db"
                    db_dst = job_dir / db_name
                    hold_running = _container_running(pod, "hold")

                    if save_all_logs or (isinstance(exit_code, int) and exit_code != 0):
                        (job_dir / f"shard-{idx}.mcode.log").write_text(
                            _logs(cfg.namespace, pod_name, "mcode"),
                            encoding="utf-8",
                            errors="replace",
                        )

                    # If we already have the DB locally, just allow the pod to terminate
                    # so the Job can make progress.
                    if isinstance(exit_code, int) and exit_code == 0 and db_dst.exists():
                        if hold_running:
                            _exec_hold(cfg.namespace, pod_name, "touch /results/_COPIED")
                        todo.remove(idx)
                        print(
                            f"  - shard {idx}: ok (already copied {db_name})",
                            file=sys.stderr,
                        )
                        continue

                    if isinstance(exit_code, int) and exit_code == 0:
                        if not hold_running:
                            raise RuntimeError(
                                f"Shard {idx} completed but DB not present locally and pod is not accessible."
                            )

                        last_err: Exception | None = None
                        for attempt in range(1, 6):
                            try:
                                _copy_from_pod(
                                    cfg.namespace,
                                    pod_name,
                                    f"/results/{db_name}",
                                    db_dst,
                                )
                                last_err = None
                                break
                            except Exception as e:  # pragma: no cover
                                last_err = e
                                time.sleep(2)
                        if last_err is not None:
                            raise RuntimeError(
                                f"Failed to copy DB for shard {idx} from {pod_name} "
                                f"after retries: {last_err}"
                            ) from last_err

                        _exec_hold(cfg.namespace, pod_name, "touch /results/_COPIED")
                        todo.remove(idx)
                        print(f"  - shard {idx}: ok (copied {db_name})", file=sys.stderr)
                        continue

                    # Non-zero exit: release the pod so the Job can retry.
                    if hold_running:
                        try:
                            _copy_from_pod(
                                cfg.namespace,
                                pod_name,
                                f"/results/{db_name}",
                                db_dst,
                            )
                        except Exception:
                            pass
                        _exec_hold(cfg.namespace, pod_name, "touch /results/_COPIED")
                    print(
                        f"  - shard {idx}: {reason} (exit={exit_code}); waiting for retry",
                        file=sys.stderr,
                    )
                continue

            exit_code_s = (check.stdout or "").strip()
            try:
                exit_code = int(exit_code_s)
            except Exception:
                exit_code = -1

            db_name = f"{cfg.benchmark}-shard-{idx}.db"
            db_src = f"/results/{db_name}"
            db_dst = job_dir / db_name

            if exit_code == 0:
                if db_dst.exists():
                    _exec_hold(cfg.namespace, pod_name, "touch /results/_COPIED")
                    todo.remove(idx)
                    print(f"  - shard {idx}: ok (already copied {db_name})", file=sys.stderr)
                    continue

                # Must copy successfully before allowing the hold container to exit, otherwise the
                # Job will mark this index complete and we may lose the only copy of the DB.
                last_err: Exception | None = None
                for attempt in range(1, 6):
                    try:
                        _copy_from_pod(cfg.namespace, pod_name, db_src, db_dst)
                        last_err = None
                        break
                    except Exception as e:  # pragma: no cover
                        last_err = e
                        time.sleep(2)
                if last_err is not None:
                    raise RuntimeError(
                        f"Failed to copy DB for shard {idx} from {pod_name} "
                        f"after retries: {last_err}"
                    ) from last_err

                if save_all_logs:
                    (job_dir / f"shard-{idx}.mcode.log").write_text(
                        _logs(cfg.namespace, pod_name, "mcode"),
                        encoding="utf-8",
                        errors="replace",
                    )

                _exec_hold(cfg.namespace, pod_name, "touch /results/_COPIED")
                todo.remove(idx)
                print(f"  - shard {idx}: ok (copied {db_name})", file=sys.stderr)
                continue

            # Failure: copy logs for debugging, then allow the pod to terminate so the Job
            # can retry.
            (job_dir / f"shard-{idx}.mcode.log").write_text(
                _logs(cfg.namespace, pod_name, "mcode"),
                encoding="utf-8",
                errors="replace",
            )
            # Try to copy whatever DB exists (it may be partial).
            try:
                _copy_from_pod(cfg.namespace, pod_name, db_src, db_dst)
            except Exception:
                pass
            _exec_hold(cfg.namespace, pod_name, "touch /results/_COPIED")
            print(f"  - shard {idx}: failed (exit={exit_code}); waiting for retry", file=sys.stderr)

        if time.time() - last_status > 30:
            print(f"  ... remaining shards: {len(todo)}", file=sys.stderr)
            last_status = time.time()

        time.sleep(2)

    # Ensure job completed.
    _run(
        [
            "oc",
            "-n",
            cfg.namespace,
            "wait",
            "--for=condition=complete",
            f"job/{cfg.job_name}",
            "--timeout=10m",
        ],
        capture=True,
        check=True,
    )

    return job_dir


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated list.")
    return out


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid env key: {item!r}")
        out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run sharded mcode benchmark Jobs on OpenShift and copy shard DBs locally, "
            "without requiring RWX storage."
        )
    )
    p.add_argument("--namespace", default="", help="OpenShift project (default: current)")
    p.add_argument(
        "--benchmarks",
        default="humaneval,mbpp",
        help="Comma-separated benchmarks (humaneval,mbpp)",
    )
    p.add_argument("--model", default="granite4:latest", help="Mellea model id")
    p.add_argument("--backend", default="ollama", help="Mellea backend name")
    p.add_argument("--ollama-host", default="http://ollama:11434", help="Ollama host URL")
    p.add_argument("--samples", default="1,2,3", help="Comma-separated samples list")
    p.add_argument("--debug-iters", default="0,1", help="Comma-separated debug-iters list")
    p.add_argument("--timeout", default="60,120", help="Comma-separated timeout seconds list")
    p.add_argument("--shard-count", type=int, default=20)
    p.add_argument("--parallelism", type=int, default=2)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run first N tasks only (pilot sweep)",
    )
    p.add_argument(
        "--out-dir",
        default="results/oc-sweep",
        help="Local output directory",
    )
    p.add_argument(
        "--build",
        dest="build",
        action="store_true",
        help="Trigger a new OpenShift binary build for the mcode image (default)",
    )
    p.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="Skip image build (use existing cluster image)",
    )
    p.add_argument(
        "--image",
        default="",
        help="Container image reference (default: internal registry imagestreamtag)",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env to pass via ConfigMap (repeatable KEY=VALUE)",
    )
    p.add_argument(
        "--save-all-logs",
        action="store_true",
        help="Save logs for successful shards too (can be large)",
    )
    p.add_argument(
        "--keep-cluster-resources",
        action="store_true",
        help="Do not delete Jobs/ConfigMaps after copying results",
    )
    args = p.parse_args()
    if not hasattr(args, "build"):
        args.build = True
    if args.build is None:
        args.build = True

    namespace = args.namespace.strip() or _current_namespace()
    from_dir = Path.cwd()

    if args.build:
        _build_image(from_dir)

    image = args.image.strip()
    if not image:
        # Use the internal OpenShift registry image for the current namespace.
        image = f"image-registry.openshift-image-registry.svc:5000/{namespace}/mcode:latest"

    samples_list = _parse_int_list(args.samples)
    debug_list = _parse_int_list(args.debug_iters)
    timeout_list = _parse_int_list(args.timeout)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    extra_env = _parse_kv_list(args.env)

    out_dir = Path(args.out_dir) / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Writing results to: {out_dir}", file=sys.stderr)

    for benchmark, samples, debug_iters, timeout_s in product(
        benchmarks, samples_list, debug_list, timeout_list
    ):
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = SweepConfig.make_job_name(
            benchmark=benchmark,
            samples=int(samples),
            debug_iters=int(debug_iters),
            timeout_s=int(timeout_s),
            limit=args.limit,
            ts=ts,
        )
        cfg = SweepConfig(
            namespace=namespace,
            image=image,
            job_name=job_name,
            configmap_name=f"{job_name}-config",
            benchmark=benchmark,
            model=args.model,
            backend=args.backend,
            ollama_host=args.ollama_host,
            samples=int(samples),
            debug_iters=int(debug_iters),
            timeout_s=int(timeout_s),
            shard_count=int(args.shard_count),
            parallelism=int(args.parallelism),
            limit=args.limit,
            extra_env=extra_env,
        )

        print(
            f"\n==> Launching {cfg.job_name} (benchmark={benchmark} samples={samples} "
            f"debug={debug_iters} timeout={timeout_s}s limit={args.limit})",
            file=sys.stderr,
        )
        _apply(namespace, _render_configmap(cfg))
        _apply(namespace, _render_job(cfg))
        _fetch_results(cfg, out_dir=out_dir, save_all_logs=bool(args.save_all_logs))

        if not args.keep_cluster_resources:
            _oc(["delete", "job", cfg.job_name], namespace=namespace)
            _oc(["delete", "configmap", cfg.configmap_name], namespace=namespace)

    print("\n==> Done.", file=sys.stderr)
    print(f"Results: {out_dir}", file=sys.stderr)
    print(
        "\nNext:\n"
        f"  .venv/bin/mcode results --db-dir {out_dir} --compare-samples --time\n"
        f"  .venv/bin/mcode report --db-dir {out_dir} --out {out_dir}/report.html\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
