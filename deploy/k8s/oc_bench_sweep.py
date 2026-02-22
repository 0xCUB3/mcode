#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
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
    loop_budget: int
    timeout_s: int
    strategy: str
    s2_model: str
    s2_backend: str
    s2_mode: str
    shard_count: int
    parallelism: int
    limit: int | None
    extra_env: dict[str, str]
    mcode_cpu_request: str
    mcode_memory_request: str
    mcode_cpu_limit: str
    mcode_memory_limit: str
    hold_cpu_request: str
    hold_memory_request: str
    hold_cpu_limit: str
    hold_memory_limit: str

    @staticmethod
    def make_job_name(
        *,
        benchmark: str,
        loop_budget: int,
        timeout_s: int,
        strategy: str = "repair",
        limit: int | None,
        ts: str,
    ) -> str:
        limit_part = f"-l{limit}" if limit is not None else ""
        strategy_part = f"-{strategy}" if strategy != "repair" else ""
        safe_bench = re.sub(r"[^a-z0-9-]", "", benchmark.lower().replace("+", "plus"))
        # Example: mcode-mbpp-b3-t120-sofai-l200-20260208-071530
        return f"mcode-{safe_bench}-b{loop_budget}-t{timeout_s}{strategy_part}{limit_part}-{ts}"


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


def _oc(
    args: list[str],
    *,
    namespace: str | None = None,
    input_text: str | None = None,
    timeout_s: int | None = 120,
    retries: int = 12,
    retry_delay_s: float = 5.0,
) -> str:
    cmd = ["oc", *args]
    if namespace and "-n" not in args and "--namespace" not in args:
        cmd = ["oc", "-n", namespace, *args]
    last_err: RuntimeError | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            proc = _run(cmd, input_text=input_text, capture=True, check=True, timeout_s=timeout_s)
            return (proc.stdout or "").strip()
        except RuntimeError as err:
            last_err = err
            if not _is_transient_oc_error(str(err)) or attempt >= retries:
                raise
            print(
                f"  ... transient oc error ({attempt}/{retries}), retrying in {retry_delay_s:.0f}s",
                file=sys.stderr,
            )
            time.sleep(retry_delay_s)
    assert last_err is not None
    raise last_err


def _is_transient_oc_error(text: str) -> bool:
    haystack = text.lower()
    needles = (
        "unable to connect to the server",
        "no such host",
        "dial tcp",
        "i/o timeout",
        "context deadline exceeded",
        "connection refused",
        "tls handshake timeout",
        "server closed idle connection",
    )
    return any(needle in haystack for needle in needles)


def _is_notfound_error(text: str) -> bool:
    haystack = text.lower()
    return ("not found" in haystack) or ("notfound" in haystack)


def _normalize_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    if not cleaned:
        raise ValueError("Invalid run id; expected letters/numbers or separators.")
    return cleaned


def _latest_run_id(out_root: Path) -> str | None:
    if not out_root.exists():
        return None
    dirs = [path for path in out_root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs[0].name


def _job_token(run_id: str) -> str:
    # Keep job names compact and k8s-safe.
    token = _normalize_run_id(run_id)
    return token[-40:]


def _current_namespace() -> str:
    try:
        return _oc(["project", "-q"])
    except RuntimeError:
        # In-cluster: read namespace from mounted ServiceAccount token.
        ns_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        if ns_path.exists():
            return ns_path.read_text().strip()
        raise


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
        "LOOP_BUDGET": str(cfg.loop_budget),
        "TIMEOUT_S": str(cfg.timeout_s),
        "STRATEGY": cfg.strategy,
        "SHARD_COUNT": str(cfg.shard_count),
        **cfg.extra_env,
    }
    if cfg.limit is not None:
        data["LIMIT"] = str(cfg.limit)
    if cfg.s2_model:
        data["S2_MODEL"] = cfg.s2_model
        data["S2_BACKEND"] = cfg.s2_backend
        data["S2_MODE"] = cfg.s2_mode

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
strategy_args="--strategy ${STRATEGY}"
if [ -n "${S2_MODEL:-}" ]; then
  strategy_args="${strategy_args} --s2-model ${S2_MODEL}"
  strategy_args="${strategy_args} --s2-backend ${S2_BACKEND}"
  strategy_args="${strategy_args} --s2-mode ${S2_MODE}"
fi
lcb_cutoff_args=""
if [ -n "${LCB_CUTOFF:-}" ]; then
  lcb_cutoff_args="--lcb-cutoff ${LCB_CUTOFF}"
fi

status=0
mcode bench "${BENCHMARK}" \
  --model "${MODEL}" \
  --backend "${BACKEND}" \
  --loop-budget "${LOOP_BUDGET}" \
  --timeout "${TIMEOUT_S}" \
  --sandbox process \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${JOB_COMPLETION_INDEX}" \
  --db "__DB_PATH__" \
  ${strategy_args} \
  ${limit_args} \
  ${lcb_cutoff_args} || status=$?

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
  backoffLimit: 10
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
            - name: HF_HOME
              value: /cache/huggingface
            - name: EVALPLUS_CACHE_DIR
              value: /cache/evalplus
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
              cpu: {_yaml_string(cfg.mcode_cpu_request)}
              memory: {_yaml_string(cfg.mcode_memory_request)}
            limits:
              cpu: {_yaml_string(cfg.mcode_cpu_limit)}
              memory: {_yaml_string(cfg.mcode_memory_limit)}
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
              cpu: {_yaml_string(cfg.hold_cpu_request)}
              memory: {_yaml_string(cfg.hold_memory_request)}
            limits:
              cpu: {_yaml_string(cfg.hold_cpu_limit)}
              memory: {_yaml_string(cfg.hold_memory_limit)}
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
          emptyDir:
            sizeLimit: 10Gi
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


def _job_exists(namespace: str, job_name: str) -> bool:
    try:
        _oc(["get", "job", job_name, "-o", "name"], namespace=namespace, retries=3, timeout_s=30)
        return True
    except RuntimeError as err:
        msg = str(err).lower()
        if "notfound" in msg or "not found" in msg:
            return False
        raise


def _shard_db_path(job_dir: Path, benchmark: str, idx: int) -> Path:
    return job_dir / f"{benchmark}-shard-{idx}.db"


def _shard_ok_path(job_dir: Path, benchmark: str, idx: int) -> Path:
    return job_dir / f"{benchmark}-shard-{idx}.ok"


def _mark_shard_ok(job_dir: Path, benchmark: str, idx: int) -> None:
    _shard_ok_path(job_dir, benchmark, idx).write_text("ok\n", encoding="utf-8")


def _clear_shard_ok(job_dir: Path, benchmark: str, idx: int) -> None:
    _shard_ok_path(job_dir, benchmark, idx).unlink(missing_ok=True)


def _has_all_shards(job_dir: Path, benchmark: str, shard_count: int) -> bool:
    if not job_dir.exists():
        return False
    db_paths = [_shard_db_path(job_dir, benchmark, idx) for idx in range(shard_count)]
    if not all(path.exists() for path in db_paths):
        return False

    # New runs write per-shard ".ok" markers only after successful copy from a
    # successful shard execution. If any markers exist, require all of them.
    ok_paths = [_shard_ok_path(job_dir, benchmark, idx) for idx in range(shard_count)]
    ok_count = sum(1 for path in ok_paths if path.exists())
    if ok_count == 0:
        # Backward-compatible fallback for older runs without markers.
        return True
    return ok_count == shard_count


def _patch_job_parallelism(namespace: str, job_name: str, parallelism: int) -> None:
    payload = json.dumps({"spec": {"parallelism": int(parallelism)}})
    _oc(
        ["patch", "job", job_name, "--type=merge", "-p", payload],
        namespace=namespace,
    )


def _job_events(namespace: str, job_name: str) -> list[dict]:
    raw = _oc(
        [
            "get",
            "events",
            "--field-selector",
            f"involvedObject.kind=Job,involvedObject.name={job_name}",
            "-o",
            "json",
        ],
        namespace=namespace,
    )
    items = list(json.loads(raw).get("items") or [])
    items.sort(
        key=lambda e: (
            str(e.get("eventTime") or ""),
            str(e.get("lastTimestamp") or ""),
            str((e.get("metadata") or {}).get("creationTimestamp") or ""),
        )
    )
    return items


def _event_text(event: dict) -> str:
    reason = str(event.get("reason") or "").strip()
    message = str(event.get("message") or "").strip()
    if reason and message:
        return f"{reason}: {message}"
    if reason:
        return reason
    return message or "<no message>"


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


def _container_waiting_message(pod: dict, container_name: str) -> str:
    state = _container_state(pod, container_name)
    if not state:
        return ""
    waiting = state.get("waiting")
    if not isinstance(waiting, dict):
        return ""
    reason = str(waiting.get("reason") or "").strip()
    message = str(waiting.get("message") or "").strip()
    if reason and message:
        return f"{reason}: {message}"
    if reason:
        return reason
    return message


def _pod_unschedulable_message(pod: dict) -> str:
    conditions = (pod.get("status") or {}).get("conditions") or []
    for condition in conditions:
        if condition.get("type") != "PodScheduled":
            continue
        if str(condition.get("status") or "").lower() != "false":
            continue
        reason = str(condition.get("reason") or "").strip()
        message = str(condition.get("message") or "").strip()
        if reason and message:
            return f"{reason}: {message}"
        if reason:
            return reason
        return message
    return ""


def _waiting_reason_counts(pods: list[dict], container_name: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for pod in pods:
        reason = _container_waiting_message(pod, container_name)
        if not reason:
            reason = _pod_unschedulable_message(pod)
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _short_reason(text: str, limit: int = 96) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _pod_index(pod: dict) -> int | None:
    ann = (pod.get("metadata") or {}).get("annotations") or {}
    idx = ann.get("batch.kubernetes.io/job-completion-index")
    if idx is None:
        return None
    try:
        return int(idx)
    except Exception:
        return None


def _exec_hold(
    namespace: str,
    pod_name: str,
    cmd: str,
    *,
    timeout_s: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return _run(
            [
                "oc",
                "-n",
                namespace,
                f"--request-timeout={timeout_s}s",
                "exec",
                "-c",
                "hold",
                pod_name,
                "--",
                "bash",
                "-c",
                cmd,
            ],
            capture=True,
            check=False,
            timeout_s=timeout_s + 5,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["oc", "exec", pod_name],
            returncode=124,
            stdout="",
            stderr="exec timeout",
        )


def _mark_copied(namespace: str, pod_name: str) -> bool:
    for _ in range(5):
        proc = _exec_hold(namespace, pod_name, "touch /results/_COPIED", timeout_s=20)
        if proc.returncode == 0:
            return True
        time.sleep(1)
    return False


def _copy_from_pod(namespace: str, pod_name: str, src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["oc", "-n", namespace, "cp", "-c", "hold", f"{pod_name}:{src}", str(dst)],
        capture=True,
        check=True,
        timeout_s=60 * 5,
    )


def _copy_with_retries(namespace: str, pod_name: str, src: str, dst: Path) -> Exception | None:
    last_err: Exception | None = None
    for _ in range(5):
        try:
            _copy_from_pod(namespace, pod_name, src, dst)
            return None
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(2)
    return last_err


def _logs(namespace: str, pod_name: str, container: str) -> str:
    proc = _run(
        ["oc", "-n", namespace, "logs", pod_name, "-c", container],
        capture=True,
        check=False,
        timeout_s=60,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _save_mcode_log(job_dir: Path, namespace: str, pod_name: str, idx: int) -> None:
    (job_dir / f"shard-{idx}.mcode.log").write_text(
        _logs(namespace, pod_name, "mcode"),
        encoding="utf-8",
        errors="replace",
    )


def _fetch_results(
    cfg: SweepConfig,
    *,
    out_dir: Path,
    save_all_logs: bool,
    stalled_seconds: int,
    auto_reduce_parallelism: bool,
) -> Path:
    job_dir = out_dir / cfg.job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    todo = set(range(cfg.shard_count))
    completed_unavailable_warned: set[int] = set()
    last_status = time.time()
    last_progress = time.time()
    current_parallelism = int(cfg.parallelism)
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
                timeout_s=20,
            )
            if check.returncode != 0:
                # If the mcode container died before it could write _READY (e.g., OOMKilled),
                # the hold container will keep the pod Running forever and we'll hang.
                terminated = _container_terminated(pod, "mcode")
                if terminated is not None:
                    reason = str(terminated.get("reason") or "").strip() or "Terminated"
                    exit_code = terminated.get("exitCode")
                    db_name = f"{cfg.benchmark}-shard-{idx}.db"
                    db_dst = _shard_db_path(job_dir, cfg.benchmark, idx)
                    hold_running = _container_running(pod, "hold")

                    if save_all_logs or (isinstance(exit_code, int) and exit_code != 0):
                        _save_mcode_log(job_dir, cfg.namespace, pod_name, idx)

                    # If we already have the DB locally, just allow the pod to terminate
                    # so the Job can make progress.
                    if isinstance(exit_code, int) and exit_code == 0 and db_dst.exists():
                        if hold_running and not _mark_copied(cfg.namespace, pod_name):
                            continue
                        _mark_shard_ok(job_dir, cfg.benchmark, idx)
                        todo.remove(idx)
                        last_progress = time.time()
                        print(
                            f"  - shard {idx}: ok (already copied {db_name})",
                            file=sys.stderr,
                        )
                        continue

                    if isinstance(exit_code, int) and exit_code == 0:
                        if not hold_running:
                            last_err = _copy_with_retries(
                                cfg.namespace,
                                pod_name,
                                f"/results/{db_name}",
                                db_dst,
                            )
                            if last_err is None and db_dst.exists():
                                _mark_shard_ok(job_dir, cfg.benchmark, idx)
                                todo.remove(idx)
                                last_progress = time.time()
                                print(
                                    f"  - shard {idx}: ok (copied {db_name})",
                                    file=sys.stderr,
                                )
                                continue
                            if idx not in completed_unavailable_warned:
                                print(
                                    f"  - shard {idx}: completed but pod not ready for copy yet; "
                                    "waiting",
                                    file=sys.stderr,
                                )
                                completed_unavailable_warned.add(idx)
                            continue

                        last_err = _copy_with_retries(
                            cfg.namespace,
                            pod_name,
                            f"/results/{db_name}",
                            db_dst,
                        )
                        if last_err is not None:
                            if _is_notfound_error(str(last_err)):
                                if idx not in completed_unavailable_warned:
                                    print(
                                        f"  - shard {idx}: pod disappeared while copying; "
                                        "waiting for replacement pod",
                                        file=sys.stderr,
                                    )
                                    completed_unavailable_warned.add(idx)
                                continue
                            _clear_shard_ok(job_dir, cfg.benchmark, idx)
                            raise RuntimeError(
                                f"Failed to copy DB for shard {idx} from {pod_name} "
                                f"after retries: {last_err}"
                            ) from last_err

                        if not _mark_copied(cfg.namespace, pod_name):
                            continue
                        _mark_shard_ok(job_dir, cfg.benchmark, idx)
                        todo.remove(idx)
                        last_progress = time.time()
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
                        except Exception as e:
                            print(f"  - shard {idx}: partial copy failed: {e}", file=sys.stderr)
                        _mark_copied(cfg.namespace, pod_name)
                    _clear_shard_ok(job_dir, cfg.benchmark, idx)
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
            db_dst = _shard_db_path(job_dir, cfg.benchmark, idx)

            if exit_code == 0:
                if db_dst.exists():
                    if not _mark_copied(cfg.namespace, pod_name):
                        continue
                    _mark_shard_ok(job_dir, cfg.benchmark, idx)
                    todo.remove(idx)
                    last_progress = time.time()
                    print(f"  - shard {idx}: ok (already copied {db_name})", file=sys.stderr)
                    continue

                # Must copy successfully before allowing the hold container to exit, otherwise the
                # Job will mark this index complete and we may lose the only copy of the DB.
                last_err = _copy_with_retries(cfg.namespace, pod_name, db_src, db_dst)
                if last_err is not None:
                    if _is_notfound_error(str(last_err)):
                        if idx not in completed_unavailable_warned:
                            print(
                                f"  - shard {idx}: pod disappeared while copying; "
                                "waiting for replacement pod",
                                file=sys.stderr,
                            )
                            completed_unavailable_warned.add(idx)
                        continue
                    _clear_shard_ok(job_dir, cfg.benchmark, idx)
                    raise RuntimeError(
                        f"Failed to copy DB for shard {idx} from {pod_name} "
                        f"after retries: {last_err}"
                    ) from last_err

                if save_all_logs:
                    _save_mcode_log(job_dir, cfg.namespace, pod_name, idx)

                if not _mark_copied(cfg.namespace, pod_name):
                    continue
                _mark_shard_ok(job_dir, cfg.benchmark, idx)
                todo.remove(idx)
                last_progress = time.time()
                print(f"  - shard {idx}: ok (copied {db_name})", file=sys.stderr)
                continue

            # Failure: copy logs for debugging, then allow the pod to terminate so the Job
            # can retry.
            _save_mcode_log(job_dir, cfg.namespace, pod_name, idx)
            # Try to copy whatever DB exists (it may be partial).
            try:
                _copy_from_pod(cfg.namespace, pod_name, db_src, db_dst)
            except Exception:
                pass
            _mark_copied(cfg.namespace, pod_name)
            _clear_shard_ok(job_dir, cfg.benchmark, idx)
            print(f"  - shard {idx}: failed (exit={exit_code}); waiting for retry", file=sys.stderr)

        if time.time() - last_status > 30:
            pod_list = list(idx_to_pod.values())
            running_mcode = sum(1 for p in pod_list if _container_running(p, "mcode"))
            waiting_reasons = _waiting_reason_counts(pod_list, "mcode")
            waiting_mcode = sum(c for _, c in waiting_reasons)
            reason_text = ""
            if waiting_reasons:
                top = "; ".join(
                    f"{count}x {_short_reason(reason)}" for reason, count in waiting_reasons[:2]
                )
                reason_text = f" waiting={top}"
            print(
                f"  ... remaining shards: {len(todo)} "
                f"(active_pods={len(idx_to_pod)} "
                f"mcode_running={running_mcode} mcode_waiting={waiting_mcode})"
                f"{reason_text}",
                file=sys.stderr,
            )
            last_status = time.time()

        now = time.time()
        if stalled_seconds > 0 and (now - last_progress) > stalled_seconds:
            pod_list = list(idx_to_pod.values())
            running_mcode = sum(1 for p in pod_list if _container_running(p, "mcode"))
            if running_mcode > 0:
                print(
                    "  ... no completed shards yet; mcode containers are still running "
                    f"({running_mcode}).",
                    file=sys.stderr,
                )
                last_progress = now
            else:
                events = _job_events(cfg.namespace, cfg.job_name)
                recent_events = events[-3:]
                waiting_reasons = _waiting_reason_counts(pod_list, "mcode")
                active_pods = len(pod_list)
                quota_blocked = any(
                    "exceeded quota" in _event_text(event).lower() for event in recent_events
                )
                unschedulable = any(
                    "unschedulable" in reason.lower() for reason, _ in waiting_reasons
                )
                no_active_pods = active_pods == 0

                auto_reduce_reason = ""
                if quota_blocked:
                    auto_reduce_reason = "quota pressure"
                elif unschedulable:
                    auto_reduce_reason = "pods unschedulable"
                elif no_active_pods:
                    auto_reduce_reason = "no active pods"

                if auto_reduce_reason and auto_reduce_parallelism and current_parallelism > 1:
                    new_parallelism = max(1, current_parallelism // 2)
                    if new_parallelism < current_parallelism:
                        _patch_job_parallelism(cfg.namespace, cfg.job_name, new_parallelism)
                        print(
                            "  ... auto-reduced parallelism due "
                            f"{auto_reduce_reason}: "
                            f"{current_parallelism} -> {new_parallelism}",
                            file=sys.stderr,
                        )
                        current_parallelism = new_parallelism
                        last_progress = now
                        continue

                recent_text = "; ".join(_event_text(event) for event in recent_events) or "none"
                waiting_text = (
                    "; ".join(
                        f"{count}x {_short_reason(reason)}" for reason, count in waiting_reasons[:3]
                    )
                    if waiting_reasons
                    else "none"
                )
                raise RuntimeError(
                    f"Job {cfg.job_name} stalled with {len(todo)} shards remaining "
                    f"for >{stalled_seconds}s. "
                    f"Recent events: {recent_text}. "
                    f"Waiting reasons: {waiting_text}"
                )

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
    p.add_argument("--loop-budget", default="1,3,5", help="Comma-separated loop-budget list")
    p.add_argument("--timeout", default="60,120", help="Comma-separated timeout seconds list")
    p.add_argument("--shard-count", type=int, default=20)
    p.add_argument("--parallelism", type=int, default=2)
    p.add_argument(
        "--stalled-seconds",
        type=int,
        default=600,
        help="Detect and diagnose no-progress stalls after this many seconds (0 disables)",
    )
    p.add_argument(
        "--auto-reduce-parallelism",
        dest="auto_reduce_parallelism",
        action="store_true",
        help="Auto-reduce Job parallelism when quota pressure stalls scheduling (default)",
    )
    p.add_argument(
        "--no-auto-reduce-parallelism",
        dest="auto_reduce_parallelism",
        action="store_false",
        help="Disable automatic parallelism reduction on quota stalls",
    )
    p.add_argument(
        "--mcode-cpu-request",
        default="500m",
        help="CPU request for mcode container (default: 500m)",
    )
    p.add_argument(
        "--mcode-memory-request",
        default="2Gi",
        help="Memory request for mcode container (default: 2Gi)",
    )
    p.add_argument(
        "--mcode-cpu-limit",
        default="2",
        help="CPU limit for mcode container (default: 2)",
    )
    p.add_argument(
        "--mcode-memory-limit",
        default="12Gi",
        help="Memory limit for mcode container (default: 12Gi)",
    )
    p.add_argument(
        "--hold-cpu-request",
        default="50m",
        help="CPU request for hold container (default: 50m)",
    )
    p.add_argument(
        "--hold-memory-request",
        default="64Mi",
        help="Memory request for hold container (default: 64Mi)",
    )
    p.add_argument(
        "--hold-cpu-limit",
        default="200m",
        help="CPU limit for hold container (default: 200m)",
    )
    p.add_argument(
        "--hold-memory-limit",
        default="256Mi",
        help="Memory limit for hold container (default: 256Mi)",
    )
    p.add_argument(
        "--strategy",
        default="repair",
        help="Sampling strategy: repair or sofai (default: repair)",
    )
    p.add_argument(
        "--s2-model",
        default="",
        help="Model ID for SOFAI S2 solver (required when --strategy=sofai)",
    )
    p.add_argument(
        "--s2-backend",
        default="ollama",
        help="Backend for SOFAI S2 solver (default: ollama)",
    )
    p.add_argument(
        "--s2-mode",
        default="best_attempt",
        help="SOFAI S2 solver mode: fresh_start|continue_chat|best_attempt",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run first N tasks only (pilot sweep)",
    )
    p.add_argument(
        "--out-dir",
        default="results/oc-sweep",
        help="Local output directory (run is stored under <out-dir>/<run-id>)",
    )
    p.add_argument(
        "--run-id",
        default="",
        help="Stable run id for resume/reattach (default: current timestamp)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume latest (or --run-id) run: reattach to existing jobs and skip completed configs"
        ),
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

    namespace = args.namespace.strip() or _current_namespace()
    from_dir = Path.cwd()

    out_root = Path(args.out_dir)
    if args.resume:
        selected_run_id = args.run_id.strip()
        if selected_run_id:
            run_id = _normalize_run_id(selected_run_id)
        else:
            latest = _latest_run_id(out_root)
            if latest is None:
                raise SystemExit("No previous run found under --out-dir; cannot --resume.")
            run_id = latest
    else:
        selected_run_id = args.run_id.strip()
        run_id = (
            _normalize_run_id(selected_run_id)
            if selected_run_id
            else _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    out_dir = out_root / run_id
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"Run directory already exists and is non-empty: {out_dir}\n"
            "Use --resume to continue that run."
        )
    if args.resume and not out_dir.exists():
        raise SystemExit(f"Run directory does not exist for --resume: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Writing results to: {out_dir}", file=sys.stderr)
    if args.resume:
        print(
            "==> Resume mode: reattaching existing jobs and skipping completed configs",
            file=sys.stderr,
        )

    if args.build:
        _build_image(from_dir)

    image = args.image.strip()
    if not image:
        # Use the internal OpenShift registry image for the current namespace.
        image = f"image-registry.openshift-image-registry.svc:5000/{namespace}/mcode:latest"

    budget_list = _parse_int_list(args.loop_budget)
    timeout_list = _parse_int_list(args.timeout)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    extra_env = _parse_kv_list(args.env)
    run_token = _job_token(run_id)

    for benchmark, loop_budget, timeout_s in product(benchmarks, budget_list, timeout_list):
        job_name = SweepConfig.make_job_name(
            benchmark=benchmark,
            loop_budget=int(loop_budget),
            timeout_s=int(timeout_s),
            strategy=args.strategy,
            limit=args.limit,
            ts=run_token,
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
            loop_budget=int(loop_budget),
            timeout_s=int(timeout_s),
            strategy=args.strategy,
            s2_model=args.s2_model,
            s2_backend=args.s2_backend,
            s2_mode=args.s2_mode,
            shard_count=int(args.shard_count),
            parallelism=int(args.parallelism),
            limit=args.limit,
            extra_env=extra_env,
            mcode_cpu_request=str(args.mcode_cpu_request),
            mcode_memory_request=str(args.mcode_memory_request),
            mcode_cpu_limit=str(args.mcode_cpu_limit),
            mcode_memory_limit=str(args.mcode_memory_limit),
            hold_cpu_request=str(args.hold_cpu_request),
            hold_memory_request=str(args.hold_memory_request),
            hold_cpu_limit=str(args.hold_cpu_limit),
            hold_memory_limit=str(args.hold_memory_limit),
        )

        job_dir = out_dir / cfg.job_name
        job_exists = _job_exists(namespace, cfg.job_name)
        if job_exists:
            print(
                f"\n==> Reattaching {cfg.job_name} (benchmark={benchmark} budget={loop_budget} "
                f"timeout={timeout_s}s limit={args.limit})",
                file=sys.stderr,
            )
        elif _has_all_shards(job_dir, cfg.benchmark, cfg.shard_count):
            print(
                f"\n==> Skipping {cfg.job_name}: already have {cfg.shard_count} shard DBs",
                file=sys.stderr,
            )
            continue
        else:
            print(
                f"\n==> Launching {cfg.job_name} (benchmark={benchmark} budget={loop_budget} "
                f"timeout={timeout_s}s limit={args.limit})",
                file=sys.stderr,
            )
            _apply(namespace, _render_configmap(cfg))
            _apply(namespace, _render_job(cfg))

        _fetch_results(
            cfg,
            out_dir=out_dir,
            save_all_logs=bool(args.save_all_logs),
            stalled_seconds=int(args.stalled_seconds),
            auto_reduce_parallelism=bool(args.auto_reduce_parallelism),
        )

        if not args.keep_cluster_resources:
            _oc(["delete", "job", cfg.job_name], namespace=namespace)
            _oc(["delete", "configmap", cfg.configmap_name], namespace=namespace)

    print("\n==> Done.", file=sys.stderr)
    print(f"Results: {out_dir}", file=sys.stderr)
    print(
        "\nNext:\n"
        f"  .venv/bin/mcode results --db-dir {out_dir} --time\n"
        f"  .venv/bin/mcode report --db-dir {out_dir} --out {out_dir}/report.html\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
