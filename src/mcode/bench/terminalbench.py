from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mcode.bench.artifacts import (
    SCHEMA_VERSION,
    TaskArtifactManifest,
    TaskArtifactStore,
    digest_json,
    iso_utc_now,
    make_task_digest,
)
from mcode.bench.results import ResultsDB, RunSummary

BENCHMARK_NAME = "terminal-bench"
DEFAULT_DATASET = "terminal-bench/terminal-bench-2"
MCODE_AGENT_IMPORT_PATH = "mcode.bench.terminalbench_agent:MCodeTerminalBenchAgent"
_TIMEOUT_EXCEPTIONS = {
    "AgentTimeoutError",
    "VerifierTimeoutError",
    "EnvironmentStartTimeoutError",
    "EnvironmentBuildTimeoutError",
}
_LOG_LIMIT_CHARS = 80_000


@dataclass(frozen=True)
class TerminalBenchConfig:
    model_id: str
    backend_name: str = "openai"
    agent: str = "mcode"
    dataset: str = DEFAULT_DATASET
    jobs_dir: Path = Path("experiments/results/terminal-bench-jobs")
    job_name: str | None = None
    environment_type: str = "docker"
    n_concurrent: int = 1
    timeout_multiplier: float = 1.0
    agent_timeout_s: int | None = None
    verifier_timeout_s: int | None = None
    harbor_executable: str = "harbor"
    artifact_dir: Path | None = None
    extra_harbor_args: tuple[str, ...] = ()
    agent_kwargs: dict[str, str] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)

    def run_config(self, *, limit: int | None, task_ids: list[str] | None) -> dict[str, Any]:
        data = asdict(self)
        data["jobs_dir"] = str(self.jobs_dir)
        data["artifact_dir"] = str(self.artifact_dir) if self.artifact_dir else None
        data["extra_harbor_args"] = list(self.extra_harbor_args)
        data["limit"] = limit
        data["task_ids"] = task_ids or []
        data["timeout_s"] = int(self.timeout_multiplier * 3600)
        data["loop_budget"] = 0
        return data


@dataclass(frozen=True)
class HarborRunResult:
    command: tuple[str, ...]
    returncode: int
    job_dir: Path
    run_id: int
    summary: RunSummary


@dataclass(frozen=True)
class ImportedTrial:
    task_id: str
    result: dict[str, Any]
    manifest: TaskArtifactManifest
    manifest_path: Path


def build_harbor_command(
    config: TerminalBenchConfig,
    *,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[str]:
    """Build the official Harbor command used for a Terminal-Bench run."""

    cmd = [
        config.harbor_executable,
        "run",
        "-d",
        config.dataset,
        "--jobs-dir",
        str(config.jobs_dir),
        "--env",
        config.environment_type,
        "--n-concurrent",
        str(config.n_concurrent),
        "--timeout-multiplier",
        str(config.timeout_multiplier),
        "--yes",
    ]
    if config.job_name:
        cmd.extend(["--job-name", config.job_name])
    if limit is not None:
        cmd.extend(["--n-tasks", str(limit)])
    if config.agent_timeout_s is not None:
        cmd.extend(["--agent-timeout", str(config.agent_timeout_s)])
    if config.verifier_timeout_s is not None:
        cmd.extend(["--verifier-timeout", str(config.verifier_timeout_s)])

    for task_id in task_ids or []:
        cmd.extend(["--include-task-name", _harbor_task_selector(task_id)])

    agent_name = config.agent.strip().lower()
    if agent_name in {"mcode", "mcode-terminal", "mcode-terminal-bench"}:
        cmd.extend(["--agent-import-path", MCODE_AGENT_IMPORT_PATH])
        cmd.extend(["--agent-kwarg", f"backend_name={config.backend_name}"])
        for key, value in sorted(config.agent_kwargs.items()):
            cmd.extend(["--agent-kwarg", f"{key}={value}"])
    else:
        cmd.extend(["--agent", config.agent])

    if config.model_id and agent_name != "oracle":
        cmd.extend(["--model", config.model_id])

    for key, value in sorted(_default_agent_env(config).items() | config.agent_env.items()):
        cmd.extend(["--agent-env", f"{key}={value}"])

    cmd.extend(config.extra_harbor_args)
    return cmd


def run_terminal_bench(
    *,
    config: TerminalBenchConfig,
    results_db: ResultsDB,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    stream_output: bool = True,
) -> HarborRunResult:
    """Run Terminal-Bench through Harbor and import official results into mCode."""

    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    command = build_harbor_command(config, limit=limit, task_ids=task_ids)
    started = datetime.now().timestamp()
    if stream_output:
        completed = subprocess.run(command, check=False)
    else:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    job_dir = resolve_job_dir(config.jobs_dir, job_name=config.job_name, since_ts=started)

    run_config = config.run_config(limit=limit, task_ids=task_ids)
    run_config["harbor_command"] = command
    run_config["planned_task_count"] = limit
    run_id = results_db.find_latest_run_by_config(BENCHMARK_NAME, run_config)
    if run_id is None:
        run_id = results_db.start_run(BENCHMARK_NAME, run_config)

    import_harbor_job(
        job_dir=job_dir,
        results_db=results_db,
        run_id=run_id,
        config=config,
    )
    summary = results_db.run_summary(run_id)
    return HarborRunResult(
        command=tuple(command),
        returncode=int(completed.returncode),
        job_dir=job_dir,
        run_id=run_id,
        summary=summary,
    )


def import_harbor_job(
    *,
    job_dir: Path,
    results_db: ResultsDB,
    run_id: int,
    config: TerminalBenchConfig,
) -> list[ImportedTrial]:
    imported: list[ImportedTrial] = []
    artifact_root = config.artifact_dir or _default_artifact_dir(results_db.path)
    for trial_dir in iter_trial_dirs(job_dir):
        trial = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
        result = trial_to_task_result(trial, trial_dir=trial_dir)
        results_db.save_task_result(run_id, result)
        manifest, manifest_path = write_trial_artifact_manifest(
            artifact_dir=artifact_root,
            trial=trial,
            trial_dir=trial_dir,
            result=result,
            config=config,
        )
        results_db.save_task_artifact_manifest(run_id, manifest, manifest_path=manifest_path)
        imported.append(
            ImportedTrial(
                task_id=str(result["task_id"]),
                result=result,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        )
    return imported


def iter_trial_dirs(job_dir: Path) -> list[Path]:
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Harbor job directory not found: {job_dir}")
    trials = [
        child
        for child in job_dir.iterdir()
        if child.is_dir() and (child / "result.json").is_file()
    ]
    return sorted(trials, key=lambda path: path.name)


def resolve_job_dir(jobs_dir: Path, *, job_name: str | None, since_ts: float | None = None) -> Path:
    if job_name:
        return jobs_dir / job_name
    candidates = [path for path in jobs_dir.iterdir() if path.is_dir()]
    if since_ts is not None:
        recent = [path for path in candidates if path.stat().st_mtime >= since_ts - 5]
        if recent:
            candidates = recent
    if not candidates:
        raise FileNotFoundError(f"No Harbor job directories found under {jobs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def trial_to_task_result(trial: dict[str, Any], *, trial_dir: Path) -> dict[str, Any]:
    task_id = normalize_task_id(str(trial.get("task_name") or trial.get("trial_name") or ""))
    rewards = _trial_rewards(trial, trial_dir=trial_dir)
    reward = _primary_reward(rewards)
    exception = (
        trial.get("exception_info") if isinstance(trial.get("exception_info"), dict) else None
    )
    agent_context = trial.get("agent_result") if isinstance(trial.get("agent_result"), dict) else {}
    prompt_tokens = _optional_int(agent_context.get("n_input_tokens"))
    completion_tokens = _optional_int(agent_context.get("n_output_tokens"))
    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    submission = {
        "harbor": {
            "trial_name": trial.get("trial_name"),
            "trial_uri": trial.get("trial_uri"),
            "task_name": trial.get("task_name"),
            "task_checksum": trial.get("task_checksum"),
            "rewards": rewards,
            "reward": reward,
            "trial_dir": str(trial_dir),
        }
    }
    if exception:
        submission["harbor"]["exception"] = exception
    return {
        "task_id": task_id,
        "passed": bool((reward or 0) >= 1),
        "attempts_used": 1,
        "time_ms": _elapsed_ms(trial.get("started_at"), trial.get("finished_at")),
        "exit_code": None,
        "timed_out": bool(exception and exception.get("exception_type") in _TIMEOUT_EXCEPTIONS),
        "stdout": _read_limited(trial_dir / "verifier" / "test-stdout.txt"),
        "stderr": _read_limited(trial_dir / "verifier" / "test-stderr.txt"),
        "error": exception.get("exception_message") if exception else None,
        "code_sha256": trial.get("task_checksum"),
        "terminal_reason": "submitted" if (reward or 0) >= 1 else _terminal_reason(exception),
        "zero_edit": False,
        "zero_verification": False,
        "verification_succeeded": bool((reward or 0) >= 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_model": _trial_model_name(trial),
        "submission_json": json.dumps(submission, sort_keys=True, default=str),
    }


def write_trial_artifact_manifest(
    *,
    artifact_dir: Path,
    trial: dict[str, Any],
    trial_dir: Path,
    result: dict[str, Any],
    config: TerminalBenchConfig,
) -> tuple[TaskArtifactManifest, Path]:
    task_id = str(result["task_id"])
    store = TaskArtifactStore.from_task(
        artifact_dir=artifact_dir,
        benchmark=BENCHMARK_NAME,
        task_id=task_id,
    )
    rewards = _trial_rewards(trial, trial_dir=trial_dir)
    report = {
        "trial_result": trial,
        "harbor_job_dir": str(trial_dir.parent),
        "harbor_trial_dir": str(trial_dir),
    }
    evaluation = store.write_evaluation(
        source_candidate_index=0,
        evaluator_name="harbor-terminal-bench",
        passed=bool(result["passed"]),
        timed_out=bool(result.get("timed_out", False)),
        exit_code=None,
        report=report,
        stdout=result.get("stdout"),
        stderr=result.get("stderr"),
        error_class=_exception_type(trial),
        runtime_ms=_optional_int(result.get("time_ms")),
        metadata={
            "reward": _primary_reward(rewards),
            "rewards": rewards,
            "trial_name": trial.get("trial_name"),
            "trial_uri": trial.get("trial_uri"),
            "trial_dir": str(trial_dir),
            "job_dir": str(trial_dir.parent),
            "agent": config.agent,
            "dataset": config.dataset,
            "environment_type": config.environment_type,
        },
    )
    task_ref = store.build_task_ref(
        repo_id=config.dataset,
        task_digest=make_task_digest(
            benchmark=BENCHMARK_NAME,
            task_id=task_id,
            repo_id=config.dataset,
            metadata={
                "task_name": trial.get("task_name"),
                "task_checksum": trial.get("task_checksum"),
            },
        ),
        metadata={
            "task_name": trial.get("task_name"),
            "task_checksum": trial.get("task_checksum"),
            "trial_name": trial.get("trial_name"),
        },
    )
    manifest = TaskArtifactManifest(
        schema_version=SCHEMA_VERSION,
        phase="run",
        generated_at=iso_utc_now(),
        run_config_digest=digest_json(config.run_config(limit=None, task_ids=None)),
        code_sha=_current_code_sha(),
        model_id=config.model_id,
        backend_name=config.backend_name,
        task=task_ref,
        candidates=(),
        evaluations=(evaluation,),
        metadata={
            "kind": "harbor-terminal-bench-trial",
            "harbor_job_dir": str(trial_dir.parent),
            "harbor_trial_dir": str(trial_dir),
            "harbor_result_path": str(trial_dir / "result.json"),
            "verifier_stdout_path": str(trial_dir / "verifier" / "test-stdout.txt"),
            "verifier_stderr_path": str(trial_dir / "verifier" / "test-stderr.txt"),
            "reward_path": _reward_path(trial_dir),
        },
    )
    return manifest, store.write_manifest(manifest)


def normalize_task_id(task_name: str) -> str:
    text = task_name.strip()
    for prefix in ("terminal-bench/", "terminal_bench/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _harbor_task_selector(task_id: str) -> str:
    text = task_id.strip()
    if "/" in text:
        return text
    return f"terminal-bench/{text}"


def _default_agent_env(config: TerminalBenchConfig) -> dict[str, str]:
    return {
        "MCODE_BACKEND": config.backend_name,
        "MCODE_MODEL": config.model_id,
    }


def _default_artifact_dir(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.stem}-artifacts")


def _trial_rewards(trial: dict[str, Any], *, trial_dir: Path) -> dict[str, float]:
    verifier = (
        trial.get("verifier_result") if isinstance(trial.get("verifier_result"), dict) else {}
    )
    raw_rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if isinstance(raw_rewards, dict) and raw_rewards:
        return {str(key): float(value) for key, value in raw_rewards.items() if _is_number(value)}
    reward_json = trial_dir / "verifier" / "reward.json"
    if reward_json.is_file():
        try:
            raw = json.loads(reward_json.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(key): float(value) for key, value in raw.items() if _is_number(value)}
        except Exception:
            pass
    reward_txt = trial_dir / "verifier" / "reward.txt"
    if reward_txt.is_file():
        try:
            return {"reward": float(reward_txt.read_text(encoding="utf-8").strip())}
        except ValueError:
            return {}
    return {}


def _primary_reward(rewards: dict[str, float]) -> float | None:
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    if rewards:
        return sum(rewards.values()) / len(rewards)
    return None


def _elapsed_ms(started_at: object, finished_at: object) -> int:
    start = _parse_datetime(started_at)
    end = _parse_datetime(finished_at)
    if start is None or end is None:
        return 0
    return max(0, int((end - start).total_seconds() * 1000))


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _read_limited(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= _LOG_LIMIT_CHARS:
        return text
    return text[:_LOG_LIMIT_CHARS] + f"\n... truncated to {_LOG_LIMIT_CHARS} characters ..."


def _terminal_reason(exception: dict[str, Any] | None) -> str:
    if not exception:
        return "wrong_answer"
    exc_type = str(exception.get("exception_type") or "")
    if exc_type in _TIMEOUT_EXCEPTIONS:
        return "timeout"
    return "error"


def _exception_type(trial: dict[str, Any]) -> str | None:
    exception = trial.get("exception_info")
    if isinstance(exception, dict):
        value = exception.get("exception_type")
        return str(value) if value else None
    return None


def _trial_model_name(trial: dict[str, Any]) -> str | None:
    agent_info = trial.get("agent_info") if isinstance(trial.get("agent_info"), dict) else {}
    model_info = agent_info.get("model_info") if isinstance(agent_info, dict) else None
    if not isinstance(model_info, dict):
        return None
    provider = model_info.get("provider")
    name = model_info.get("name")
    if provider and name:
        return f"{provider}/{name}"
    return str(name) if name else None


def _reward_path(trial_dir: Path) -> str | None:
    for path in (trial_dir / "verifier" / "reward.json", trial_dir / "verifier" / "reward.txt"):
        if path.is_file():
            return str(path)
    return None


def _current_code_sha() -> str | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return sha or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
