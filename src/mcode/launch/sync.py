from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mcode.launch.models import SyncMode, SyncSpec, WorkspaceHandle


@dataclass(frozen=True)
class WorkspaceSignatureInput:
    repo_url: str
    ref_sha: str
    overlay_patch_sha: str
    bootstrap_key: str


@dataclass(frozen=True)
class SyncPlan:
    signature: str
    repo_url: str
    ref_sha: str
    overlay_patch_sha: str
    remote_path: str
    mode: SyncMode
    is_noop: bool
    diff_summary: str


def build_workspace_signature(data: WorkspaceSignatureInput) -> str:
    payload = json.dumps(
        {
            "repo_url": data.repo_url,
            "ref_sha": data.ref_sha,
            "overlay_patch_sha": data.overlay_patch_sha,
            "bootstrap_key": data.bootstrap_key,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def resolve_git_ref(repo_root: Path, ref: str) -> str:
    return _git_output(repo_root, "rev-parse", ref).strip()


def repo_remote_url(repo_root: Path) -> str:
    return _git_output(repo_root, "remote", "get-url", "origin").strip()


def tracked_overlay_patch(repo_root: Path, ref: str) -> str:
    return _git_output(repo_root, "diff", "--binary", ref, "--", ".")


def tracked_working_tree_fingerprint(repo_root: Path) -> str:
    paths = _git_output(repo_root, "ls-files", "-z")
    digest = hashlib.sha256()
    for raw_path in paths.split("\0"):
        if not raw_path:
            continue
        path = repo_root / raw_path
        digest.update(raw_path.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def list_untracked_files(repo_root: Path) -> list[str]:
    output = _git_output(repo_root, "ls-files", "--others", "--exclude-standard")
    return [line for line in output.splitlines() if line]


def build_sync_plan(
    repo_root: Path,
    *,
    sync: SyncSpec,
    workspace_root: str,
    existing: WorkspaceHandle | None = None,
) -> SyncPlan:
    ref_sha = resolve_git_ref(repo_root, sync.ref)
    remote = repo_remote_url(repo_root)
    if sync.mode == SyncMode.WORKING_TREE:
        overlay_patch_sha = tracked_working_tree_fingerprint(repo_root)
    elif sync.mode == SyncMode.GIT_REF:
        overlay_patch_sha = ""
    else:
        overlay_patch_sha = hashlib.sha256(
            tracked_overlay_patch(repo_root, sync.ref).encode()
        ).hexdigest()
    signature = build_workspace_signature(
        WorkspaceSignatureInput(
            repo_url=remote,
            ref_sha=ref_sha,
            overlay_patch_sha=overlay_patch_sha,
            bootstrap_key=sync.bootstrap_key,
        )
    )
    remote_path = f"{workspace_root.rstrip('/')}/workspaces/{signature}"
    return SyncPlan(
        signature=signature,
        repo_url=remote,
        ref_sha=ref_sha,
        overlay_patch_sha=overlay_patch_sha,
        remote_path=remote_path,
        mode=sync.mode,
        is_noop=existing is not None
        and existing.signature == signature
        and existing.path == remote_path,
        diff_summary="clean" if not overlay_patch_sha else overlay_patch_sha[:12],
    )


def _git_output(repo_root: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return res.stdout
