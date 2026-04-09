from __future__ import annotations

import shlex


def build_remote_healthcheck_command(base_url: str) -> str:
    root = base_url.rstrip("/")
    return (
        "python - <<'PY'\n"
        "import sys\n"
        "import urllib.request\n"
        "urls = [\n"
        f"    {root!r} + '/health',\n"
        f"    {root!r} + '/models',\n"
        f"    {root!r} + '/v1/models',\n"
        f"    {root!r} + '/api/tags',\n"
        "]\n"
        "for url in urls:\n"
        "    try:\n"
        "        with urllib.request.urlopen(url, timeout=5) as response:\n"
        "            if 200 <= response.status < 300:\n"
        "                sys.exit(0)\n"
        "    except Exception:\n"
        "        continue\n"
        "sys.exit(1)\n"
        "PY"
    )


def build_uv_sync_command(bootstrap_key: str) -> str:
    if not bootstrap_key.startswith("uv-sync:"):
        raise ValueError(f"Unsupported bootstrap key: {bootstrap_key}")
    extras = [part.strip() for part in bootstrap_key.removeprefix("uv-sync:").split(",") if part]
    args = ["uv", "run", "mcode", "deps", "sync", "--no-dev"]
    for extra in extras:
        args.extend(["--extra", extra])
    return " ".join(shlex.quote(arg) for arg in args)
