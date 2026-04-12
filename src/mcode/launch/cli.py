"""Typer app for `mcode launch`.

Subcommand tree (per target, no unified dispatcher):

    mcode launch bluevela      --model ... [--shards N] [--reuse-server ID]
    mcode launch local-vllm    --model ...
    mcode launch local-ollama  --model ...

    mcode launch status        [--json] [--raw]
    mcode launch logs   <id>
    mcode launch fetch  <id>   [--destination DIR] [--snapshot]
    mcode launch stop   <id> | --all
    mcode launch doctor <target> [--deep] [--init]
    mcode launch gc            [--older-than 7d]

CLI catches LaunchError and renders:

    ✗ {what}
      why:  {why}
      next: {next}
      logs: {logs}

MCODE_DEBUG=1 disables error formatting (tracebacks leak through).
"""

from __future__ import annotations
