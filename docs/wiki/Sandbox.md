# Docker Sandbox

Implementation: `src/mcode/execution/sandbox.py`.

Default properties:

- Network disabled (`network_disabled=True`)
- Read-only code mount
- Runs as `nobody` (`65534:65534`)
- Memory + PID limits
- `--timeout` is enforced by killing the container if it exceeds `timeout_s`

Note: this is “secure-ish” isolation; treat model-generated code as untrusted.
