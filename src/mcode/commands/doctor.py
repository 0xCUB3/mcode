from __future__ import annotations

import typer


def register_doctor_command(app: typer.Typer) -> None:
    app.command("doctor")(doctor_cmd)


def doctor_cmd(
    target: str = typer.Argument(
        None,
        help=(
            "optional: bluevela | local-vllm | local-ollama | terminal-bench. "
            "Omit for system-wide checks."
        ),
    ),
    deep: bool = typer.Option(False, "--deep"),
    init: bool = typer.Option(False, "--init", help="bootstrap launch.toml (bluevela only)"),
    login: str | None = typer.Option(None, "--login", help="user@host for --init"),
) -> None:
    """Run system and launch checks."""
    from mcode.bench import terminalbench_doctor
    from mcode.doctor import render_check_lines, system_checks
    from mcode.launch import bluevela, local_ollama, local_vllm
    from mcode.launch import config as config_mod
    from mcode.launch.cli import _run as _launch_run
    from mcode.launch.models import Check as _Check
    from mcode.ui.errors import MCodeError, print_error

    if init:
        if target != "bluevela":
            print_error(
                MCodeError(
                    what="--init is only supported for `bluevela`",
                    why=f"target was {target!r}",
                    next="local targets don't need probing — edit launch.toml by hand",
                )
            )
            raise typer.Exit(1)
        if not login:
            login = typer.prompt("Blue Vela login (user@host)")
        written = _launch_run(lambda: bluevela.doctor_init(login=login))
        print(f"wrote {written}")
        print(f"review with `cat {written}` and re-run `mcode doctor bluevela`")
        return

    checks: list[_Check] = []
    if target is None:
        checks.extend(system_checks())
        try:
            cfg = config_mod.load()
            checks.extend(bluevela.doctor(cfg))
            checks.extend(local_vllm.doctor(cfg))
            checks.extend(local_ollama.doctor(cfg))
        except Exception as e:
            checks.append(
                _Check(
                    name="launch config",
                    ok=False,
                    detail=str(e),
                    next="fix or recreate launch.toml; run `mcode doctor bluevela --init`",
                )
            )
    else:
        # Validate target BEFORE loading config so an unknown target produces
        # a clean error instead of surfacing an unrelated TOML parse failure.
        if target not in ("bluevela", "local-vllm", "local-ollama", "terminal-bench"):
            print_error(
                MCodeError(
                    what=f"unknown target {target!r}",
                    why="valid: bluevela, local-vllm, local-ollama, terminal-bench",
                    next="pick one or omit for system-wide checks",
                )
            )
            raise typer.Exit(1)
        if target == "terminal-bench":
            checks = terminalbench_doctor.doctor(deep=deep)
        else:
            cfg = _launch_run(config_mod.load)
            if target == "bluevela":
                checks = bluevela.doctor(cfg)
            elif target == "local-vllm":
                checks = local_vllm.doctor(cfg)
            else:
                checks = local_ollama.doctor(cfg)

    lines, any_failed = render_check_lines(checks)
    for line in lines:
        print(line)
    if any_failed:
        raise typer.Exit(1)
