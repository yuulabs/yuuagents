"""execute_skill_cli — run skill-provided CLI commands on the host (restricted)."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

import yuutools as yt

from yuuagents.context import CliGuard


_FORBIDDEN_PROGS = {
    "bash",
    "dash",
    "fish",
    "ksh",
    "sh",
    "zsh",
    "rm",
    "rmdir",
    "mv",
    "dd",
    "mkfs",
    "fdisk",
    "sfdisk",
    "parted",
    "kill",
    "killall",
    "pkill",
    "chown",
    "chmod",
    "chgrp",
    "useradd",
    "userdel",
    "groupadd",
    "groupdel",
    "passwd",
    "sudo",
    "su",
    "doas",
    "pkexec",
    "systemctl",
    "service",
    "shutdown",
    "reboot",
    "poweroff",
    "mount",
    "umount",
    "python",
    "python3",
    "node",
    "perl",
    "ruby",
    "php",
    "lua",
}

_FORBIDDEN_TOKENS = {
    ";",
    "|",
    "&",
    "&&",
    "||",
    ">",
    ">>",
    "<",
    "<<",
}


def _is_forbidden_prog(prog: str) -> bool:
    assert isinstance(prog, str)
    prog = prog.strip()
    if not prog:
        return True
    base = Path(prog).name
    return base in _FORBIDDEN_PROGS or base.startswith("mkfs.")


def _validate_cli_command(command: str) -> list[str]:
    assert isinstance(command, str)
    command = command.strip()
    assert command
    assert "\x00" not in command
    assert "\n" not in command and "\r" not in command

    argv = shlex.split(command, posix=True)
    assert argv

    if any(t in _FORBIDDEN_TOKENS for t in argv):
        raise ValueError("dangerous shell control operators are not allowed")

    if any("$(" in t or "`" in t for t in argv):
        raise ValueError("shell expansions are not allowed")

    if _is_forbidden_prog(argv[0]):
        raise ValueError(f"dangerous command is not allowed: {Path(argv[0]).name}")

    return argv


@yt.tool(
    params={
        "command": "Skill CLI command to execute (e.g. `ybot im send ...`, `cat /path/to/SKILL.md`)",
        "timeout": "Timeout in seconds (default 300, max 3600)",
    },
    description=(
        "执行 skill 提供的 CLI 命令。可用 skill 及其命令摘要见 system prompt 中的 <available_skills>。"
        "首次调用某 skill 前，必须先 `cat <location>` 阅读其 SKILL.md 确认参数格式，不要猜测参数。"
    ),
)
async def execute_skill_cli(
    command: str,
    timeout: int = 300,
    cli_guard: CliGuard | None = yt.depends(lambda ctx: ctx.cli_guard),
) -> str:
    timeout = max(1, min(timeout, 3600))
    argv = _validate_cli_command(command)

    if cli_guard is not None:
        cli_guard(argv)

    home = Path.home().expanduser().resolve()
    assert home.is_dir()

    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(home),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except TimeoutError:
            proc.kill()
        return f"[ERROR] Command timed out after {timeout}s"

    out = "" if out_b is None else out_b.decode("utf-8", errors="replace")
    out = out.rstrip()
    code = 0 if proc.returncode is None else int(proc.returncode)
    if code != 0:
        return f"[ERROR] 命令执行失败 (exit {code})\n{out}" if out else f"[ERROR] 命令执行失败 (exit {code})"
    return out or "(无输出)"
