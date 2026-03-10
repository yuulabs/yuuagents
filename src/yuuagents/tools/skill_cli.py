"""execute_skill_cli — run skill-provided CLI commands on the host (restricted)."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

import yuutools as yt
from yuutools._depends import DependencyMarker

from yuuagents.context import CliGuard
from yuuagents.running_tools import OutputBuffer


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
    ">",
    ">>",
    "<",
}

# Tokens that separate commands in a pipeline/chain.
_CHAIN_OPERATORS = {"|", "&&", "||"}


def _is_forbidden_prog(prog: str) -> bool:
    assert isinstance(prog, str)
    prog = prog.strip()
    if not prog:
        return True
    base = Path(prog).name
    return base in _FORBIDDEN_PROGS or base.startswith("mkfs.")


def _validate_segment(tokens: list[str]) -> list[str]:
    """Validate a single command segment (no pipe)."""
    assert tokens

    if any(t in _FORBIDDEN_TOKENS for t in tokens):
        raise ValueError("dangerous shell control operators are not allowed")

    if any("$(" in t or "`" in t for t in tokens):
        raise ValueError("shell expansions are not allowed")

    if _is_forbidden_prog(tokens[0]):
        raise ValueError(f"dangerous command is not allowed: {Path(tokens[0]).name}")

    return tokens


import re

_HEREDOC_RE = re.compile(r"<<-?\s*'?(\w+)'?")


def _parse_heredoc(command: str) -> tuple[str, bool]:
    """Split command into (first_line, heredoc_body, ...) and validate structure.

    Returns (command_line, has_heredoc).
    If heredoc is present, validates that nothing follows the closing delimiter.
    """
    lines = command.split("\n")
    first_line = lines[0]

    m = _HEREDOC_RE.search(first_line)
    if not m:
        # Has << but no valid delimiter?
        if "<<" in first_line:
            raise ValueError("heredoc with empty delimiter")
        return command, False

    delimiter = m.group(1)
    if not delimiter:
        raise ValueError("heredoc with empty delimiter")

    # Find closing delimiter
    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == delimiter:
            close_idx = i
            break

    if close_idx is None:
        raise ValueError("unclosed heredoc — missing closing delimiter")

    # Nothing allowed after closing delimiter
    trailing = [l for l in lines[close_idx + 1:] if l.strip()]
    if trailing:
        raise ValueError("no commands allowed after heredoc closing delimiter")

    # Return just the command part (before <<) for validation
    cmd_part = first_line[:m.start()].strip()
    return cmd_part, True


def _validate_cli_command(command: str) -> tuple[list[str], bool]:
    assert isinstance(command, str)
    command = command.strip()
    assert command
    assert "\x00" not in command

    # Check for heredoc first (multi-line)
    cmd_to_validate, has_heredoc = _parse_heredoc(command)

    if not has_heredoc:
        # Single-line: no newlines allowed
        assert "\n" not in command and "\r" not in command

    argv = shlex.split(cmd_to_validate, posix=True)
    assert argv

    # Split by chain operators (|, &&, ||) and validate each segment
    segments: list[list[str]] = [[]]
    for token in argv:
        if token in _CHAIN_OPERATORS:
            segments.append([])
        else:
            segments[-1].append(token)

    for seg in segments:
        if not seg:
            raise ValueError("empty command in pipeline")
        _validate_segment(seg)

    has_chain = any(t in _CHAIN_OPERATORS for t in argv)

    return argv, has_chain or has_heredoc


async def _read_streaming(
    proc: asyncio.subprocess.Process, buffer: OutputBuffer | None
) -> bytes:
    """Read stdout incrementally, writing chunks to buffer if provided."""
    assert proc.stdout is not None
    chunks: list[bytes] = []
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if buffer is not None and hasattr(buffer, "write"):
            buffer.write(chunk)
    return b"".join(chunks)


@yt.tool(
    params={
        "command": "Skill CLI command to execute (e.g. `ybot im send ...`, `cat /path/to/SKILL.md`). "
                   "Supports heredoc for passing structured data: `ybot im send --ctx 3 << 'EOF'\\n{json}\\nEOF`",
        "timeout": "Timeout in seconds (default 300, max 3600)",
    },
    description=(
        "执行 skill 提供的 CLI 命令。可用 skill 及其命令摘要见 system prompt 中的 <available_skills>。"
        "首次调用某 skill 前，必须先 `cat <location>` 阅读其 SKILL.md 确认参数格式，不要猜测参数。"
        "支持管道 (|) 和组合命令 (&&, ||)，可一次执行多条命令以节省调用次数。"
        "需要传入复杂数据（如 JSON）时，使用 heredoc 语法: command << 'EOF'\\ndata\\nEOF"
    ),
)
async def execute_skill_cli(
    command: str,
    timeout: int = 300,
    cli_guard: CliGuard | None = yt.depends(lambda ctx: ctx.cli_guard),
    output_buffer: OutputBuffer | None = yt.depends(
        lambda ctx: ctx.current_output_buffer
    ),
    subprocess_env: dict | None = yt.depends(lambda ctx: ctx.subprocess_env),
) -> str:
    timeout = max(1, min(timeout, 3600))
    argv, has_chain = _validate_cli_command(command)

    if isinstance(cli_guard, DependencyMarker):
        cli_guard = None
    if isinstance(output_buffer, DependencyMarker):
        output_buffer = None
    if isinstance(subprocess_env, DependencyMarker):
        subprocess_env = None

    if cli_guard is not None:
        cli_guard(argv)

    home = Path.home().expanduser().resolve()
    assert home.is_dir()

    # Use explicit env if provided (isolates concurrent agents from each other),
    # otherwise fall back to inheriting the process environment.
    env_for_proc = subprocess_env or None

    if has_chain:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(home),
            env=env_for_proc,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(home),
            env=env_for_proc,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    try:
        out_b = await asyncio.wait_for(
            _read_streaming(proc, output_buffer), timeout=timeout
        )
        await proc.wait()
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
