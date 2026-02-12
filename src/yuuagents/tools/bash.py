"""execute_bash — run commands in a Docker container."""

from __future__ import annotations

import yuutools as yt

from yuuagents.context import DockerExecutor


@yt.tool(
    params={
        "command": "The bash command to execute",
        "timeout": "Timeout in seconds (default 120, max 600)",
    },
    description=(
        "Execute a bash command in the Docker container(work dir is `/home/yuu`). Commands run in a persistent "
        "terminal-like session: working directory and environment variables persist "
        "across calls."
    ),
)
async def execute_bash(
    command: str,
    timeout: int = 120,
    session_id: str = yt.depends(lambda ctx: ctx.task_id),
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: DockerExecutor = yt.depends(lambda ctx: ctx.docker),
) -> str:
    timeout = max(1, min(timeout, 600))
    return await docker.exec_terminal(container, session_id, command, timeout)
