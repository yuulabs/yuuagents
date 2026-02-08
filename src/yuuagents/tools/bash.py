"""execute_bash — run commands in a Docker container."""

from __future__ import annotations

import yuutools as yt


@yt.tool(
    params={
        "command": "The bash command to execute",
        "timeout": "Timeout in seconds (default 120, max 600)",
    },
    description="Execute a bash command in the Docker container.",
)
async def execute_bash(
    command: str,
    timeout: int = 120,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: object = yt.depends(lambda ctx: ctx.docker),
) -> str:
    timeout = max(1, min(timeout, 600))
    return await docker.exec(container, command, timeout)
