"""File tools — read / write / delete inside the Docker container."""

from __future__ import annotations

import shlex

import yuutools as yt


@yt.tool(
    params={"path": "Absolute file path to read"},
    description="Read the contents of a file.",
)
async def read_file(
    path: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: object = yt.depends(lambda ctx: ctx.docker),
) -> str:
    cmd = f"cat {shlex.quote(path)}"
    return await docker.exec(container, cmd, timeout=30)


@yt.tool(
    params={
        "path": "Absolute file path to write",
        "content": "File content to write",
    },
    description="Write content to a file. Creates parent directories if needed.",
)
async def write_file(
    path: str,
    content: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: object = yt.depends(lambda ctx: ctx.docker),
) -> str:
    safe_path = shlex.quote(path)
    cmd = f"mkdir -p $(dirname {safe_path}) && cat > {safe_path} << 'YUUEOF'\n{content}\nYUUEOF"
    return await docker.exec(container, cmd, timeout=30)


@yt.tool(
    params={"path": "Absolute file path to delete"},
    description="Delete a file.",
)
async def delete_file(
    path: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: object = yt.depends(lambda ctx: ctx.docker),
) -> str:
    cmd = f"rm -f {shlex.quote(path)}"
    return await docker.exec(container, cmd, timeout=30)
