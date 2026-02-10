"""File tools — read / write / delete inside the Docker container."""

from __future__ import annotations

import base64
import shlex

import yuutools as yt

from yuuagents.context import DockerExecutor

_EXIT_CODE_MARKER = "__YAGENTS_EXIT_CODE__="
_APPLY_PATCH_BIN = "yagents-apply-patch"


def _parse_exec_result(output: str) -> tuple[int, str]:
    assert isinstance(output, str)
    lines = output.splitlines()
    assert lines, "empty tool output"
    marker = lines[-1].strip()
    assert marker.startswith(_EXIT_CODE_MARKER), "missing exit code marker"
    code_raw = marker.removeprefix(_EXIT_CODE_MARKER).strip()
    assert code_raw.isdigit(), "invalid exit code marker"
    code = int(code_raw)
    body = "\n".join(lines[:-1]).strip()
    return code, body


async def _write_file_patch(
    *,
    path: str,
    patch: str,
    container: str,
    docker: DockerExecutor,
) -> str:
    assert isinstance(path, str)
    assert path.startswith("/")
    assert isinstance(patch, str)
    patch = patch.replace("\r\n", "\n").replace("\r", "\n").strip()
    assert patch

    patch_b64 = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    cmd = (
        f"printf %s {shlex.quote(patch_b64)}"
        f" | base64 -d"
        f" | {_APPLY_PATCH_BIN} {shlex.quote(path)}"
    )
    raw = await docker.exec(container, cmd, timeout=60)
    code, summary = _parse_exec_result(raw)
    assert code == 0, summary or raw
    assert summary
    return summary


@yt.tool(
    params={"path": "Absolute file path to read"},
    description="Read the contents of a file.",
)
async def read_file(
    path: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: DockerExecutor = yt.depends(lambda ctx: ctx.docker),
) -> str:
    cmd = f"cat {shlex.quote(path)}"
    return await docker.exec(container, cmd, timeout=30)


@yt.tool(
    params={
        "path": "Absolute file path to patch",
        "patch": "Unified diff patch to apply to the file",
    },
    description="Apply a unified diff patch to a file and return a diff summary.",
)
async def write_file(
    path: str,
    patch: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: DockerExecutor = yt.depends(lambda ctx: ctx.docker),
) -> str:
    return await _write_file_patch(
        path=path,
        patch=patch,
        container=container,
        docker=docker,
    )


@yt.tool(
    params={"path": "Absolute file path to delete"},
    description="Delete a file.",
)
async def delete_file(
    path: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: DockerExecutor = yt.depends(lambda ctx: ctx.docker),
) -> str:
    cmd = f"rm -f {shlex.quote(path)}"
    return await docker.exec(container, cmd, timeout=30)
