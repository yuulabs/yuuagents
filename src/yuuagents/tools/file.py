"""File tools — read / write / delete inside the Docker container."""

from __future__ import annotations

import base64
import json
import shlex

import yuutools as yt

from yuuagents.context import DockerExecutor

_EXIT_CODE_MARKER = "__YAGENTS_EXIT_CODE__="
_APPLY_PATCH_BIN = "yagents-apply-patch"
_READ_FILE_MAX_BYTES = 2048
_READ_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


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
) -> str | list[dict[str, object]]:
    assert isinstance(path, str)
    assert path.startswith("/")

    path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
    mime_json = json.dumps(_READ_IMAGE_EXTS, sort_keys=True)
    cmd = (
        "python3 - <<'PY'\n"
        "import base64\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"path = base64.b64decode({path_b64!r}).decode('utf-8')\n"
        f"mime_map = json.loads({mime_json!r})\n"
        f"max_bytes = {_READ_FILE_MAX_BYTES}\n"
        "p = Path(path)\n"
        "if not p.is_file():\n"
        "    raise RuntimeError(f'File not found: {path}')\n"
        "suffix = p.suffix.lower()\n"
        "data = p.read_bytes()\n"
        "if suffix in mime_map:\n"
        "    payload = [\n"
        "        {'type': 'image_url', 'image_url': {'url': 'data:' + mime_map[suffix] + ';base64,' + base64.b64encode(data).decode('ascii')}}\n"
        "    ]\n"
        "    print(json.dumps({'kind': 'image', 'payload': payload}, ensure_ascii=False))\n"
        "elif len(data) > max_bytes:\n"
        "    raise RuntimeError(f'File too large to read safely: {len(data)} bytes > {max_bytes} bytes')\n"
        "elif b'\\x00' in data:\n"
        "    raise RuntimeError('Binary file is not supported')\n"
        "else:\n"
        "    text = data.decode('utf-8')\n"
        "    print(json.dumps({'kind': 'text', 'payload': text}, ensure_ascii=False))\n"
        "PY"
    )
    raw = await docker.exec(container, cmd, timeout=30)
    payload = json.loads(raw)
    kind = payload["kind"]
    if kind == "text":
        return str(payload["payload"])
    assert kind == "image"
    assert isinstance(payload["payload"], list)
    return payload["payload"]


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


@yt.tool(
    params={
        "path": "Absolute file path to edit",
        "old_string": "Exact string to replace (must occur exactly once)",
        "new_string": "Replacement string",
    },
    description="Replace an exact string in a file (requires exactly one occurrence).",
)
async def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    container: str = yt.depends(lambda ctx: ctx.docker_container),
    docker: DockerExecutor = yt.depends(lambda ctx: ctx.docker),
) -> str:
    assert isinstance(path, str)
    assert path
    assert isinstance(old_string, str)
    assert isinstance(new_string, str)

    path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
    old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
    new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")

    cmd = (
        "python3 - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        "\n"
        f"path = base64.b64decode({path_b64!r}).decode('utf-8')\n"
        f"old = base64.b64decode({old_b64!r}).decode('utf-8')\n"
        f"new = base64.b64decode({new_b64!r}).decode('utf-8')\n"
        "\n"
        "p = Path(path)\n"
        "text = p.read_text(encoding='utf-8')\n"
        "count = text.count(old)\n"
        "if count != 1:\n"
        "    raise RuntimeError(f'Expected 1 occurrence, found {count}')\n"
        "p.write_text(text.replace(old, new, 1), encoding='utf-8')\n"
        "print(f'Edited {path}')\n"
        "PY"
    )
    return await docker.exec(container, cmd, timeout=30)
