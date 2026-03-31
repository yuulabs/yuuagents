"""File tools — read / write / delete inside the Docker container."""

from __future__ import annotations

import base64
import json
import shlex

import yuutools as yt

from yuuagents.capabilities import DockerCapability, require_docker

_READ_FILE_MAX_LINES = 200
_READ_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@yt.tool(
    params={
        "path": "Absolute file path to read",
        "start_line": "1-based first line to read (default: 1)",
        "end_line": "1-based last line to read, inclusive (default: EOF)",
    },
    description="Read the contents of a file.",
)
async def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    docker: DockerCapability = yt.depends(require_docker),
) -> str | list[dict[str, object]]:
    assert isinstance(path, str)
    assert path.startswith("/")
    assert isinstance(start_line, int)
    assert start_line >= 1
    assert end_line is None or (isinstance(end_line, int) and end_line >= start_line)

    path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
    mime_json = json.dumps(_READ_IMAGE_EXTS, sort_keys=True)
    end_line_expr = "None" if end_line is None else str(end_line)
    cmd = (
        "python3 - <<'PY'\n"
        "import base64\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"path = base64.b64decode({path_b64!r}).decode('utf-8')\n"
        f"mime_map = json.loads({mime_json!r})\n"
        f"max_lines = {_READ_FILE_MAX_LINES}\n"
        f"start_line = {start_line}\n"
        f"end_line = {end_line_expr}\n"
        "p = Path(path)\n"
        "if not p.is_file():\n"
        "    print(json.dumps({'kind': 'error', 'message': f'File not found: {path}'}, ensure_ascii=False))\n"
        "    import sys; sys.exit(0)\n"
        "suffix = p.suffix.lower()\n"
        "data = p.read_bytes()\n"
        "if suffix in mime_map:\n"
        "    payload = [\n"
        "        {'type': 'image_url', 'image_url': {'url': 'data:' + mime_map[suffix] + ';base64,' + base64.b64encode(data).decode('ascii')}}\n"
        "    ]\n"
        "    print(json.dumps({'kind': 'image', 'payload': payload}, ensure_ascii=False))\n"
        "elif b'\\x00' in data:\n"
        "    print(json.dumps({'kind': 'error', 'message': 'Binary file is not supported'}, ensure_ascii=False))\n"
        "    import sys; sys.exit(0)\n"
        "else:\n"
        "    text = data.decode('utf-8')\n"
        "    lines = text.splitlines()\n"
        "    total_lines = len(lines)\n"
        "    if total_lines == 0:\n"
        "        selected_start = start_line\n"
        "        selected_end = start_line - 1 if end_line is None else min(end_line, start_line - 1)\n"
        "        selected_text = ''\n"
        "        selected_count = 0\n"
        "    else:\n"
        "        selected_start = min(start_line, total_lines)\n"
        "        selected_end = total_lines if end_line is None else min(end_line, total_lines)\n"
        "        if selected_end < selected_start:\n"
        "            selected_text = ''\n"
        "            selected_count = 0\n"
        "        else:\n"
        "            selected = lines[selected_start - 1:selected_end]\n"
        "            selected_text = '\\n'.join(selected)\n"
        "            if text.endswith('\\n') and selected_end == total_lines:\n"
        "                selected_text += '\\n'\n"
        "            selected_count = len(selected)\n"
        "    if selected_count > max_lines:\n"
        "        print(json.dumps({\n"
        "            'kind': 'error',\n"
        "            'message': f'Too many lines to read at once: {selected_count} lines > {max_lines} lines. '\n"
        "                       f'Total lines: {total_lines}. Use start_line/end_line to read in chunks '\n"
        "                       f'(e.g. start_line=1, end_line={max_lines}).'\n"
        "        }, ensure_ascii=False))\n"
        "        import sys; sys.exit(0)\n"
        "    print(json.dumps({\n"
        "        'kind': 'text',\n"
        "        'payload': selected_text,\n"
        "        'total_lines': total_lines,\n"
        "        'start_line': selected_start,\n"
        "        'end_line': selected_end,\n"
        "        'returned_lines': selected_count,\n"
        "    }, ensure_ascii=False))\n"
        "PY"
    )
    raw = await docker.executor.exec(docker.container_id, cmd, timeout=30)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        msg = raw.strip()
        if msg:
            return f"[read_file error] {msg}"
        return "[read_file error] read_file returned empty response"
    kind = payload["kind"]
    if kind == "error":
        return f"[read_file error] {payload['message']}"
    if kind == "text":
        total_lines = int(payload["total_lines"])
        start = int(payload["start_line"])
        end = int(payload["end_line"])
        returned = int(payload["returned_lines"])
        text = str(payload["payload"])
        header = (
            f"[read_file] path={path} lines={start}-{end} returned_lines={returned} "
            f"total_lines={total_lines}"
        )
        if text:
            return f"{header}\n{text}"
        return header
    assert kind == "image"
    assert isinstance(payload["payload"], list)
    return payload["payload"]


@yt.tool(
    params={"path": "Absolute file path to delete"},
    description="Delete a file.",
)
async def delete_file(
    path: str,
    docker: DockerCapability = yt.depends(require_docker),
) -> str:
    cmd = f"rm -f {shlex.quote(path)}"
    return await docker.executor.exec(docker.container_id, cmd, timeout=30)


@yt.tool(
    params={
        "path": "Absolute file path to edit",
        "old_string": "Exact string to replace (must occur exactly once)",
        "start_line": "1-based first line to replace",
        "end_line": "1-based last line to replace, inclusive",
        "new_string": "Replacement string",
    },
    description="Replace an exact string or a line range in a file.",
)
async def edit_file(
    path: str,
    new_string: str,
    old_string: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    docker: DockerCapability = yt.depends(require_docker),
) -> str:
    assert isinstance(path, str)
    assert path
    assert isinstance(new_string, str)
    has_old_string = old_string is not None
    has_line_range = start_line is not None or end_line is not None
    if has_old_string == has_line_range:
        raise ValueError(
            "Provide exactly one edit selector: either old_string or start_line/end_line"
        )
    if has_old_string:
        assert isinstance(old_string, str)
    else:
        if start_line is None or end_line is None:
            raise ValueError("start_line and end_line must be provided together")
        assert isinstance(start_line, int)
        assert isinstance(end_line, int)
        if start_line < 1 or end_line < start_line:
            raise ValueError("line range must satisfy 1 <= start_line <= end_line")

    path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
    new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")
    old_expr = (
        repr(base64.b64encode(old_string.encode("utf-8")).decode("ascii"))
        if old_string is not None
        else "None"
    )
    start_line_expr = "None" if start_line is None else str(start_line)
    end_line_expr = "None" if end_line is None else str(end_line)

    cmd = (
        "python3 - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        "\n"
        f"path = base64.b64decode({path_b64!r}).decode('utf-8')\n"
        f"new = base64.b64decode({new_b64!r}).decode('utf-8')\n"
        f"old_b64 = {old_expr}\n"
        f"start_line = {start_line_expr}\n"
        f"end_line = {end_line_expr}\n"
        "old = base64.b64decode(old_b64).decode('utf-8') if old_b64 is not None else None\n"
        "\n"
        "p = Path(path)\n"
        "text = p.read_text(encoding='utf-8')\n"
        "if old is not None:\n"
        "    count = text.count(old)\n"
        "    if count != 1:\n"
        "        raise RuntimeError(f'Expected 1 occurrence, found {count}')\n"
        "    updated = text.replace(old, new, 1)\n"
        "else:\n"
        "    assert start_line is not None and end_line is not None\n"
        "    lines = text.splitlines(keepends=True)\n"
        "    total_lines = len(lines)\n"
        "    if end_line > total_lines:\n"
        "        raise RuntimeError(\n"
        "            f'Line range out of bounds: lines={start_line}-{end_line}, total_lines={total_lines}'\n"
        "        )\n"
        "    replacement = new.splitlines(keepends=True)\n"
        "    updated = ''.join(lines[: start_line - 1] + replacement + lines[end_line:])\n"
        "p.write_text(updated, encoding='utf-8')\n"
        "print(f'Edited {path}')\n"
        "PY"
    )
    raw = await docker.executor.exec(docker.container_id, cmd, timeout=30)
    msg = raw.strip()
    if msg.startswith("Traceback (most recent call last):"):
        raise RuntimeError(msg)
    return raw
