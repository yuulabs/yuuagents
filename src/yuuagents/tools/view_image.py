"""view_image — read a local image file and return multimodal content blocks."""

from __future__ import annotations

import base64
from pathlib import Path

import yuutools as yt

_MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


@yt.tool(
    params={"path": "Local file path (may include file:// prefix)"},
    description=(
        "View a local image file. Returns the image as multimodal content "
        "so you can actually see it. Use this when you encounter "
        '<image url="file:///path"/> in messages and want to see the image.'
    ),
)
async def view_image(path: str) -> list[dict]:
    """Read a local image and return it as multimodal content blocks."""
    # Strip file:// prefix if present
    if path.startswith("file://"):
        path = path[7:]

    p = Path(path)
    if not p.is_file():
        return [{"type": "text", "text": f"Error: file not found: {path}"}]

    suffix = p.suffix.lower()
    mime = _MIME_MAP.get(suffix)
    if mime is None:
        return [{"type": "text", "text": f"Error: unsupported image format: {suffix}"}]

    data = base64.b64encode(p.read_bytes()).decode()
    return [
        {"type": "text", "text": f"Image: {p.name}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
    ]
