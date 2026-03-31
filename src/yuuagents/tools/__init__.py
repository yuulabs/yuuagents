"""Lazy builtin tool registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_TOOL_IMPORTS = {
    "execute_bash": ("yuuagents.tools.bash", "execute_bash"),
    "inspect_background": ("yuuagents.tools.control", "inspect_background"),
    "cancel_background": ("yuuagents.tools.control", "cancel_background"),
    "input_background": ("yuuagents.tools.control", "input_background"),
    "defer_background": ("yuuagents.tools.control", "defer_background"),
    "wait_background": ("yuuagents.tools.control", "wait_background"),
    "delegate": ("yuuagents.tools.delegate", "delegate"),
    "read_file": ("yuuagents.tools.file", "read_file"),
    "edit_file": ("yuuagents.tools.file", "edit_file"),
    "delete_file": ("yuuagents.tools.file", "delete_file"),
    "web_search": ("yuuagents.tools.web", "web_search"),
    "sleep": ("yuuagents.tools.control", "sleep"),
    "view_image": ("yuuagents.tools.view_image", "view_image"),
}

TOOL_NAMES = tuple(_TOOL_IMPORTS)


def _load_tool(name: str) -> Any:
    try:
        module_name, attr_name = _TOOL_IMPORTS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown builtin tool {name!r}; available: {list(_TOOL_IMPORTS)}"
        ) from exc
    module = import_module(module_name)
    return getattr(module, attr_name)


def get(names: list[str]) -> list[Any]:
    """Return Tool objects for the given names."""
    return [_load_tool(name) for name in names]
