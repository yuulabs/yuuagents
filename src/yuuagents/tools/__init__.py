"""Builtin tools registry."""

from __future__ import annotations

from yuuagents.tools.bash import execute_bash
from yuuagents.tools.control import (
    cancel_background,
    defer_background,
    input_background,
    inspect_background,
    sleep,
    wait_background,
)
from yuuagents.tools.delegate import delegate
from yuuagents.tools.file import delete_file, edit_file, read_file
from yuuagents.tools.todo import update_todo
from yuuagents.tools.view_image import view_image
from yuuagents.tools.web import web_search

BUILTIN_TOOLS = {
    "execute_bash": execute_bash,
    "inspect_background": inspect_background,
    "cancel_background": cancel_background,
    "input_background": input_background,
    "defer_background": defer_background,
    "wait_background": wait_background,
    "delegate": delegate,
    "read_file": read_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "web_search": web_search,
    "sleep": sleep,
    "view_image": view_image,
    "update_todo": update_todo,
}


def get(names: list[str]) -> list:
    """Return Tool objects for the given names.

    Raises ``KeyError`` for unknown names.
    """
    out = []
    for n in names:
        if n not in BUILTIN_TOOLS:
            raise KeyError(
                f"unknown builtin tool {n!r}; available: {list(BUILTIN_TOOLS)}"
            )
        out.append(BUILTIN_TOOLS[n])
    return out
