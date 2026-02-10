"""Builtin tools registry."""

from __future__ import annotations

from yuuagents.tools.bash import execute_bash
from yuuagents.tools.file import delete_file, read_file, write_file
from yuuagents.tools.user_input import user_input
from yuuagents.tools.web import web_search

BUILTIN_TOOLS = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
    "delete_file": delete_file,
    "user_input": user_input,
    "web_search": web_search,
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
