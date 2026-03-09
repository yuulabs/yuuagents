"""Builtin tools registry."""

from __future__ import annotations

from yuuagents.tools.bash import execute_bash
from yuuagents.tools.delegate import delegate
from yuuagents.tools.file import delete_file, edit_file, read_file, write_file
from yuuagents.tools.running import cancel_running_tool, check_running_tool
from yuuagents.tools.session_tools import (
    launch_agent,
    session_interrupt,
    session_poll,
    session_result,
)
from yuuagents.tools.read_skill import read_skill
from yuuagents.tools.skill_cli import execute_skill_cli
from yuuagents.tools.sleep import sleep
from yuuagents.tools.todo import update_todo
from yuuagents.tools.user_input import user_input
from yuuagents.tools.view_image import view_image
from yuuagents.tools.web import web_search

BUILTIN_TOOLS = {
    "execute_bash": execute_bash,
    "execute_skill_cli": execute_skill_cli,
    "read_skill": read_skill,
    "delegate": delegate,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "delete_file": delete_file,
    "user_input": user_input,
    "web_search": web_search,
    "launch_agent": launch_agent,
    "session_poll": session_poll,
    "session_interrupt": session_interrupt,
    "session_result": session_result,
    "sleep": sleep,
    "view_image": view_image,
    "check_running_tool": check_running_tool,
    "cancel_running_tool": cancel_running_tool,
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
