"""Tools for managing long-running tool calls (soft timeout)."""

from __future__ import annotations

import yuutools as yt

from yuuagents.running_tools import RunningToolRegistry


@yt.tool(
    params={
        "handle": "Handle returned by a timed-out tool call",
        "wait": "Max seconds to block waiting for the tool to finish (default 120)",
    },
    description=(
        "等待一个仍在运行的工具完成。工具完成时立刻返回结果，"
        "超过 wait 秒仍未完成则返回当前 tail output。"
    ),
)
async def check_running_tool(
    handle: str,
    wait: int = 120,
    registry: RunningToolRegistry = yt.depends(lambda ctx: ctx.running_tools),
) -> str:
    return await registry.check(handle, wait=max(1, min(wait, 3600)))


@yt.tool(
    params={
        "handle": "Handle of the running tool to cancel",
    },
    description="取消一个仍在运行的工具。",
)
async def cancel_running_tool(
    handle: str,
    registry: RunningToolRegistry = yt.depends(lambda ctx: ctx.running_tools),
) -> str:
    return registry.cancel(handle)
