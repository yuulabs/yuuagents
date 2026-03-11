"""Tools for managing long-running tool calls (soft timeout)."""

from __future__ import annotations

import asyncio
import time

import yuutools as yt

from yuuagents.flow import FlowManager, FlowStatus


@yt.tool(
    params={
        "handle": "Handle (flow_id) returned by a timed-out tool call",
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
    flow_manager: FlowManager = yt.depends(lambda ctx: ctx.flow_manager),
) -> str:
    wait = max(1, min(wait, 3600))
    flow = flow_manager.get(handle)
    if flow is None:
        return f"[ERROR] unknown handle {handle!r}"

    if flow.status in (FlowStatus.DONE, FlowStatus.ERROR, FlowStatus.CANCELLED):
        return _format_done(flow)

    # Wait for the task to finish
    if flow.task is not None and not flow.task.done():
        try:
            await asyncio.wait_for(asyncio.shield(flow.task), timeout=wait)
        except (TimeoutError, asyncio.TimeoutError):
            elapsed = time.monotonic() - flow.started
            tail = flow._output_buffer.tail()
            tail_display = tail if tail.strip() else "<no output captured yet>"
            return (
                f"Still running ({elapsed:.0f}s since registered). handle={handle}\n"
                f"name: {flow.name or '?'}\n"
                f"tool_call_id: {flow.tool_call_id}\n"
                f"Tail output:\n{tail_display}"
            )

    return _format_done(flow)


def _format_done(flow: object) -> str:
    """Format a completed flow's result."""
    if flow.status == FlowStatus.CANCELLED:  # type: ignore[union-attr]
        return "Tool was cancelled."
    if flow.result is not None:
        if flow.result.error:  # type: ignore[union-attr]
            return f"[ERROR] {flow.result.error}"  # type: ignore[union-attr]
        return str(flow.result.output)  # type: ignore[union-attr]
    if flow.error:  # type: ignore[union-attr]
        return f"[ERROR] {flow.error}"  # type: ignore[union-attr]
    if flow.task is not None:  # type: ignore[union-attr]
        try:
            result = flow.task.result()  # type: ignore[union-attr]
            if result.error:
                return f"[ERROR] {result.error}"
            return str(result.output)
        except asyncio.CancelledError:
            return "Tool was cancelled."
        except Exception as exc:
            return f"[ERROR] {type(exc).__name__}: {exc}"
    return "[ERROR] flow completed with no result"


@yt.tool(
    params={
        "handle": "Handle (flow_id) of the running tool to cancel",
    },
    description="取消一个仍在运行的工具。",
)
async def cancel_running_tool(
    handle: str,
    flow_manager: FlowManager = yt.depends(lambda ctx: ctx.flow_manager),
) -> str:
    return flow_manager.cancel(handle)
