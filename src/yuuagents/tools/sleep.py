"""sleep — unified wait primitive with ping collection."""

from __future__ import annotations

import asyncio
import time

import yuutools as yt

from yuuagents.flow import FlowManager, Ping, format_ping


def _format_summary(
    collected: list[Ping],
    flow_manager: FlowManager,
    elapsed: float,
    running_count: int,
) -> str:
    if not collected:
        parts = [f"等待超时（{elapsed:.0f}s）。无新通知。"]
    else:
        parts = [f"等待结束（耗时 {elapsed:.0f}s）。收到 {len(collected)} 个通知："]
        for p in collected:
            parts.append("- " + format_ping(p, flow_manager))
    if running_count > 0:
        parts.append(f"仍有 {running_count} 个子任务运行中。")
    return "\n".join(parts)


@yt.tool(
    params={
        "seconds": "Maximum seconds to wait (1-600, default 300)",
        "mode": "'all' (default): wake when all children done or timeout. "
                "'any': wake on first incoming notification.",
    },
    description=(
        "Wait for background tasks or events. "
        "Collects all notifications during the wait and returns a summary."
    ),
)
async def sleep(
    seconds: int = 300,
    mode: str = "all",
    root_flow=yt.depends(lambda ctx: ctx.root_flow),
    flow_manager=yt.depends(lambda ctx: ctx.flow_manager),
) -> str:
    seconds = max(1, min(600, int(seconds)))
    mode = mode if mode in ("all", "any") else "all"
    flow_id = root_flow.flow_id
    has_children = flow_manager.has_running_children(flow_id)

    if not has_children:
        await asyncio.sleep(seconds)
        return f"等待 {seconds}s 完成。无后台任务。"

    collected: list[Ping] = []
    start = time.monotonic()
    deadline = start + seconds

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ping = await root_flow.recv(timeout=min(remaining, 5.0))
        if ping is not None:
            collected.append(ping)
            if mode == "any":
                break
        if mode == "all" and not flow_manager.has_running_children(flow_id):
            break

    # Final non-blocking drain
    while True:
        try:
            collected.append(root_flow._ping_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    elapsed = time.monotonic() - start
    running_count = flow_manager.running_children_count(flow_id)
    return _format_summary(collected, flow_manager, elapsed, running_count)
