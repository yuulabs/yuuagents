"""Control tools built on top of Basin and live flow ids."""

from __future__ import annotations

import asyncio
from typing import Any

import yuutools as yt

from yuuagents.capabilities import require_basin
from yuuagents.basin import Basin
from yuuagents.core.flow import Flow, render_agent_event


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 32:
        return text[:max_chars]
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


def _render_tree(flow: Flow[Any, Any], *, limit: int, indent: str = "") -> list[str]:
    lines = [
        f"{indent}flow_id: {flow.id}",
        f"{indent}kind: {flow.kind}",
        f"{indent}state: {flow.state.value}",
    ]
    rendered = flow.render(render_agent_event if flow.kind == "agent" else str, limit=limit)
    lines.append(f"{indent}stem:")
    lines.append(f"{indent}{rendered or '<empty>'}")
    for child in flow.children:
        lines.append(f"{indent}child:")
        lines.extend(_render_tree(child, limit=limit, indent=indent + "  "))
    return lines


@yt.tool(
    params={
        "seconds": "Seconds to wait (1-600, default 300). This blocks the current flow until done.",
    },
    description=(
        "Sleep for a while in the current flow. "
        "This is a normal long-running tool, so the host may still defer it to the background."
    ),
)
async def sleep(seconds: int = 300) -> str:
    seconds = max(1, min(600, int(seconds)))
    await asyncio.sleep(seconds)
    return f"等待 {seconds}s 完成。"


@yt.tool(
    params={
        "run_id": "Flow id of a deferred/background flow.",
        "limit": "Maximum number of recent stem events to show per flow.",
        "max_chars": "Maximum total output length in characters after rendering.",
    },
    description="Inspect a live flow tree by flow id through Basin.",
)
async def inspect_background(
    run_id: str,
    limit: int = 200,
    max_chars: int = 4000,
    basin: Basin = yt.depends(require_basin),
) -> str:
    flow = basin.require(run_id)
    return _truncate("\n".join(_render_tree(flow, limit=limit)), max_chars)


@yt.tool(
    params={
        "run_id": "Flow id of a deferred/background flow.",
    },
    description="Cancel a live flow immediately.",
)
async def cancel_background(
    run_id: str,
    basin: Basin = yt.depends(require_basin),
) -> str:
    flow = basin.require(run_id)
    flow.cancel()
    return f"Cancelled flow {run_id}"


@yt.tool(
    params={
        "run_id": "Flow id of a deferred/background flow.",
        "data": "Content to send into the flow mailbox.",
        "append_newline": "Whether to append Enter/newline after a string input. Default true.",
    },
    description="Send content into a live flow mailbox through Basin.",
)
async def input_background(
    run_id: str,
    data: str,
    append_newline: bool = True,
    basin: Basin = yt.depends(require_basin),
) -> str:
    flow = basin.require(run_id)
    flow.send(data + ("\n" if append_newline else ""))
    return f"Input sent to flow {run_id}"


@yt.tool(
    params={
        "run_id": "Flow id of a deferred/background agent flow.",
        "message": "Optional progress/defer message to send into the flow mailbox.",
    },
    description="Send a generic defer/progress nudge into a live flow.",
)
async def defer_background(
    run_id: str,
    message: str = "",
    basin: Basin = yt.depends(require_basin),
) -> str:
    flow = basin.require(run_id)
    flow.send(
        message.strip()
        or "请停止当前同步等待，把工作继续留在后台，并先汇报一条简短进展。"
    )
    return f"Deferred flow {run_id}"


@yt.tool(
    params={
        "run_ids": "List of one or more live flow ids to wait for.",
    },
    description="Wait until one or more flows finish.",
)
async def wait_background(
    run_ids: list[str],
    basin: Basin = yt.depends(require_basin),
) -> str:
    if not run_ids:
        raise RuntimeError("run_ids must not be empty")
    await asyncio.gather(*(basin.require(run_id).wait() for run_id in run_ids))
    return f"Wait finished for flows: {', '.join(run_ids)}"
