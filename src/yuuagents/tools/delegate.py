"""delegate — run another configured agent as a child flow."""

from __future__ import annotations

import asyncio
from typing import Any

import yuullm
import yuutools as yt

from yuuagents.capabilities import require_spawn_agent
from yuuagents.context import DelegateDepthExceededError
from yuuagents.core.flow import Agent, Flow, FlowState
from yuuagents.input import HandoffInput


def _content_items(message: yuullm.Message | None) -> list[yuullm.Item]:
    if message is None:
        return []
    _, items = message
    return [
        item for item in items
        if isinstance(item, dict) and item.get("type") in ("text", "image_url")
    ]


def _final_response(agent: Agent) -> yuullm.Message | None:
    for message in reversed(agent.messages):
        if message[0] == "assistant":
            return message
    return None


@yt.tool(
    params={
        "agent": "Agent name",
        "context": "Task context for the delegated agent",
        "task": "What the delegated agent should do",
        "tools": "Optional tool name list overriding the agent default tools",
    },
    description=(
        "Delegate work to another configured agent. "
        "The child agent runs as a child flow beneath the current tool flow."
    ),
)
async def delegate(
    agent: str,
    context: str,
    task: str,
    tools: list[str] | None = None,
    spawn_agent: Any = yt.depends(require_spawn_agent),
    current_flow: Flow[Any, Any] | None = yt.depends(lambda ctx: ctx.current_flow),
    delegate_depth: int = yt.depends(lambda ctx: ctx.delegate_depth),
) -> list[yuullm.Item] | str:
    if current_flow is None:
        raise RuntimeError("delegate requires an active flow")

    next_depth = delegate_depth + 1
    if next_depth > 3:
        raise DelegateDepthExceededError(
            max_depth=3,
            current_depth=delegate_depth,
            target_agent=agent,
        )

    handoff_input = HandoffInput(
        context=[yuullm.user(context.strip())] if context.strip() else [],
        task=[yuullm.user(task.strip())] if task.strip() else [],
    )
    child: Agent = await spawn_agent(
        current_flow,
        agent,
        handoff_input,
        tools,
        next_depth,
    )

    wait_task = asyncio.create_task(child.flow.wait())
    try:
        while not wait_task.done():
            while not current_flow.mailbox.empty():
                child.send(current_flow.mailbox.get_nowait())
            await asyncio.sleep(0.05)
        await wait_task
    except asyncio.CancelledError:
        child.flow.cancel()
        try:
            await child.flow.wait()
        except asyncio.CancelledError:
            pass
        raise

    if child.flow.state is FlowState.ERROR:
        raise RuntimeError(str(child.flow.info.get("error", "delegated agent failed")))
    if child.flow.state is FlowState.CANCELLED:
        raise RuntimeError("delegated agent cancelled")

    items = _content_items(_final_response(child))
    return items if items else "[delegate] agent produced no output"
