"""delegate — run another configured agent and return its final response."""

from __future__ import annotations

import asyncio

import yuullm
import yuutools as yt

from yuuagents.context import (
    DelegateDepthExceededError,
)
from yuuagents.capabilities import require_agent_pool
from yuuagents.pool import AgentPool
from yuuagents.input import HandoffInput
from yuuagents.runtime_session import Session
from yuuagents.types import AgentStatus


@yt.tool(
    params={
        "agent": "Agent name",
        "context": "Task context for the delegated agent",
        "task": "What the delegated agent should do",
        "tools": "Optional tool name list overriding the agent default tools",
    },
    description=(
        "Delegate work to another configured agent. "
        "Builds a structured handoff input for the delegated agent. "
        "Returns the delegated agent's final response."
    ),
)
async def delegate(
    agent: str,
    context: str,
    task: str,
    tools: list[str] | None = None,
    pool: AgentPool = yt.depends(require_agent_pool),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
    parent_run_id: str = yt.depends(lambda ctx: ctx.current_run_id),
    delegate_depth: int = yt.depends(lambda ctx: ctx.delegate_depth),
) -> list[yuullm.Item] | str:
    if parent is None:
        raise RuntimeError("delegate requires an active parent session")

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
    child = await pool.spawn(
        parent=parent,
        parent_run_id=parent_run_id,
        agent=agent,
        input=handoff_input,
        tools=tools,
        delegate_depth=next_depth,
    )
    try:
        async for _step in child.step_iter():
            pass
        child.status = AgentStatus.DONE
    except asyncio.CancelledError:
        child.status = AgentStatus.CANCELLED
        raise
    except Exception:
        child.status = AgentStatus.ERROR
    response = child.final_response()
    if response is None:
        return "[delegate] agent produced no response"
    _, items = response
    content: list[yuullm.Item] = [
        item for item in items
        if isinstance(item, dict) and item.get("type") in ("text", "image_url")
    ]
    return content if content else "[delegate] agent produced no output"
