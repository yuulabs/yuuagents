"""delegate — run another configured agent and return its final text response."""

from __future__ import annotations

import asyncio

import yuullm
import yuutools as yt
from yuullm.types import is_text_item

from yuuagents.context import (
    DelegateDepthExceededError,
)
from yuuagents.capabilities import DelegateManager, require_delegate_manager
from yuuagents.input import HandoffInput
from yuuagents.runtime_session import Session
from yuuagents.types import AgentStatus


def _last_assistant_text(session: Session) -> str:
    for role, items in reversed(session.history):
        if role != "assistant":
            continue
        text = "".join(item["text"] for item in items if is_text_item(item)).strip()
        if text:
            return text
    return ""


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
        "Returns the delegated agent's final text response."
    ),
)
async def delegate(
    agent: str,
    context: str,
    task: str,
    tools: list[str] | None = None,
    manager: DelegateManager = yt.depends(require_delegate_manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
    parent_run_id: str = yt.depends(lambda ctx: ctx.current_run_id),
    delegate_depth: int = yt.depends(lambda ctx: ctx.delegate_depth),
) -> str:
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
    child = await manager.start_delegate(
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
    return _last_assistant_text(child) or child.status.value
