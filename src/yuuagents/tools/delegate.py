"""delegate — run another configured agent and return its final text response."""

from __future__ import annotations

import yuutools as yt

from yuuagents.context import (
    DelegateDepthExceededError,
    DelegateManager,
)
from yuuagents.runtime_session import Session


def _last_assistant_text(session: Session) -> str:
    for role, items in reversed(session.history):
        if role != "assistant":
            continue
        text = "".join(item for item in items if isinstance(item, str)).strip()
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
        "Combines context and task into the delegated agent's first user message. "
        "Returns the delegated agent's final text response."
    ),
)
async def delegate(
    agent: str,
    context: str,
    task: str,
    tools: list[str] | None = None,
    manager: DelegateManager | None = yt.depends(lambda ctx: ctx.manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
    parent_run_id: str = yt.depends(lambda ctx: ctx.current_run_id),
    delegate_depth: int = yt.depends(lambda ctx: ctx.delegate_depth),
) -> str:
    if manager is None:
        return "[ERROR] delegate manager unavailable"
    if parent is None:
        return "[ERROR] delegate requires an active parent session"

    next_depth = delegate_depth + 1
    if next_depth > 3:
        raise DelegateDepthExceededError(
            max_depth=3,
            current_depth=delegate_depth,
            target_agent=agent,
        )

    first_user_message = (
        f"Context:\n{context.strip()}\n\nTask:\n{task.strip()}"
        if context.strip()
        else task.strip()
    )
    child = await manager.start_delegate(
        parent=parent,
        parent_run_id=parent_run_id,
        agent=agent,
        first_user_message=first_user_message,
        tools=tools,
        delegate_depth=next_depth,
    )
    try:
        async for _step in child.step_iter():
            pass
    except BaseException:
        pass
    return _last_assistant_text(child) or child.status.value
