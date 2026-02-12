"""delegate — run another configured agent and return its final text response."""

from __future__ import annotations

import yuutools as yt

from yuuagents.context import (
    DelegateDepthExceededError,
    DelegateManager,
)


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
        "Returns only the delegated agent's final text response."
    ),
)
async def delegate(
    agent: str,
    context: str,
    task: str,
    tools: list[str] | None = None,
    manager: DelegateManager | None = yt.depends(lambda ctx: ctx.manager),
    caller_agent: str = yt.depends(lambda ctx: ctx.agent_id),
    delegate_depth: int = yt.depends(lambda ctx: ctx.delegate_depth),
) -> str:
    assert isinstance(caller_agent, str)
    caller_agent = caller_agent.strip()
    assert caller_agent
    assert isinstance(agent, str)
    agent = agent.strip()
    assert agent
    if delegate_depth >= 3:
        raise DelegateDepthExceededError(
            max_depth=3,
            current_depth=delegate_depth,
            target_agent=agent,
        )

    assert isinstance(context, str)
    assert isinstance(task, str)
    task = task.strip()
    assert task

    if tools is not None:
        assert isinstance(tools, list)
        assert all(isinstance(t, str) and t.strip() for t in tools)

    assert manager is not None

    ctx_text = context.strip()
    first_user_message = (
        task if not ctx_text else f"context:\n{ctx_text}\n\ntask:\n{task}"
    )

    return await manager.delegate(
        caller_agent=caller_agent,
        agent=agent,
        first_user_message=first_user_message,
        tools=tools,
        delegate_depth=delegate_depth + 1,
    )
