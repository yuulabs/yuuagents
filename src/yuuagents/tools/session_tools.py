"""Session tools — launch, poll, interrupt, and get results from agent sessions."""

from __future__ import annotations

import yuutools as yt

from yuuagents.context import SessionManager


@yt.tool(
    params={
        "agent": "Agent name to launch",
        "task": "Task description for the agent",
        "context": "Optional context to provide",
    },
    description=(
        "Launch a background agent session. Returns a session_id for tracking. "
        "The agent runs asynchronously — use session_poll to check progress."
    ),
)
async def launch_agent(
    agent: str,
    task: str,
    context: str = "",
    session_manager: SessionManager | None = yt.depends(lambda ctx: ctx.session_manager),
    caller_agent: str = yt.depends(lambda ctx: ctx.agent_id),
) -> str:
    assert session_manager is not None, "session_manager not configured"
    session_id = await session_manager.launch(
        caller_agent=caller_agent,
        agent=agent,
        task=task,
        context=context,
    )
    return f"Session launched: {session_id}"


@yt.tool(
    params={
        "session_id": "Session ID to check",
    },
    description="Poll a background agent session for status and progress.",
)
async def session_poll(
    session_id: str,
    session_manager: SessionManager | None = yt.depends(lambda ctx: ctx.session_manager),
) -> str:
    assert session_manager is not None, "session_manager not configured"
    info = session_manager.poll(session_id)
    parts = [f"status: {info['status']}", f"elapsed: {info['elapsed']:.0f}s"]
    progress = info.get("progress", [])
    if progress:
        last = progress[-1]
        if len(last) > 500:
            last = last[-500:]
        parts.append(f"latest:\n{last}")
    return "\n".join(parts)


@yt.tool(
    params={
        "session_id": "Session ID to interrupt",
    },
    description="Interrupt a running agent session.",
)
async def session_interrupt(
    session_id: str,
    session_manager: SessionManager | None = yt.depends(lambda ctx: ctx.session_manager),
) -> str:
    assert session_manager is not None, "session_manager not configured"
    return await session_manager.interrupt(session_id)


@yt.tool(
    params={
        "session_id": "Session ID to get result from",
    },
    description="Get the final result from a completed agent session.",
)
async def session_result(
    session_id: str,
    session_manager: SessionManager | None = yt.depends(lambda ctx: ctx.session_manager),
) -> str:
    assert session_manager is not None, "session_manager not configured"
    result = session_manager.result(session_id)
    if result is None:
        return "Session not yet complete or no result available."
    return result
