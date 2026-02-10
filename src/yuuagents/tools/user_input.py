"""user_input — block until the user replies via the daemon input endpoint."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import yuutools as yt

from yuuagents.types import AgentStatus

if TYPE_CHECKING:
    from yuuagents.agent import AgentState


@yt.tool(
    params={"prompt": "Prompt for the user"},
    description="Request user input. Blocks until input is received.",
)
async def user_input(
    prompt: str,
    state: AgentState = yt.depends(lambda ctx: ctx.state),
    input_queue: asyncio.Queue[str] = yt.depends(lambda ctx: ctx.input_queue),
) -> str:
    assert state is not None
    assert isinstance(prompt, str)
    prompt = prompt.strip()
    assert prompt

    state.pending_input_prompt = prompt
    state.status = AgentStatus.BLOCKED_ON_INPUT
    value = await input_queue.get()
    state.pending_input_prompt = ""
    state.status = AgentStatus.RUNNING
    return value
