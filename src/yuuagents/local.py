"""SDK-first local execution helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import attrs
import yuullm
import yuutools as yt
from yuullm.types import is_text_item

from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.pool import AgentPool
from yuuagents.input import AgentInput, conversation_input_from_text
from yuuagents.runtime_session import Session


def _default_workdir() -> str:
    return str(Path.cwd())


def _coerce_tools(
    tools: yt.ToolManager[Any] | Iterable[yt.Tool[Any]] | None,
) -> yt.ToolManager[Any]:
    if tools is None:
        return yt.ToolManager()
    if isinstance(tools, yt.ToolManager):
        return tools
    return yt.ToolManager(list(tools))


def _coerce_input(task: str | AgentInput) -> AgentInput:
    if isinstance(task, str):
        return conversation_input_from_text(task)
    return task


def final_response(messages: list[yuullm.Message]) -> str:
    """Return the last non-empty assistant text from a message history."""
    for role, items in reversed(messages):
        if role != "assistant":
            continue
        text = "".join(item["text"] for item in items if is_text_item(item)).strip()
        if text:
            return text
    return ""


async def run_once(
    task: str | AgentInput,
    *,
    llm: yuullm.YLLMClient,
    tools: yt.ToolManager[Any] | Iterable[yt.Tool[Any]] | None = None,
    agent_id: str = "local",
    system: str = "",
    workdir: str | None = None,
    pool: AgentPool | None = None,
    task_id: str | None = None,
    context: AgentContext | None = None,
    tool_batch_timeout: float = 0,
) -> Session:
    """Run an agent to completion and return the session."""
    agent_input = _coerce_input(task)
    actual_pool = pool if pool is not None else AgentPool()
    config = AgentConfig(
        agent_id=agent_id,
        tools=_coerce_tools(tools),
        llm=llm,
        system=system,
        tool_batch_timeout=tool_batch_timeout,
    )
    if context is not None:
        ctx = context if task_id is None else attrs.evolve(context, task_id=task_id)
    else:
        ctx = AgentContext(
            task_id=task_id or uuid4().hex,
            agent_id=agent_id,
            workdir=workdir or _default_workdir(),
            pool=actual_pool,
        )
    session = Session(config=config, context=ctx)
    session.start(agent_input)
    async for _ in session.step_iter():
        pass
    return session
