"""SDK-first local execution helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

import yuullm
import yuutools as yt
from attrs import define, field
from yuullm.types import is_text_item

from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.pool import AgentPool
from yuuagents.core.flow import AgentState
from yuuagents.input import AgentInput, conversation_input_from_text
from yuuagents.runtime_session import Session
from yuuagents.types import StepResult


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


def _last_assistant_text(messages: list[yuullm.Message]) -> str:
    for role, items in reversed(messages):
        if role != "assistant":
            continue
        text = "".join(item["text"] for item in items if is_text_item(item)).strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# LocalRunResult / LocalRun
# ---------------------------------------------------------------------------


@define(frozen=True)
class LocalRunResult:
    """Final result of a local SDK run."""

    session: Session
    input: AgentInput
    steps: tuple[StepResult, ...]
    state: AgentState
    output_text: str

    @property
    def messages(self) -> tuple[yuullm.Message, ...]:
        return self.state.messages


@define
class LocalRun:
    """Single local run over a prepared session."""

    session: Session
    input: AgentInput
    _steps: list[StepResult] = field(factory=list, init=False)
    _result: LocalRunResult | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)

    async def step_iter(self) -> AsyncGenerator[StepResult, None]:
        """Yield per-round step results for this run.

        Consume this iterator to completion before calling :meth:`result`.
        """
        if self._finished:
            for step in self._steps:
                yield step
            return
        if self._started:
            raise RuntimeError("LocalRun.step_iter() already started")
        self._started = True
        async for step in self.session.step_iter():
            self._steps.append(step)
            yield step
        self._finished = True
        self._result = await self._build_result()

    async def result(self) -> LocalRunResult:
        """Run to completion and return the final result."""
        if self._result is not None:
            return self._result
        if self._started and not self._finished:
            raise RuntimeError(
                "LocalRun.step_iter() already started; exhaust it before calling result()"
            )
        async for _ in self.step_iter():
            pass
        assert self._result is not None
        return self._result

    async def _build_result(self) -> LocalRunResult:
        state = await self.session.snapshot()
        output_text = _last_assistant_text(self.session.history)
        return LocalRunResult(
            session=self.session,
            input=self.input,
            steps=tuple(self._steps),
            state=state,
            output_text=output_text,
        )


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
) -> LocalRunResult:
    """Convenience helper for the pure SDK path.

    For multi-agent use, pass an :class:`AgentPool` as *pool*.
    """
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
        ctx = context if task_id is None else context.evolve(task_id=task_id)
    else:
        ctx = AgentContext(
            task_id=task_id or uuid4().hex,
            agent_id=agent_id,
            workdir=workdir or _default_workdir(),
            pool=actual_pool,
        )
    session = Session(config=config, context=ctx)
    session.start(agent_input)
    return await LocalRun(session=session, input=agent_input).result()
