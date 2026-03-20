"""Runtime session — thin wrapper around core.flow.Agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import attrs
import yuullm
from attrs import define, field

from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.core.flow import AgentState, Agent as FlowAgent
from yuuagents.types import AgentStatus, ErrorInfo, StepResult

if TYPE_CHECKING:
    from yuuagents.core.flow import Flow


@define
class Session:
    """Host-facing runtime session wrapping a core.flow.Agent."""

    config: AgentConfig
    context: AgentContext
    task: str = ""
    history: list[yuullm.Message] = field(factory=list)
    status: AgentStatus = AgentStatus.IDLE
    error: ErrorInfo | None = None
    stop_reason: str = ""
    created_at: datetime = field(factory=lambda: datetime.now(timezone.utc))
    mailbox_id: str = field(factory=lambda: uuid4().hex)
    stored_steps: int = 0  # for persisted sessions loaded without a live agent
    agent: FlowAgent[AgentContext] | None = field(default=None, init=False)

    # -- property delegates (read from agent when alive, zero otherwise) --

    @property
    def flow(self) -> Flow | None:
        return self.agent.flow if self.agent is not None else None

    @property
    def steps(self) -> int:
        if self.agent is not None:
            return self.agent.rounds
        return self.stored_steps

    @property
    def total_tokens(self) -> int:
        return self.agent.total_tokens if self.agent is not None else 0

    @property
    def last_usage(self) -> yuullm.Usage | None:
        return self.agent.last_usage if self.agent is not None else None

    @property
    def total_usage(self) -> yuullm.Usage | None:
        return self.agent.total_usage if self.agent is not None else None

    @property
    def last_cost_usd(self) -> float:
        return self.agent.last_cost_usd if self.agent is not None else 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.agent.total_cost_usd if self.agent is not None else 0.0

    @property
    def last_input_tokens(self) -> int:
        if self.agent is not None and self.agent.last_usage is not None:
            return self.agent.last_usage.input_tokens
        return 0

    @property
    def conversation_id(self) -> UUID | None:
        if self.agent is None:
            return None
        return self.agent.conversation_id_value

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def task_id(self) -> str:
        return self.context.task_id

    def _make_agent(
        self,
        *,
        system: str | None = None,
        conversation_id: UUID | None = None,
        initial_messages: list[yuullm.Message] | None = None,
    ) -> FlowAgent[AgentContext]:
        cfg = self.config if system is None else attrs.evolve(self.config, system=system)
        return FlowAgent(
            config=cfg,
            ctx=self.context,
            conversation_id=conversation_id,
            initial_messages=initial_messages or [],
        )

    def start(self, task: str) -> None:
        """Create the underlying FlowAgent and queue the task. Does not launch a background task."""
        self.task = task
        self.context = self.context.evolve(session=self)
        agent = self._make_agent()
        agent.start()
        agent.send_first(task)
        self.agent = agent

    def send(self, content: str, *, defer_tools: bool = False) -> None:
        """Forward a message to the running agent."""
        if self.agent is None:
            raise RuntimeError("session not started")
        self.agent.send(content, defer_tools=defer_tools)

    def cancel(self) -> None:
        """Cancel the running agent flow."""
        if self.agent is not None:
            self.agent.flow.cancel()

    def resume(
        self,
        task: str,
        *,
        history: list[yuullm.Message],
        conversation_id: UUID | None = None,
        system: str | None = None,
    ) -> None:
        """Resume from prior history: pre-fill messages, then queue a new task."""
        self.task = task
        self.context = self.context.evolve(session=self)
        agent = self._make_agent(
            system=system,
            conversation_id=conversation_id,
            initial_messages=list(history),
        )
        agent.start()
        agent.send_first(task)
        self.agent = agent

    @property
    def has_pending_background(self) -> bool:
        if self.agent is not None:
            return self.agent.has_pending_background
        return False

    async def snapshot(self, *, as_interrupted: bool = False) -> AgentState:
        """Delegate to the underlying agent's snapshot."""
        if self.agent is None:
            raise RuntimeError("session not started")
        return await self.agent.snapshot(as_interrupted=as_interrupted)

    async def kill(self) -> None:
        """Delegate to the underlying agent's kill."""
        if self.agent is not None:
            await self.agent.kill()

    async def step_iter(self) -> AsyncGenerator[StepResult, None]:
        """Host-driven step iteration. Syncs history on exit."""
        if self.agent is None:
            raise RuntimeError("session not started")
        try:
            async for step in self.agent.steps():
                yield step
        finally:
            if self.agent is not None:
                self.history = list(self.agent.messages)
