"""Runtime session — thin wrapper around core.flow.Agent."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import yuullm
from attrs import define, field

from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.core.flow import Agent as FlowAgent
from yuuagents.types import AgentStatus, ErrorInfo


@define
class Session:
    """Host-facing runtime session wrapping a core.flow.Agent."""

    config: AgentConfig
    context: AgentContext
    task: str = ""
    history: list[yuullm.Message] = field(factory=list)
    status: AgentStatus = AgentStatus.IDLE
    error: ErrorInfo | None = None
    steps: int = 0
    total_tokens: int = 0
    last_usage: yuullm.Usage | None = None
    total_usage: yuullm.Usage | None = None
    last_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    last_input_tokens: int = 0
    created_at: datetime = field(factory=lambda: datetime.now(timezone.utc))
    mailbox_id: str = field(factory=lambda: uuid4().hex)
    _agent: FlowAgent[AgentContext] | None = field(default=None, init=False)

    @property
    def conversation_id(self) -> UUID | None:
        if self._agent is None:
            return None
        return self._agent.conversation_id_value

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def task_id(self) -> str:
        return self.context.task_id

    def start(self, task: str) -> None:
        """Create and start the underlying FlowAgent, then send the task."""
        self.task = task
        self.context = self.context.evolve(session=self)
        agent = FlowAgent(
            client=self.config.llm,
            manager=self.config.tools,
            ctx=self.context,
            system=self.config.system,
            model=self.config.llm.default_model,
            agent_name=self.config.agent_id,
        )
        agent.start()
        agent.send(task)
        self._agent = agent

    def send(self, content: str, *, defer_tools: bool = False) -> None:
        """Forward a message to the running agent."""
        if self._agent is None:
            raise RuntimeError("session not started")
        self._agent.send(content, defer_tools=defer_tools)

    def cancel(self) -> None:
        """Cancel the running agent flow."""
        if self._agent is not None:
            self._agent.flow.cancel()

    def resume(
        self,
        task: str,
        *,
        history: list[yuullm.Message],
        conversation_id: UUID | None = None,
        system: str | None = None,
    ) -> None:
        """Resume from prior history: pre-fill messages, then send a new task."""
        self.task = task
        self.context = self.context.evolve(session=self)
        agent = FlowAgent(
            client=self.config.llm,
            manager=self.config.tools,
            ctx=self.context,
            system=self.config.system if system is None else system,
            model=self.config.llm.default_model,
            agent_name=self.config.agent_id,
            conversation_id=conversation_id,
            initial_messages=list(history),
        )
        agent.start()
        agent.send(task)
        self._agent = agent

    async def wait(self) -> None:
        """Await agent completion and sync history + token counts."""
        if self._agent is None:
            raise RuntimeError("session not started")
        await self._agent.wait()
        self.history = self._agent.messages
        self.total_tokens = self._agent.total_tokens
        self.last_usage = self._agent.last_usage
        self.total_usage = self._agent.total_usage
        self.last_cost_usd = self._agent.last_cost_usd
        self.total_cost_usd = self._agent.total_cost_usd
        self.last_input_tokens = 0 if self.last_usage is None else self.last_usage.input_tokens
