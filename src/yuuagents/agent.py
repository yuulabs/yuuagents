"""Agent — the core stateful conversation entity."""

from __future__ import annotations

from datetime import datetime, timezone

import yuullm
import yuutools as yt
from attrs import define, field

from yuuagents.types import AgentStatus


@define
class Agent:
    """A single agent instance.

    ``tools`` is a :class:`yuutools.ToolManager` — yuuagents never
    re-implements tool infrastructure.
    """

    agent_id: str
    persona: str
    tools: yt.ToolManager
    llm: yuullm.YLLMClient
    task: str = ""
    history: list[yuullm.Message] = field(factory=list)
    skills_xml: str = ""
    docker_prompt: str = ""
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = field(factory=lambda: datetime.now(timezone.utc))
    steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def full_system_prompt(self) -> str:
        parts = [self.persona]
        if self.docker_prompt:
            parts.append(self.docker_prompt)
        if self.skills_xml:
            parts.append(self.skills_xml)
        return "\n\n".join(parts)

    def setup(self, task: str) -> None:
        """Initialise the agent for a new task."""
        self.task = task
        self.status = AgentStatus.RUNNING
        self.history = [
            yuullm.system(self.full_system_prompt),
            yuullm.user(task),
        ]

    def done(self) -> bool:
        return self.status in (
            AgentStatus.DONE,
            AgentStatus.ERROR,
            AgentStatus.CANCELLED,
        )
