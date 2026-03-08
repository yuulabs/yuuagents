"""Agent — the core stateful conversation entity."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Protocol

import yuullm
import yuutools as yt
from attrs import define, field

from yuuagents.types import AgentStatus, ErrorInfo


class PromptBuilder(Protocol):
    """Protocol for building system prompts."""

    def build(self) -> str:
        """Build and return the complete system prompt."""
        ...


class SimplePromptBuilder:
    """Simple prompt builder that concatenates sections."""

    def __init__(self) -> None:
        self._sections: list[str] = []

    def add_section(self, section: str) -> "SimplePromptBuilder":
        """Add a section to the prompt."""
        if section:
            self._sections.append(section)
        return self

    def build(self) -> str:
        """Build the complete prompt."""
        return "\n\n".join(self._sections)


@define(frozen=True)
class AgentConfig:
    """Immutable agent configuration — created once, never changes."""

    task_id: str
    agent_id: str
    persona: str
    tools: yt.ToolManager
    llm: yuullm.YLLMClient
    prompt_builder: PromptBuilder
    max_steps: int = 0  # 0 = unlimited
    soft_timeout: float | None = None  # seconds; None = disabled
    silence_timeout: float | None = None  # seconds; None = disabled


@define
class AgentState:
    """Mutable agent runtime state — changes during execution."""

    task: str = ""
    history: list[yuullm.Message] = field(factory=list)
    status: AgentStatus = AgentStatus.IDLE
    error: ErrorInfo | None = None
    pending_input_prompt: str = ""
    steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    created_at: datetime = field(factory=lambda: datetime.now(timezone.utc))


@define
class Agent:
    """A single agent instance with separated config and state."""

    config: AgentConfig
    state: AgentState = field(factory=AgentState)

    # Proxy properties for convenience
    @property
    def task_id(self) -> str:
        return self.config.task_id

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def persona(self) -> str:
        return self.config.persona

    @property
    def tools(self) -> yt.ToolManager:
        return self.config.tools

    @property
    def llm(self) -> yuullm.YLLMClient:
        return self.config.llm

    @property
    def task(self) -> str:
        return self.state.task

    @property
    def history(self) -> list[yuullm.Message]:
        return self.state.history

    @property
    def status(self) -> AgentStatus:
        return self.state.status

    @status.setter
    def status(self, value: AgentStatus) -> None:
        self.state.status = value

    @property
    def steps(self) -> int:
        return self.state.steps

    @steps.setter
    def steps(self, value: int) -> None:
        self.state.steps = value

    @property
    def total_tokens(self) -> int:
        return self.state.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        self.state.total_tokens = value

    @property
    def total_cost_usd(self) -> float:
        return self.state.total_cost_usd

    @total_cost_usd.setter
    def total_cost_usd(self, value: float) -> None:
        self.state.total_cost_usd = value

    @property
    def created_at(self) -> datetime:
        return self.state.created_at

    @property
    def error(self) -> ErrorInfo | None:
        return self.state.error

    @property
    def full_system_prompt(self) -> str:
        """Build the complete system prompt using the configured builder."""
        return self.config.prompt_builder.build()

    def setup(self, task: str) -> None:
        """Initialise the agent for a new task."""
        self.state.task = task
        self.state.status = AgentStatus.RUNNING
        self.state.history = [
            yuullm.system(self.full_system_prompt),
            yuullm.user(task),
        ]

    def fail(self, exc: BaseException) -> None:
        """Mark the agent as failed with detailed error information."""
        self.state.status = AgentStatus.ERROR
        self.state.error = ErrorInfo(
            message=traceback.format_exc(),
            error_type=type(exc).__name__,
            timestamp=datetime.now(timezone.utc),
        )

    @property
    def max_steps(self) -> int:
        return self.config.max_steps

    @property
    def soft_timeout(self) -> float | None:
        return self.config.soft_timeout

    @property
    def silence_timeout(self) -> float | None:
        return self.config.silence_timeout

    def done(self) -> bool:
        """Check if the agent has reached a terminal state."""
        return self.state.status in (
            AgentStatus.DONE,
            AgentStatus.ERROR,
            AgentStatus.CANCELLED,
        )
