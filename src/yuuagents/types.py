"""API data-transfer types for yuuagents."""

from __future__ import annotations

import enum
from datetime import datetime

import attrs
import msgspec
import yuullm

from yuuagents.input import AgentInput, agent_input_from_jsonable


@attrs.define(frozen=True)
class StepResult:
    """Result yielded by Agent.steps() after each LLM round."""

    done: bool          # True = LLM finished naturally (no tool calls)
    tokens: int = 0     # cumulative total tokens so far
    rounds: int = 0     # cumulative LLM rounds so far
    delta: tuple[yuullm.Message, ...] = ()  # messages appended this step


class AgentStatus(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    BLOCKED_ON_INPUT = "blocked_on_input"
    CANCELLED = "cancelled"


class ErrorInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Structured error information with full stack trace in message.

    The message field contains the complete error message including stack trace
    for debugging purposes.
    """

    message: str  # Full error message with stack trace
    error_type: str
    timestamp: datetime


@attrs.define(frozen=True, kw_only=True)
class TaskRequest:
    """Payload for POST /api/agents.

    ``agent`` selects a named agent configuration (provider, model, tools,
    persona).  Defaults to ``"main"``.

    ``persona`` optionally overrides the system prompt from the agent config.
    If empty, the persona from the agent config is used.  If no agent config
    matches, ``persona`` is used as-is.
    """

    agent: str = "main"
    persona: str = ""
    input: AgentInput
    tools: list[str] = attrs.field(factory=list)
    model: str = ""
    container: str = ""
    image: str = ""

    @classmethod
    def from_jsonable(cls, value: object) -> TaskRequest:
        if not isinstance(value, dict):
            raise TypeError("task request must be an object")
        agent = value.get("agent", "main")
        persona = value.get("persona", "")
        tools = value.get("tools", [])
        model = value.get("model", "")
        container = value.get("container", "")
        image = value.get("image", "")
        if not isinstance(agent, str):
            raise TypeError("agent must be a string")
        if not isinstance(persona, str):
            raise TypeError("persona must be a string")
        if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
            raise TypeError("tools must be a list of strings")
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        if not isinstance(container, str):
            raise TypeError("container must be a string")
        if not isinstance(image, str):
            raise TypeError("image must be a string")
        return cls(
            agent=agent,
            persona=persona,
            input=agent_input_from_jsonable(value.get("input")),
            tools=list(tools),
            model=model,
            container=container,
            image=image,
        )


class AgentInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Summary returned by GET /api/agents and GET /api/agents/{id}."""

    task_id: str
    agent_id: str
    persona: str
    input_kind: str
    input_preview: str
    status: str
    created_at: str
    last_assistant_message: str = ""
    pending_input_prompt: str = ""
    steps: int = 0
    total_tokens: int = 0
    last_usage: yuullm.Usage | None = None
    total_usage: yuullm.Usage | None = None
    last_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    error: ErrorInfo | None = None
