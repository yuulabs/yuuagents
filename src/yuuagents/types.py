"""API data-transfer types for yuuagents."""

from __future__ import annotations

import enum
from datetime import datetime

import msgspec


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


class TaskRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Payload for POST /api/agents.

    ``agent`` selects a named agent configuration (provider, model, tools,
    skills, persona).  Defaults to ``"main"``.

    ``persona`` optionally overrides the system prompt from the agent config.
    If empty, the persona from the agent config is used.  If no agent config
    matches, ``persona`` is used as-is.
    """

    agent: str = "main"
    persona: str = ""
    task: str
    tools: list[str] = msgspec.field(default_factory=list)
    skills: list[str] = msgspec.field(default_factory=list)
    model: str = ""
    container: str = ""
    image: str = ""


class AgentInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Summary returned by GET /api/agents and GET /api/agents/{id}."""

    task_id: str
    agent_id: str
    persona: str
    task: str
    status: str
    created_at: str
    last_assistant_message: str = ""
    pending_input_prompt: str = ""
    steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error: ErrorInfo | None = None


class SkillInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata for a discovered Agent Skill."""

    name: str
    description: str
    location: str
