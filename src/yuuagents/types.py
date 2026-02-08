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


class TaskRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Payload for POST /api/agents."""

    persona: str
    task: str
    tools: list[str] = []
    skills: list[str] = []
    model: str = ""
    container: str = ""
    image: str = ""


class AgentInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Summary returned by GET /api/agents and GET /api/agents/{id}."""

    agent_id: str
    persona: str
    task: str
    status: str
    created_at: str
    steps: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class SkillInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata for a discovered Agent Skill."""

    name: str
    description: str
    location: str
