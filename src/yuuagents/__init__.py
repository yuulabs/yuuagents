"""yuuagents — minimal agent framework: Persona + Tools + LLM."""

# Re-export yuutools (tool infrastructure)
from yuutools import (
    BoundTool,
    DependencyMarker,
    ParamSpec,
    Tool,
    ToolManager,
    ToolSpec,
    depends,
    tool,
)

# yuuagents own symbols
from yuuagents.agent import Agent
from yuuagents.context import AgentContext
from yuuagents.loop import run as run_agent
from yuuagents.types import AgentInfo, AgentStatus, SkillInfo, TaskRequest

from yuuagents import tools  # noqa: F401 — yuuagents.tools.execute_bash etc.
from yuuagents import init  # noqa: F401 — yuuagents.init.setup

__all__ = [
    # from yuutools
    "tool",
    "Tool",
    "BoundTool",
    "ToolSpec",
    "ParamSpec",
    "ToolManager",
    "depends",
    "DependencyMarker",
    # from yuuagents
    "Agent",
    "AgentContext",
    "AgentStatus",
    "AgentInfo",
    "TaskRequest",
    "SkillInfo",
    "run_agent",
    "tools",
    "init",
]
