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
from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.runtime_session import Session
from yuuagents.types import AgentInfo, AgentStatus, StepResult, TaskRequest

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
    "AgentConfig",
    "AgentContext",
    "Session",
    "AgentStatus",
    "AgentInfo",
    "StepResult",
    "TaskRequest",
    "tools",
    "init",
]
