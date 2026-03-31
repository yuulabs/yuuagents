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
from yuuagents.basin import Basin
from yuuagents.capabilities import AgentCapabilities, DockerCapability, WebCapability
from yuuagents.context import AgentContext
from yuuagents.core.flow import Agent, AgentState, Flow, FlowState
from yuuagents.input import (
    AgentInput,
    ConversationInput,
    HandoffInput,
    RolloverInput,
    ScheduledInput,
)
from yuuagents.task_host import TaskHost
from yuuagents.types import AgentInfo, AgentStatus, StepResult, TaskRequest

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
    "AgentCapabilities",
    "AgentInput",
    "Agent",
    "AgentState",
    "AgentContext",
    "Basin",
    "ConversationInput",
    "DockerCapability",
    "Flow",
    "FlowState",
    "HandoffInput",
    "RolloverInput",
    "ScheduledInput",
    "AgentStatus",
    "AgentInfo",
    "StepResult",
    "TaskHost",
    "TaskRequest",
    "WebCapability",
]
