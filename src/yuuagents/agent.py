"""Agent configuration — immutable config for an agent instance."""

from __future__ import annotations

import yuullm
import yuutools as yt
from attrs import define


@define(frozen=True)
class AgentConfig:
    """Immutable agent configuration."""

    agent_id: str
    tools: yt.ToolManager
    llm: yuullm.YLLMClient
    system: str = ""
    tool_batch_timeout: float = 0  # seconds; 0 = no per-batch timeout
