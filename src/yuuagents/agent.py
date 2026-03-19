"""Agent configuration — immutable config for an agent instance."""

from __future__ import annotations

import yuullm
import yuutools as yt
from attrs import define


def _resolve_system(
    *,
    system: str,
    system_prompt: str,
    persona: str,
) -> str:
    for value in (system, system_prompt, persona):
        if value:
            return value
    return ""


@define(frozen=True, init=False)
class AgentConfig:
    """Immutable agent configuration.

    `system` is the canonical field.
    `system_prompt` and `persona` remain as deprecated compatibility aliases.
    """

    agent_id: str
    system: str
    tools: yt.ToolManager
    llm: yuullm.YLLMClient
    max_steps: int = 0  # 0 = unlimited
    soft_timeout: float | None = None  # seconds; None = disabled
    silence_timeout: float | None = None  # seconds; None = disabled
    tool_batch_timeout: float = 0  # seconds; 0 = no per-batch timeout

    def __init__(
        self,
        *,
        agent_id: str,
        tools: yt.ToolManager,
        llm: yuullm.YLLMClient,
        system: str = "",
        system_prompt: str = "",
        persona: str = "",
        max_steps: int = 0,
        soft_timeout: float | None = None,
        silence_timeout: float | None = None,
        tool_batch_timeout: float = 0,
    ) -> None:
        resolved = _resolve_system(
            system=system,
            system_prompt=system_prompt,
            persona=persona,
        )
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "system", resolved)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "llm", llm)
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "soft_timeout", soft_timeout)
        object.__setattr__(self, "silence_timeout", silence_timeout)
        object.__setattr__(self, "tool_batch_timeout", tool_batch_timeout)

    @property
    def system_prompt(self) -> str:
        return self.system

    @property
    def persona(self) -> str:
        return self.system
